from typing import List
import os
import asyncio
import csv
import io
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from database import TaskManager
from datetime import datetime
from config import FILE_CONFIG
from utils.deepeval_generator import deepeval_generator
from utils.logger import get_logger
from schemas import (
    QAItem,
    FromContextRequest,
    FromTopicRequest,
    TopicDatasetResponseItem,
    AugmentDatasetResponseItem,
    QAResponse,
    TopicResponse,
    AugmentResponse,
)

router = APIRouter(prefix="/datasets", tags=["Datasets"])

# 获取日志记录器
logger = get_logger('datasets')

# 创建任务管理器实例
task_manager = TaskManager()

def generate_task_name(generation_type: str) -> str:
    """生成任务名称"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"generation_datasets_from_{generation_type}_{timestamp}"

def write_qa_to_csv(items: List[QAItem]) -> str:
    """将问答对写入CSV格式"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # 写入表头
    writer.writerow(['question', 'expected_output', 'context', 'context_length'])
    
    # 写入数据
    for item in items:
        writer.writerow([
            item.question,
            item.expected_output,
            ';'.join(item.context) if item.context else '',
            item.context_length
        ])
    
    return output.getvalue()


@router.post("/from-document", response_model=QAResponse, summary="根据文档生成数据集")
async def generate_from_document(
        document: UploadFile = File(..., description="上传文档，支持 .txt/.docx/.pdf"),
        num_questions: int = Form(10, description="生成的问题数量，默认10"),
):
    logger.info(f"文档生成请求: 文件名={document.filename}, 问题数量={num_questions}")

    # 这里只做演示：读取文件名与长度，实际应替换为真实生成逻辑
    content_bytes = await document.read()
    if len(content_bytes) == 0:
        logger.error("上传的文件为空")
        raise HTTPException(status_code=400, detail="上传的文件为空")

    preview = document.filename or "document"
    dummy = QAItem(
        question=f"基于 {preview} 生成的问题示例?",
        expected_output="这是一个示例答案",
        context=[f"文件名: {preview}", f"字节数: {len(content_bytes)}"],
        context_length=len(content_bytes),
    )
    items: List[QAItem] = [dummy for _ in range(max(1, num_questions))]

    logger.info(f"文档生成完成，生成了 {len(items)} 个问答对")
    return QAResponse(items=items)


@router.post("/from-context", summary="根据上下文生成数据集（使用DeepEval）")
async def generate_from_context(payload: FromContextRequest):
    """
    使用DeepEval基于提供的上下文信息生成问答对，创建异步任务
    """
    logger.info(f"上下文生成请求: 上下文数量={len(payload.contexts)}, 问题数量={payload.num_questions}")

    # 验证输入
    if not payload.contexts or len(payload.contexts) == 0:
        logger.error("上下文信息为空")
        raise HTTPException(status_code=400, detail="上下文信息不能为空")

    if payload.num_questions <= 0:
        logger.error("生成的问题数量必须大于0")
        raise HTTPException(status_code=400, detail="生成的问题数量必须大于0")

    try:
        # 生成任务名称
        task_name = generate_task_name("context")

        # 准备输入内容
        input_content = f"上下文数量: {len(payload.contexts)}\n上下文内容:\n" + "\n".join([f"- {ctx[:100]}{'...' if len(ctx) > 100 else ''}" for ctx in payload.contexts])
        
        # 创建任务
        task_id = task_manager.create_task(
            task_name=task_name,
            generation_type="context",
            total_items=payload.num_questions,
            input_content=input_content
        )

        if not task_id:
            logger.error("创建任务失败")
            raise HTTPException(status_code=500, detail="创建任务失败")

        # 启动异步任务
        asyncio.create_task(process_context_generation(task_id, payload))

        logger.info(f"上下文生成任务已创建: ID={task_id}, 名称={task_name}")

        return {
            "task_id": task_id,
            "task_name": task_name,
            "status": "pending",
            "message": "任务已创建，正在异步处理"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


async def process_context_generation(task_id: int, payload: FromContextRequest):
    """
    异步处理上下文生成任务，使用DeepEval生成数据集
    """
    logger.info(f"开始处理上下文生成任务: ID={task_id}")

    try:
        # 更新任务状态为运行中
        task_manager.update_task_status(task_id, "running")

        logger.info(f"使用DeepEval生成数据集: 上下文数量={len(payload.contexts)}, 问题数量={payload.num_questions}")

        # 定义进度回调函数
        def progress_callback(completed: int, total: int, status: str):
            """进度回调函数，更新数据库中的任务进度"""
            try:
                task_manager.update_task_status(
                    task_id, 
                    "running", 
                    completed_items=completed
                )
                logger.info(f"任务 {task_id} 进度: {completed}/{total} ({completed/total*100:.1f}%) - {status}")
            except Exception as e:
                logger.error(f"更新任务进度失败: {str(e)}")
        
        # 使用DeepEval生成数据集，带进度跟踪
        qa_items = await deepeval_generator.generate_from_contexts(
            contexts=payload.contexts,
            num_questions=payload.num_questions,
            scenario="educational",
            progress_callback=progress_callback
        )

        # 检查DeepEval生成结果
        if not qa_items:
            raise Exception("DeepEval没有成功生成任何问答对")

        logger.info(f"DeepEval生成完成，总共生成了 {len(qa_items)} 个问答对")

        # 转换为QAItem格式
        final_items = []
        for item in qa_items:
            qa_item = QAItem(
                question=item['question'],
                expected_output=item['expected_output'],
                context=item['context'],
                context_length=item['context_length']
            )
            final_items.append(qa_item)

        # 生成CSV文件
        csv_content = write_qa_to_csv(final_items)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"context_qa_dataset_{timestamp}.csv"
        output_path = f"{FILE_CONFIG['output_dir']}/{filename}"

        # 确保输出目录存在
        os.makedirs(FILE_CONFIG['output_dir'], exist_ok=True)

        # 保存CSV文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)

        # 更新任务状态为完成
        task_manager.update_task_status(
            task_id, 
            "completed", 
            completed_items=len(final_items),
            output_file_path=output_path
        )

        logger.info(f"任务 {task_id} 完成，生成了 {len(final_items)} 个问答对，文件保存至: {output_path}")

    except Exception as e:
        # 如果处理失败，更新任务状态为失败
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        task_manager.update_task_status(
            task_id,
            "failed",
            error_message=f"任务处理失败: {str(e)}"
        )


@router.post("/from-topic", response_model=TopicResponse, summary="根据主题生成数据集")
async def generate_from_topic(payload: FromTopicRequest):
    logger.info(f"主题生成请求: 主题={payload.topic}, 问题数量={payload.num_questions}")

    base_q = f"围绕主题『{payload.topic}』给出一个典型问题?"
    items = [TopicDatasetResponseItem(input=base_q) for _ in range(max(1, payload.num_questions))]

    logger.info(f"主题生成完成，生成了 {len(items)} 个问题")
    return TopicResponse(items=items)


@router.post("/augment", response_model=AugmentResponse, summary="根据数据集扩写数据集")
async def augment_dataset(
        dataset: UploadFile = File(..., description="上传已有数据集CSV，包含 input,expected_output 列"),
        ratio: float = Form(1.0, description="生成比例，默认1.0"),
):
    logger.info(f"数据集扩写请求: 文件名={dataset.filename}, 比例={ratio}")

    content = await dataset.read()
    if not content:
        logger.error("上传的文件为空")
        raise HTTPException(status_code=400, detail="上传的文件为空")

    # 仅演示：返回两条示例扩写
    items = [
        AugmentDatasetResponseItem(input="示例问题1", expected_output="示例答案1"),
        AugmentDatasetResponseItem(input="示例问题2", expected_output="示例答案2"),
    ]

    logger.info(f"数据集扩写完成，生成了 {len(items)} 个扩写项")
    return AugmentResponse(items=items)


@router.get("/template", summary="下载扩写模板CSV")
async def download_template():
    logger.info("下载扩写模板")
    content = "input,expected_output\n\"问题示例1\",\"答案示例1\"\n\"问题示例2\",\"答案示例2\"\n"
    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=template.csv"},
    )
