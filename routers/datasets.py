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


@router.post("/from-document", summary="根据文档生成数据集（异步）")
async def generate_from_document(
        document: UploadFile = File(..., description="上传文档，支持 .txt/.docx/.pdf"),
        num_questions: int = Form(10, description="生成的问题数量，默认10"),
):
    logger.info(f"文档生成请求: 文件名={document.filename}, 问题数量={num_questions}")

    # 基本校验
    if num_questions < 1 or num_questions > 100:
        raise HTTPException(status_code=400, detail="生成的问题数量需在 1~100 之间")
    if not document or not document.filename:
        raise HTTPException(status_code=400, detail="未选择上传文件")

    # 校验类型与大小
    suffix = os.path.splitext(document.filename)[1].lower().lstrip('.')
    if suffix not in FILE_CONFIG['allowed_file_types']:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: .{suffix}")

    content_bytes = await document.read()
    if not content_bytes:
        logger.error("上传的文件为空")
        raise HTTPException(status_code=400, detail="上传的文件为空")
    if len(content_bytes) > FILE_CONFIG['max_file_size']:
        raise HTTPException(status_code=400, detail="文件过大")

    # 生成任务名称与输入摘要
    task_name = generate_task_name("document")
    input_content = f"文件名: {document.filename}; 大小: {len(content_bytes)} bytes"

    task_id = task_manager.create_task(
        task_name=task_name,
        generation_type="document",
        total_items=num_questions
    )
    if not task_id:
        raise HTTPException(status_code=500, detail="创建任务失败")

    # 将文件内容与后续参数交给异步 worker
    asyncio.create_task(process_document_generation(task_id, document.filename, content_bytes, suffix, num_questions))

    return {
        "task_id": task_id,
        "task_name": task_name,
        "status": "pending",
        "message": "任务已创建，正在异步处理"
    }


async def process_document_generation(task_id: int, filename: str, content_bytes: bytes, suffix: str, num_questions: int):
    logger.info(f"开始处理文档生成任务: ID={task_id}, 文件={filename}")
    try:
        task_manager.update_task_status(task_id, "running")

        # 保存到临时目录
        temp_dir = os.path.join(FILE_CONFIG['output_dir'], 'tmp_docs')
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(temp_dir, f"upload_{timestamp}.{suffix}")
        with open(temp_path, 'wb') as f:
            f.write(content_bytes)

        # 进度回调（文档路径只有一份，按完成条数更新）
        def progress_callback(completed: int, total: int, status: str):
            try:
                task_manager.update_task_status(task_id, "running", completed_items=completed)
            except Exception as e:
                logger.error(f"文档任务进度更新失败: {str(e)}")

        qa_items = await deepeval_generator.generate_from_documents(
            document_paths=[temp_path],
            num_questions=num_questions,
            scenario="educational"
        )
        if not qa_items:
            raise Exception("未生成任何问答对")

        # 转换为QAItem
        final_items: List[QAItem] = []
        for item in qa_items[:num_questions]:
            final_items.append(QAItem(
                question=item['question'],
                expected_output=item['expected_output'],
                context=item.get('context', []),
                context_length=item.get('context_length', 0)
            ))

        # 保存CSV
        csv_content = write_qa_to_csv(final_items)
        out_filename = f"document_qa_dataset_{timestamp}.csv"
        output_path = f"{FILE_CONFIG['output_dir']}/{out_filename}"
        os.makedirs(FILE_CONFIG['output_dir'], exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)

        task_manager.update_task_status(
            task_id,
            "completed",
            completed_items=len(final_items),
            output_file_path=output_path,
            preview=( (final_items and ((final_items[0].question + ' ' + final_items[0].expected_output)[:50] + ('' if len(final_items[0].question + ' ' + final_items[0].expected_output) <= 50 else '...'))) or '' )
        )
        logger.info(f"文档任务 {task_id} 完成，生成 {len(final_items)} 条，文件: {output_path}")
    except Exception as e:
        logger.error(f"文档任务 {task_id} 失败: {str(e)}")
        task_manager.update_task_status(task_id, "failed", error_message=str(e))


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

        # 创建任务
        task_id = task_manager.create_task(
            task_name=task_name,
            generation_type="context",
            total_items=payload.num_questions
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
        preview_text = ""
        if final_items:
            first = final_items[0]
            combined = f"{first.question} {first.expected_output}"
            preview_text = (combined if len(combined) <= 50 else combined[:50] + "...")
        task_manager.update_task_status(
            task_id, 
            "completed", 
            completed_items=len(final_items),
            output_file_path=output_path,
            preview=preview_text
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


@router.post("/from-topic", summary="根据主题生成数据集（同步）")
async def generate_from_topic(payload: FromTopicRequest):
    logger.info(f"主题生成请求: 主题={payload.topic}, 数量={payload.num_questions}")

    # 校验数量
    if payload.num_questions < 1 or payload.num_questions > 200:
        raise HTTPException(status_code=400, detail="期望生成的问题数量需在 1~200 之间")

    try:
        # 直接调用生成方法，不使用异步任务
        qa_items = await deepeval_generator.generate_from_scratch(
            num_questions=payload.num_questions,
            scenario="educational",
            topic=payload.topic,  # 传递主题信息
            progress_callback=None  # 不使用进度回调
        )

        if not qa_items:
            raise Exception("没有生成任何问答对")

        # 转换为QAItem
        final_items: List[QAItem] = []
        for item in qa_items:
            final_items.append(QAItem(
                question=item['question'],
                expected_output=item['expected_output'],
                context=[f"主题: {payload.topic}"],  # 将主题作为上下文
                context_length=len(payload.topic)
            ))

        # 保存CSV
        csv_content = write_qa_to_csv(final_items)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"topic_qa_dataset_{timestamp}.csv"
        output_path = f"{FILE_CONFIG['output_dir']}/{filename}"
        os.makedirs(FILE_CONFIG['output_dir'], exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)

        # 生成预览
        preview_text = ""
        if final_items:
            first = final_items[0]
            combined = f"{first.question} {first.expected_output}"
            preview_text = (combined if len(combined) <= 50 else combined[:50] + "...")

        logger.info(f"主题生成完成，生成了 {len(final_items)} 个问答对，文件保存至: {output_path}")

        return {
            "status": "success",
            "message": f"成功生成 {len(final_items)} 个问答对",
            "data": {
                "total_items": len(final_items),
                "output_file_path": output_path,
                "preview": preview_text,
                "qa_items": [item.dict() for item in final_items]
            }
        }

    except Exception as e:
        logger.error(f"主题生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.post("/augment", summary="根据数据集扩写数据集（异步）")
async def augment_dataset(
        dataset: UploadFile = File(..., description="上传已有数据集CSV，包含 input,expected_output 列"),
        ratio: float = Form(1.0, description="生成比例，默认1.0"),
):
    logger.info(f"数据集扩写请求: 文件名={dataset.filename}, 比例={ratio}")

    # 校验文件
    if not dataset or not dataset.filename:
        raise HTTPException(status_code=400, detail="未选择上传文件")
    suffix = os.path.splitext(dataset.filename)[1].lower()
    if suffix != '.csv':
        raise HTTPException(status_code=400, detail="仅支持 CSV 文件")

    content_bytes = await dataset.read()
    if not content_bytes:
        logger.error("上传的文件为空")
        raise HTTPException(status_code=400, detail="上传的文件为空")

    try:
        text = content_bytes.decode('utf-8')
    except Exception:
        try:
            text = content_bytes.decode('utf-8-sig')
        except Exception:
            raise HTTPException(status_code=400, detail="CSV 编码错误，请使用 UTF-8")

    # 解析CSV
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader if row]
    if not rows:
        raise HTTPException(status_code=400, detail="CSV 文件无内容")
    if 'input' not in reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 缺少 input 列")
    # expected_output 可选

    # 计算目标生成数量
    base_count = len(rows)
    if ratio <= 0:
        ratio = 1.0
    target_num = max(1, int(base_count * ratio))

    # 汇总上下文（将每条 input 与其 expected_output 合并为一条上下文）
    contexts: List[str] = []
    for row in rows:
        parts = [row.get('input', '').strip()]
        eo = row.get('expected_output')
        if eo:
            parts.append(eo.strip())
        contexts.append("\n".join([p for p in parts if p]))
    # 若上下文过多，仅取前若干有代表性的
    if len(contexts) > 20:
        contexts = contexts[:20]

    # 创建任务
    task_name = generate_task_name("augment")
    input_content = f"原始样本: {base_count}; 生成比例: {ratio}; 目标数量: {target_num}"
    task_id = task_manager.create_task(
        task_name=task_name,
        generation_type="augment",
        total_items=target_num
    )
    if not task_id:
        raise HTTPException(status_code=500, detail="创建任务失败")

    # 启动异步处理
    asyncio.create_task(process_augment_generation(task_id, contexts, target_num))

    return {
        "task_id": task_id,
        "task_name": task_name,
        "status": "pending",
        "message": "任务已创建，正在异步处理"
    }


async def process_augment_generation(task_id: int, contexts: List[str], target_num: int):
    logger.info(f"开始处理扩写任务: ID={task_id}, 上下文数={len(contexts)}, 目标数量={target_num}")
    try:
        task_manager.update_task_status(task_id, "running")

        # 将上下文转换为Golden对象
        from deepeval.test_case import Golden
            
        goldens = []
        for context in contexts:
            # 简单地将context作为input，创建一个基础的Golden
            golden = Golden(input=context)
            goldens.append(golden)
        
        # 使用DeepEval的generate_goldens_from_goldens方法进行数据集扩写
        qa_items = await deepeval_generator.generate_from_goldens(
            goldens=goldens,
            num_questions=target_num,
            scenario="educational"
        )
        if not qa_items:
            raise Exception("未生成任何扩写数据")

        # 转换为扩写项（将生成的 question/expected_output 映射到 input/expected_output）
        augment_items: List[AugmentDatasetResponseItem] = []
        for item in qa_items:
            augment_items.append(AugmentDatasetResponseItem(
                input=item['question'],
                expected_output=item.get('expected_output')
            ))

        # 保存CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"augment_dataset_{timestamp}.csv"
        output_path = f"{FILE_CONFIG['output_dir']}/{filename}"
        os.makedirs(FILE_CONFIG['output_dir'], exist_ok=True)
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'expected_output'])
            for it in augment_items:
                writer.writerow([it.input, it.expected_output or ''])

        task_manager.update_task_status(
            task_id,
            "completed",
            completed_items=len(augment_items),
            output_file_path=output_path,
            preview=(augment_items and ( (augment_items[0].input + ' ' + (augment_items[0].expected_output or ''))[:50] + ('' if len(augment_items[0].input + ' ' + (augment_items[0].expected_output or '')) <= 50 else '...') ) or '')
        )
        logger.info(f"扩写任务 {task_id} 完成，生成 {len(augment_items)} 条，文件: {output_path}")
    except Exception as e:
        logger.error(f"扩写任务 {task_id} 失败: {str(e)}")
        task_manager.update_task_status(task_id, "failed", error_message=str(e))


@router.get("/template", summary="下载扩写模板CSV")
async def download_template():
    logger.info("下载扩写模板")
    content = "input,expected_output\n\"问题示例1\",\"答案示例1\"\n\"问题示例2\",\"答案示例2\"\n"
    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=template.csv"},
    )
