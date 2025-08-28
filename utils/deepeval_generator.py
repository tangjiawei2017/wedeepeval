import os
import sys
import asyncio
import logging
from typing import List, Dict, Optional, Callable
from datetime import datetime

# 获取项目根目录的绝对路径并添加 DeepEval 源码路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPEVAL_SOURCE_PATH = os.path.join(PROJECT_ROOT, 'deepeval_source')

if DEEPEVAL_SOURCE_PATH not in sys.path:
    sys.path.insert(0, DEEPEVAL_SOURCE_PATH)
    print(f"🔧 已添加本地 DeepEval 源码路径: {DEEPEVAL_SOURCE_PATH}")

# 验证导入路径
try:
    import deepeval
    print(f"✅ 成功导入 DeepEval，路径: {deepeval.__file__}")
    if 'deepeval_source' in deepeval.__file__:
        print("🎯 使用的是本地 DeepEval 源码")
    else:
        print("⚠️  警告：使用的是系统安装的 DeepEval 包")
except ImportError as e:
    print(f"❌ DeepEval 导入失败: {e}")
    raise

# 导入所需的模块
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset
from utils.logger import get_logger
from config import API_CONFIG

# 获取日志记录器
logger = get_logger('deepeval_generator', 'business')

class DeepEvalDatasetGenerator:
    """使用DeepEval生成数据集的工具类"""
    
    def __init__(self):
        """初始化DeepEval数据集生成器"""
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = API_CONFIG['openai_api_key']
        os.environ["OPENAI_BASE_URL"] = API_CONFIG['openai_base_url']
        
        # 初始化Synthesizer，使用更保守的配置
        self.synthesizer = Synthesizer(
            model=API_CONFIG['openai_model'],
            async_mode=True,
            max_concurrent=1,  # 减少到1个并发，避免过载
            cost_tracking=False
        )
        
        # 设置模型超时和重试参数
        if hasattr(self.synthesizer.model, 'timeout'):
            self.synthesizer.model.timeout = 30  # 30秒超时
        if hasattr(self.synthesizer.model, 'max_retries'):
            self.synthesizer.model.max_retries = 2  # 最多重试2次
        
        logger.info(f"DeepEval数据集生成器初始化完成，使用模型: {API_CONFIG['openai_model']}")
        logger.info(f"API配置: {API_CONFIG['openai_base_url']}")
        logger.info(f"并发数: 1, 超时: 30秒, 重试: 2次")
    
    async def generate_from_contexts(self, num_questions: int, contexts: List[str], progress_callback=None) -> List[Dict]:
        """基于上下文生成数据集"""
        logger.info(f"开始使用DeepEval基于上下文生成数据集: 问题数量={num_questions}, 上下文数量={len(contexts)}")
        
        if progress_callback:
            progress_callback(0, num_questions, "开始全量生成...")
        
        try:
            # 设置上下文
            self.synthesizer.contexts = contexts
            
            if progress_callback:
                progress_callback(1, num_questions, "20%")
            
            # 使用DeepEval生成数据集
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_contexts(
                        num_goldens=num_questions
                    )
                    break
                except Exception as e:
                    logger.error(f"DeepEval上下文生成失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)  # 等待1秒后重试
            
            if progress_callback:
                progress_callback(2, num_questions, "40%")
            
            # 处理生成结果
            qa_items = []
            for item in dataset:
                qa_items.append({
                    'question': item.question,
                    'answer': item.answer,
                    'context': item.context if hasattr(item, 'context') else ""
                })
            
            if progress_callback:
                progress_callback(3, num_questions, "60%")
            
            logger.info(f"DeepEval上下文生成完成，共生成 {len(qa_items)} 个中文问答对")
            
            if progress_callback:
                progress_callback(4, num_questions, "80%")
            
            if progress_callback:
                progress_callback(len(qa_items), num_questions, "100% - 生成完成")
            
            return qa_items
            
        except Exception as e:
            logger.error(f"上下文生成数据集失败: {str(e)}")
            raise e

    async def generate_from_documents(self, num_questions: int, documents: List[str], progress_callback=None) -> List[Dict]:
        """基于文档生成数据集"""
        logger.info(f"开始使用DeepEval基于文档生成数据集: 问题数量={num_questions}, 文档数量={len(documents)}")
        
        if progress_callback:
            progress_callback(0, num_questions, "开始从文档生成...")
        
        try:
            # 设置文档作为上下文
            self.synthesizer.contexts = documents
            
            if progress_callback:
                progress_callback(1, num_questions, "20%")
            
            # 使用DeepEval生成数据集
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_documents(
                        num_goldens=num_questions
                    )
                    break
                except Exception as e:
                    logger.error(f"DeepEval文档生成失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)  # 等待1秒后重试
            
            if progress_callback:
                progress_callback(2, num_questions, "40%")
            
            # 处理生成结果
            qa_items = []
            for item in dataset:
                qa_items.append({
                    'question': item.question,
                    'answer': item.answer,
                    'context': item.context if hasattr(item, 'context') else ""
                })
            
            if progress_callback:
                progress_callback(3, num_questions, "60%")
            
            logger.info(f"DeepEval文档生成完成，共生成 {len(qa_items)} 个中文问答对")
            
            if progress_callback:
                progress_callback(4, num_questions, "80%")
            
            if progress_callback:
                progress_callback(len(qa_items), num_questions, "100% - 生成完成")
            
            return qa_items
            
        except Exception as e:
            logger.error(f"文档生成数据集失败: {str(e)}")
            raise e
    
    async def generate_from_scratch(self, num_questions: int, topic: str, task_description: str, scenario_description: str, progress_callback=None) -> List[Dict]:
        """从零开始生成数据集"""
        logger.info(f"开始使用DeepEval从零生成数据集: 问题数量={num_questions}, 主题={topic}, 任务描述={task_description}, 场景描述={scenario_description}")
        
        if progress_callback:
            progress_callback(0, num_questions, "开始从零生成...")
        
        try:
            # 配置生成参数
            self.synthesizer.contexts = [f"主题: {topic}\n任务描述: {task_description}\n场景描述: {scenario_description}"]
            
            if progress_callback:
                progress_callback(1, num_questions, "20%")
            
            # 使用DeepEval生成数据集
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_scratch(
                        num_goldens=num_questions
                    )
                    break
                except Exception as e:
                    logger.error(f"DeepEval生成失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)  # 等待1秒后重试
            
            if progress_callback:
                progress_callback(2, num_questions, "40%")
            
            # 处理生成结果
            qa_items = []
            for item in dataset:
                qa_items.append({
                    'question': item.question,
                    'answer': item.answer,
                    'context': item.context if hasattr(item, 'context') else ""
                })
            
            if progress_callback:
                progress_callback(3, num_questions, "60%")
            
            logger.info(f"DeepEval从零生成完成，共生成 {len(qa_items)} 个中文问答对")
            
            if progress_callback:
                progress_callback(4, num_questions, "80%")
            
            if progress_callback:
                progress_callback(len(qa_items), num_questions, "100% - 生成完成")
            
            return qa_items
            
        except Exception as e:
            logger.error(f"从零生成数据集失败: {str(e)}")
            raise e
    
    async def generate_from_goldens(self, num_questions: int, goldens: List[Dict], progress_callback=None) -> List[Dict]:
        """基于现有问答对扩写生成数据集"""
        logger.info(f"开始使用DeepEval扩写生成数据集: 问题数量={num_questions}, 现有问答对数量={len(goldens)}")
        
        if progress_callback:
            progress_callback(0, num_questions, "开始扩写生成...")
        
        try:
            # 准备现有问答对作为上下文
            contexts = []
            for golden in goldens:
                context = f"问题: {golden.get('question', '')}\n答案: {golden.get('answer', '')}"
                contexts.append(context)
            
            self.synthesizer.contexts = contexts
            
            if progress_callback:
                progress_callback(1, num_questions, "20%")
            
            # 使用DeepEval生成数据集
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_goldens(
                        num_goldens=num_questions
                    )
                    break
                except Exception as e:
                    logger.error(f"DeepEval扩写生成失败，尝试 {attempt + 1}/{max_retries}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(1)  # 等待1秒后重试
            
            if progress_callback:
                progress_callback(2, num_questions, "40%")
            
            # 处理生成结果
            qa_items = []
            for item in dataset:
                qa_items.append({
                    'question': item.question,
                    'answer': item.answer,
                    'context': item.context if hasattr(item, 'context') else ""
                })
            
            if progress_callback:
                progress_callback(3, num_questions, "60%")
            
            logger.info(f"DeepEval扩写生成完成，共生成 {len(qa_items)} 个中文问答对")
            
            if progress_callback:
                progress_callback(4, num_questions, "80%")
            
            if progress_callback:
                progress_callback(len(qa_items), num_questions, "100% - 生成完成")
            
            return qa_items
            
        except Exception as e:
            logger.error(f"扩写生成数据集失败: {str(e)}")
            raise e
    
    def save_dataset(self, qa_items: List[Dict], file_path: str, file_type: str = 'csv'):
        """
        保存数据集到文件
        
        Args:
            qa_items: 问答对列表
            file_path: 文件路径
            file_type: 文件类型 (csv, json, jsonl)
        """
        try:
            # 创建临时数据集用于保存
            temp_dataset = []
            for item in qa_items:
                # 确保使用源码路径导入
                from deepeval.dataset import Golden
                golden = Golden(
                    input=item['question'],
                    expected_output=item['expected_output']
                )
                temp_dataset.append(golden)
            
            # 使用DeepEval的保存功能
            self.synthesizer.synthetic_goldens = temp_dataset
            saved_path = self.synthesizer.save_as(
                file_type=file_type,
                directory=os.path.dirname(file_path),
                file_name=os.path.basename(file_path).split('.')[0],
                quiet=True
            )
            
            logger.info(f"数据集已保存到: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"保存数据集失败: {str(e)}")
            raise

# 全局实例
deepeval_generator = DeepEvalDatasetGenerator() 