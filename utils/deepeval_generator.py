import os
import asyncio
import sys
from typing import List, Dict, Optional, Callable

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPEVAL_SOURCE_PATH = os.path.join(PROJECT_ROOT, 'deepeval_source')

# 添加本地 DeepEval 源码路径到 Python 路径的最前面
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

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.dataset import EvaluationDataset
from utils.logger import get_logger
from config import API_CONFIG, GENERATION_CONFIG

# 获取日志记录器
logger = get_logger('deepeval_generator')

class DeepEvalDatasetGenerator:
    """使用DeepEval生成数据集的工具类"""
    
    def __init__(self):
        """初始化DeepEval数据集生成器"""
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = API_CONFIG['openai_api_key']
        os.environ["OPENAI_BASE_URL"] = API_CONFIG['openai_base_url']
        
        # 初始化Synthesizer
        self.synthesizer = Synthesizer(
            model=API_CONFIG['openai_model'],
            async_mode=True,
            max_concurrent=10
        )
        
        logger.info(f"DeepEval数据集生成器初始化完成，使用模型: {API_CONFIG['openai_model']}")
    
    def _prepare_contexts_with_instruction(self, contexts: List[str]) -> List[List[str]]:
        """为上下文添加推理指示"""
        instruction = "请基于以下信息进行推理，生成中文问题和答案："
        contexts_with_instruction = [instruction] + contexts
        return [contexts_with_instruction]
    
    def _create_styling_config(self, scenario: str = "educational") -> StylingConfig:
        """创建生成风格配置"""
        return StylingConfig(
            scenario=scenario,  # 教育场景
            task="question_generation",  # 问题生成任务
            input_format="Chinese",  # 输入格式为中文
            expected_output_format="Chinese"  # 期望输出格式为中文
        )
    
    def _calculate_batch_strategy(self, num_questions: int) -> Dict:
        """保持兼容但不再使用分批，统一返回单批全量。"""
        return {
            'strategy': 'single_batch',
            'batch_size': num_questions,
            'num_batches': 1
        }
    
    async def generate_from_contexts(
        self, 
        contexts: List[str], 
        num_questions: int,
        scenario: str = "educational",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict]:
        """
        从上下文生成问答对，支持分批生成和进度跟踪
        
        Args:
            contexts: 上下文信息列表
            num_questions: 需要生成的问题数量
            scenario: 生成场景 (educational, conversational, technical)
            progress_callback: 进度回调函数 (completed, total, status)
            
        Returns:
            生成的问答对列表
        """
        try:
            logger.info(f"开始使用DeepEval生成数据集: 上下文数量={len(contexts)}, 问题数量={num_questions}")

            # 一次性全量生成（无分批）
            contexts_with_instruction = self._prepare_contexts_with_instruction(contexts)
            styling_config = self._create_styling_config(scenario)

            if progress_callback:
                progress_callback(0, num_questions, "开始全量生成...")

            # 挂载外部进度钩子，将 rich 进度更新透传到回调
            prev_hook = None
            try:
                import deepeval.utils as dutils

                prev_hook = getattr(dutils, 'EXTERNAL_PROGRESS_HOOK', None)

                def _hook(event: dict):
                    try:
                        desc = (event.get('description') or '').lower()
                        # 只拦截主进度条，避免加载/切分等子任务干扰
                        if not (
                            'generating goldens from context' in desc
                            or 'generating input' in desc
                            or 'generate goldens' in desc
                        ):
                            return
                        raw_completed = int(event.get('completed') or 0)
                        raw_total = int((event.get('total') or 0) or 1)
                        ratio = max(0.0, min(1.0, (raw_completed / raw_total)))
                        mapped_completed = int(min(num_questions, max(0, int(ratio * num_questions))))
                        status = event.get('description') or "生成中"
                        logger.info(f"[progress_hook] {status} raw={raw_completed}/{raw_total} -> mapped={mapped_completed}/{num_questions}")
                        if progress_callback:
                            progress_callback(mapped_completed, num_questions, status)
                    except Exception:
                        pass

                dutils.EXTERNAL_PROGRESS_HOOK = _hook

                max_goldens_per_context = max(1, num_questions)
                dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_contexts(
                    contexts=contexts_with_instruction,
                    include_expected_output=True,
                    max_goldens_per_context=max_goldens_per_context
                )
            finally:
                try:
                    import deepeval.utils as dutils
                    dutils.EXTERNAL_PROGRESS_HOOK = prev_hook
                except Exception:
                    pass

            qa_items = []
            seen = set()
            for golden in dataset:
                q = golden.input
                if q in seen:
                    continue
                seen.add(q)
                qa_items.append({
                    'question': q,
                    'expected_output': golden.expected_output,
                    'context': contexts,
                    'context_length': sum(len(x) for x in contexts)
                })

            final_items = qa_items[:num_questions]
            logger.info(f"DeepEval生成完成，总共生成了 {len(final_items)} 个问答对")

            if progress_callback:
                progress_callback(len(final_items), num_questions, "生成完成")

            return final_items
            
        except Exception as e:
            logger.error(f"DeepEval生成失败: {str(e)}")
            if progress_callback:
                progress_callback(0, num_questions, f"生成失败: {str(e)}")
            raise
    
    async def generate_from_documents(
        self, 
        document_paths: List[str], 
        num_questions: int,
        scenario: str = "educational"
    ) -> List[Dict]:
        """
        从文档生成问答对
        
        Args:
            document_paths: 文档路径列表
            num_questions: 需要生成的问题数量
            scenario: 生成场景
            
        Returns:
            生成的问答对列表
        """
        try:
            logger.info(f"开始使用DeepEval从文档生成数据集: 文档数量={len(document_paths)}, 问题数量={num_questions}")
            
            # 计算每个文档生成的问题数量
            max_goldens_per_context = max(1, num_questions // len(document_paths))
            
            # 使用DeepEval生成数据集
            dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_docs(
                document_paths=document_paths,
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context
            )
            
            # 转换为我们的格式
            qa_items = []
            for golden in dataset:
                qa_item = {
                    'question': golden.input,
                    'expected_output': golden.expected_output,
                    'context': [f"文档: {golden.source_file}"] if hasattr(golden, 'source_file') else [],
                    'context_length': len(golden.input)
                }
                qa_items.append(qa_item)
            
            logger.info(f"DeepEval文档生成完成，共生成 {len(qa_items)} 个问答对")
            return qa_items
            
        except Exception as e:
            logger.error(f"DeepEval文档生成失败: {str(e)}")
            raise
    
    async def generate_from_scratch(
        self, 
        num_questions: int,
        scenario: str = "educational"
    ) -> List[Dict]:
        """
        从零开始生成问答对
        
        Args:
            num_questions: 需要生成的问题数量
            scenario: 生成场景
            
        Returns:
            生成的问答对列表
        """
        try:
            logger.info(f"开始使用DeepEval从零生成数据集: 问题数量={num_questions}")
            
            # 使用DeepEval生成数据集
            dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_scratch(
                num_goldens=num_questions
            )
            
            # 转换为我们的格式
            qa_items = []
            for golden in dataset:
                qa_item = {
                    'question': golden.input,
                    'expected_output': golden.expected_output,
                    'context': [],
                    'context_length': 0
                }
                qa_items.append(qa_item)
            
            logger.info(f"DeepEval从零生成完成，共生成 {len(qa_items)} 个问答对")
            return qa_items
            
        except Exception as e:
            logger.error(f"DeepEval从零生成失败: {str(e)}")
            raise
    
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