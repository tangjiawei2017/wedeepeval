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
        """计算分批生成策略"""
        if num_questions <= GENERATION_CONFIG['single_batch_threshold']:
            # 小数量：一次性生成
            return {
                'strategy': 'single_batch',
                'batch_size': min(num_questions, GENERATION_CONFIG['max_single_batch_size']),
                'num_batches': 1
            }
        else:
            # 大数量：分批生成
            batch_size = GENERATION_CONFIG['batch_size']
            num_batches = (num_questions + batch_size - 1) // batch_size
            return {
                'strategy': 'multi_batch',
                'batch_size': batch_size,
                'num_batches': num_batches
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
            
            # 计算分批策略
            strategy = self._calculate_batch_strategy(num_questions)
            logger.info(f"使用生成策略: {strategy['strategy']}, 批次大小: {strategy['batch_size']}, 总批次数: {strategy['num_batches']}")
            
            # 准备上下文
            contexts_with_instruction = self._prepare_contexts_with_instruction(contexts)
            
            # 创建风格配置
            styling_config = self._create_styling_config(scenario)
            
            all_qa_items = []
            completed_count = 0
            
            # 分批生成
            for batch_num in range(strategy['num_batches']):
                batch_start = batch_num * strategy['batch_size']
                batch_end = min(batch_start + strategy['batch_size'], num_questions)
                current_batch_size = batch_end - batch_start
                
                logger.info(f"生成第 {batch_num + 1}/{strategy['num_batches']} 批，数量: {current_batch_size}")
                
                # 更新进度
                if progress_callback:
                    progress_callback(completed_count, num_questions, f"正在生成第 {batch_num + 1} 批...")
                
                # 计算当前批次每个上下文生成的问题数量
                max_goldens_per_context = max(1, current_batch_size // len(contexts_with_instruction))
                
                try:
                    # 使用DeepEval生成当前批次
                    dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_contexts(
                        contexts=contexts_with_instruction,
                        include_expected_output=True,
                        max_goldens_per_context=max_goldens_per_context
                    )
                    
                    # 转换为我们的格式
                    batch_items = []
                    for golden in dataset:
                        qa_item = {
                            'question': golden.input,
                            'expected_output': golden.expected_output,
                            'context': contexts,
                            'context_length': sum(len(x) for x in contexts)
                        }
                        batch_items.append(qa_item)
                    
                    # 去重处理
                    new_items = []
                    existing_questions = set(item['question'] for item in all_qa_items)
                    
                    for item in batch_items:
                        if item['question'] not in existing_questions:
                            new_items.append(item)
                            existing_questions.add(item['question'])
                        else:
                            logger.warning(f"发现重复问题，跳过: {item['question'][:50]}...")
                    
                    # 添加到总列表
                    all_qa_items.extend(new_items)
                    completed_count += len(new_items)
                    
                    logger.info(f"第 {batch_num + 1} 批完成，生成了 {len(new_items)} 个问答对")
                    logger.info(f"总进度: {completed_count}/{num_questions} ({completed_count/num_questions*100:.1f}%)")
                    
                    # 更新进度
                    if progress_callback:
                        progress_callback(completed_count, num_questions, f"第 {batch_num + 1} 批完成")
                    
                    # 如果已经达到目标数量，提前结束
                    if completed_count >= num_questions:
                        break
                        
                except Exception as e:
                    logger.error(f"第 {batch_num + 1} 批生成失败: {str(e)}")
                    if progress_callback:
                        progress_callback(completed_count, num_questions, f"第 {batch_num + 1} 批失败: {str(e)}")
                    # 继续下一批，不中断整个任务
                    continue
            
            # 如果生成的问答对数量不足，继续生成直到达到目标数量
            if len(all_qa_items) < num_questions:
                logger.warning(f"生成的问答对数量不足 ({len(all_qa_items)}/{num_questions})，继续生成")
                
                if progress_callback:
                    progress_callback(completed_count, num_questions, "继续生成补充数据...")
                
                # 继续生成直到达到目标数量
                while len(all_qa_items) < num_questions:
                    remaining = num_questions - len(all_qa_items)
                    current_batch_size = min(strategy['batch_size'], remaining)
                    
                    logger.info(f"继续生成，剩余 {remaining} 个，本批生成 {current_batch_size} 个")
                    
                    try:
                        # 使用DeepEval继续生成
                        dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_contexts(
                            contexts=contexts_with_instruction,
                            include_expected_output=True,
                            max_goldens_per_context=max(1, current_batch_size // len(contexts_with_instruction))
                        )
                        
                        # 转换为我们的格式
                        batch_items = []
                        for golden in dataset:
                            qa_item = {
                                'question': golden.input,
                                'expected_output': golden.expected_output,
                                'context': contexts,
                                'context_length': sum(len(x) for x in contexts)
                            }
                            batch_items.append(qa_item)
                        
                        # 去重处理
                        new_items = []
                        existing_questions = set(item['question'] for item in all_qa_items)
                        
                        for item in batch_items:
                            if item['question'] not in existing_questions:
                                new_items.append(item)
                                existing_questions.add(item['question'])
                            else:
                                logger.warning(f"发现重复问题，跳过: {item['question'][:50]}...")
                        
                        all_qa_items.extend(new_items)
                        completed_count += len(new_items)
                        
                        logger.info(f"继续生成完成，新增 {len(new_items)} 个问答对")
                        logger.info(f"总进度: {completed_count}/{num_questions} ({completed_count/num_questions*100:.1f}%)")
                        
                        # 更新进度
                        if progress_callback:
                            progress_callback(completed_count, num_questions, f"继续生成完成，新增 {len(new_items)} 个")
                        
                        if len(new_items) == 0:
                            logger.warning("本批没有生成新的问答对，可能达到生成上限")
                            break
                            
                    except Exception as e:
                        logger.error(f"继续生成失败: {str(e)}")
                        if progress_callback:
                            progress_callback(completed_count, num_questions, f"继续生成失败: {str(e)}")
                        break
            
            # 只取需要的数量
            final_items = all_qa_items[:num_questions]
            
            logger.info(f"DeepEval生成完成，总共生成了 {len(final_items)} 个问答对")
            
            # 最终进度更新
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