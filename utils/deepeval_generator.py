import os
import asyncio
import sys
from typing import List, Dict

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
from deepeval.synthesizer.config import StylingConfig
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
        
        # 初始化Synthesizer
        self.synthesizer = Synthesizer(
            model=API_CONFIG['openai_model'],
            async_mode=True,
            max_concurrent=10
        )
        
        logger.info(f"DeepEval数据集生成器初始化完成，使用模型: {API_CONFIG['openai_model']}")
    
    def _prepare_contexts_with_instruction(self, contexts: List[str]) -> List[List[str]]:
        """为上下文添加推理指示"""
        instruction = "请基于以下信息进行推理，生成中文问题和答案。要求：1. 问题必须用中文提问；2. 答案必须用中文回答；3. 内容要符合中文表达习惯；4. 不要生成任何英文内容。"
        contexts_with_instruction = [instruction] + contexts
        return [contexts_with_instruction]
    
    def _create_styling_config(self, scenario: str = "educational") -> StylingConfig:
        """创建生成风格配置"""
        return StylingConfig(
            scenario=scenario,  # 教育场景
            task="生成中文问答对。严格要求：1. 问题必须用中文提问；2. 答案必须用中文回答；3. 内容要符合中文表达习惯；4. 不要生成任何英文内容；5. 所有文本必须是中文。",  # 问题生成任务
            input_format="中文问题，必须以中文开头，不能包含英文",  # 输入格式为中文
            expected_output_format="中文答案，必须用中文回答，不能包含英文"  # 期望输出格式为中文
        )
    

    
    async def generate_from_contexts(
        self, 
        contexts: List[str], 
        num_questions: int,
        scenario: str = "educational"
    ) -> List[Dict]:
        """
        从上下文生成问答对，一次性完成所有生成
        
        Args:
            contexts: 上下文信息列表
            num_questions: 需要生成的问题数量
            scenario: 生成场景 (educational, conversational, technical)
            
        Returns:
            生成的问答对列表
        """
        try:
            logger.info(f"开始使用DeepEval生成数据集: 上下文数量={len(contexts)}, 问题数量={num_questions}")
            
            # 准备上下文
            contexts_with_instruction = self._prepare_contexts_with_instruction(contexts)
            
            # 创建风格配置
            styling_config = self._create_styling_config(scenario)
            # 进一步强化中文生成要求
            styling_config.task = "基于上下文内容生成中文问答对。严格要求：1. 问题必须用中文提问；2. 答案必须用中文回答；3. 内容要符合中文表达习惯；4. 不要生成任何英文内容；5. 所有文本必须是中文；6. 禁止使用英文单词或短语。"
            
            # 计算每个上下文生成的问题数量
            max_goldens_per_context = max(1, num_questions // len(contexts_with_instruction))
            
            logger.info(f"一次性生成 {num_questions} 个问答对，每个上下文生成 {max_goldens_per_context} 个")
            
            # 设置风格配置
            self.synthesizer.styling_config = styling_config
            
            # 使用DeepEval一次性生成所有问答对
            dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_contexts(
                contexts=contexts_with_instruction,
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context
            )
            
            # 转换为我们的格式
            qa_items = []
            for golden in dataset:
                qa_item = {
                    'question': golden.input,
                    'expected_output': golden.expected_output,
                    'context': contexts,
                    'context_length': sum(len(x) for x in contexts)
                }
                qa_items.append(qa_item)
            
            # 去重处理
            final_items = []
            seen_questions = set()
            
            for item in qa_items:
                if item['question'] not in seen_questions:
                    final_items.append(item)
                    seen_questions.add(item['question'])
                else:
                    logger.warning(f"发现重复问题，跳过: {item['question'][:50]}...")
            
            # 只取需要的数量
            final_items = final_items[:num_questions]
            
            logger.info(f"DeepEval生成完成，总共生成了 {len(final_items)} 个问答对")
            
            return final_items
            
        except Exception as e:
            logger.error(f"DeepEval生成失败: {str(e)}")
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
            
            # 创建风格配置并设置到 synthesizer
            styling_config = self._create_styling_config(scenario)
            # 进一步强化中文生成要求
            styling_config.task = "基于文档内容生成中文问答对。严格要求：1. 问题必须用中文提问；2. 答案必须用中文回答；3. 内容要符合中文表达习惯；4. 不要生成任何英文内容；5. 所有文本必须是中文；6. 禁止使用英文单词或短语。"
            self.synthesizer.styling_config = styling_config
            
            # 使用DeepEval生成数据集
            dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_docs(
                document_paths=document_paths,
                include_expected_output=True,
                max_goldens_per_context=max_goldens_per_context
            )
            
            # 转换为我们的格式
            qa_items = []
            for golden in dataset:
                # 获取真实的文档片段内容
                context_content = []
                if hasattr(golden, 'context') and golden.context:
                    # 如果 Golden 对象有 context 属性，直接使用
                    if isinstance(golden.context, list):
                        context_content = golden.context
                    else:
                        context_content = [str(golden.context)]
                elif hasattr(golden, 'source_file'):
                    # 如果没有 context，至少记录来源文件
                    context_content = [f"文档: {golden.source_file}"]
                
                qa_item = {
                    'question': golden.input,
                    'expected_output': golden.expected_output,
                    'context': context_content,
                    'context_length': sum(len(str(x)) for x in context_content)
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
        scenario: str = "educational",
        topic: str = None
    ) -> List[Dict]:
        """
        从零开始生成问答对
        
        Args:
            num_questions: 需要生成的问题数量
            scenario: 生成场景
            topic: 主题信息（可选）
            
        Returns:
            生成的问答对列表
        """
        try:
            logger.info(f"开始使用DeepEval从零生成数据集: 问题数量={num_questions}, 主题={topic}")
            
            # 创建风格配置并设置到 synthesizer
            styling_config = self._create_styling_config(scenario)
            
            # 如果有主题信息，修改任务描述
            if topic:
                styling_config.task = f"基于主题'{topic}'生成中文问题和答案。严格要求：1. 问题必须用中文提问；2. 答案必须用中文回答；3. 内容要符合中文表达习惯；4. 不要生成任何英文内容；5. 所有文本必须是中文；6. 禁止使用英文单词或短语；7. 问题必须以中文开头，不能以英文开头；8. 绝对不允许生成英文问题；9. 问题必须以'什么是'、'如何'、'为什么'、'请解释'等中文词汇开头；10. 禁止使用任何英文单词，包括技术术语也要用中文表达。"
                styling_config.scenario = f"关于{topic}的中文教育问答"
                styling_config.input_format = "中文问题，必须以中文开头，不能包含英文"
                styling_config.expected_output_format = "中文答案，必须用中文回答，不能包含英文"
            
            self.synthesizer.styling_config = styling_config
            
            # 使用DeepEval生成数据集
            dataset: EvaluationDataset = await self.synthesizer.a_generate_goldens_from_scratch(
                num_goldens=num_questions
            )
            
            # 转换为我们的格式
            qa_items = []
            for golden in dataset:
                # 处理 expected_output 为 None 的情况
                expected_output = golden.expected_output if golden.expected_output is not None else "暂无标准答案"
                
                qa_item = {
                    'question': golden.input,
                    'expected_output': expected_output,
                    'context': [f"主题: {topic}"] if topic else [],
                    'context_length': len(topic) if topic else 0
                }
                qa_items.append(qa_item)
            
            logger.info(f"DeepEval从零生成完成，共生成 {len(qa_items)} 个中文问答对")
            

            
            return qa_items
            
        except Exception as e:
            logger.error(f"DeepEval从零生成失败: {str(e)}")
            raise
    
    async def generate_from_goldens(
        self, 
        goldens: List, 
        num_questions: int,
        scenario: str = "educational"
    ) -> List[Dict]:
        """
        从现有的Golden数据集生成新的问答对，用于数据集扩写
        
        Args:
            goldens: 现有的Golden对象列表
            num_questions: 需要生成的问题数量
            scenario: 生成场景 (educational, conversational, technical)
            
        Returns:
            List[Dict]: 生成的问答对列表
        """
        try:
            logger.info(f"开始从Golden数据集生成扩写数据: 原始数据={len(goldens)}, 目标数量={num_questions}")
            

            
            # 创建风格配置
            styling_config = self._create_styling_config(scenario)
            styling_config.task = "基于现有数据生成中文问答对。严格要求：1. 问题必须用中文提问；2. 答案必须用中文回答；3. 内容要符合中文表达习惯；4. 不要生成任何英文内容；5. 所有文本必须是中文；6. 禁止使用英文单词或短语。"
            self.synthesizer.styling_config = styling_config
            
            # 计算每个Golden生成多少个新Golden
            max_goldens_per_golden = max(1, num_questions // len(goldens))
            
            # 使用DeepEval的generate_goldens_from_goldens方法
            new_goldens = await self.synthesizer.a_generate_goldens_from_goldens(
                goldens=goldens,
                max_goldens_per_golden=max_goldens_per_golden,
                include_expected_output=True
            )
            
            # 转换为我们的格式
            qa_items = []
            for golden in new_goldens[:num_questions]:  # 限制数量
                qa_item = {
                    'question': golden.input,
                    'expected_output': golden.expected_output if golden.expected_output else "暂无标准答案",
                    'context': golden.context if golden.context else [],
                    'context_length': len(str(golden.context)) if golden.context else 0
                }
                qa_items.append(qa_item)
            
            logger.info(f"DeepEval从Golden扩写完成，共生成 {len(qa_items)} 个问答对")
            
            return qa_items
            
        except Exception as e:
            logger.error(f"DeepEval从Golden扩写失败: {str(e)}")
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