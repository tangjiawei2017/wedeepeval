import os
import json
from typing import Optional, List, Dict, Any
from openai import OpenAI
import asyncio
import pandas as pd
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig, ContextConstructionConfig, FiltrationConfig
from datetime import datetime
from deepeval.models import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
import requests

OPENAI_API_KEY = "sk-fzg2n3v8IpAnyk5l4IK5Mw"
OPENAI_BASE_URL = "http://litellm.uattest.weoa.com/v1"
LLM_MODEL = "qwen3-32b"
EMBEDDING_MODEL = "qwen3-embedding-8b"
EMBEDDING_API_KEY = "sk-fzg2n3v8IpAnyk5l4IK5Mw"
EMBEDDING_BASE_URL = "http://litellm.uattest.weoa.com/v1"



# 
class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        
        # Use provided API key or fall back to environment variable
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def load_model(self):
        # For OpenAI-compatible APIs, no model loading is needed
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt},{"role": "system", "content": "Please return a JSON object"}]
			
            
            extra_body = {}
            if not self.enable_thinking:
                extra_body["enable_thinking"] = False
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body=extra_body if extra_body else None,
                response_format={"type": "json_object"}
            )
			
			
            
            response = completion.choices[0].message.content
            print(f"Response: {response}")
            return response
            
        except Exception as e:
            error_msg = str(e)
            if "model_not_found" in error_msg or "does not exist" in error_msg:
                return f"Error: Model '{self.model_name}' not available. Please use one of the listed models."
            else:
                print(f"Error generating response: {e}")
                return f"Error: {str(e)}"

    async def a_generate(self, prompt: str) -> str:
        # For simplicity, using synchronous version
        # In production, you might want to use async OpenAI client
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"OpenAI-Qwen-{self.model_name}"

class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,  # Use a valid model name
        api_key: str = EMBEDDING_API_KEY,
        base_url: str = EMBEDDING_BASE_URL,
        batch_size: int = 10,  # Set batch size below DashScope limit
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def load_model(self):
        return self.client

    def embed_text(self, text: str) -> List[float]:
        try:
            print(f"Embedding text: {text[:100]}...")  # Only print first 100 chars
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            embedding = response.data[0].embedding
            print(f"Successfully generated embedding of length: {len(embedding)}")
            return embedding
        except Exception as e:
            error_msg = str(e)
            print(f"Error generating embedding: {error_msg}")
            raise e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            print(f"Embedding {len(texts)} texts...")
            print(f"Texts[0]: {texts[0]}")
            all_embeddings: List[List[float]] = []
            n = len(texts)
            # DashScope 兼容模式的批量上限为 25
            bs = min(max(1, getattr(self, "batch_size", 20)), 25)

            for i in range(0, n, bs):
                batch = texts[i:i + bs]
                try:
                    resp = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in resp.data]

                    # 严格对齐长度。若接口返回条数不足，补零向量到与 batch 等长
                    if len(batch_embeddings) != len(batch):
                        dim = len(batch_embeddings[0]) if batch_embeddings else 1536
                        batch_embeddings += [[0.0] * dim] * (len(batch) - len(batch_embeddings))

                    all_embeddings.extend(batch_embeddings)
                except Exception as batch_err:
                    # 该批失败，使用零向量占位，保证总长度不变
                    print(f"Embedding batch failed ({i//bs + 1}): {batch_err}")
                    import traceback
                    traceback.print_exc()
                    dim = 1536
                    all_embeddings.extend([[0.0] * dim for _ in batch])
            print(f"All embeddings: len {len(all_embeddings)}")
            return all_embeddings
        except Exception as e:
            print(f"Error in embed_texts: {e}")
            import traceback
            traceback.print_exc()
            return [[0.0] * 1536 for _ in texts]

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        # 简单同步包装，保持语义：调用方用 await 也能拿到列表
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return f"Qwen-Embedding-{self.model_name}"

if __name__ == "__main__":
    # Example usage with OpenAI-compatible API
	custom_llm = CustomOpenAI(model_name="qwen3-32b")
	custom_embedding = CustomEmbeddingModel()
	
	# Test the model
	try:
		response = custom_llm.generate("Hello! Please introduce yourself briefly.")
		print(f"Model: {custom_llm.get_model_name()}")
		print(f"Response: {response}")
	except Exception as e:
		print(f"Generation failed: {e}")
    
    
    
	print("=== 使用DeepEval Synthesizer从文档生成数据集 ===")
    
	try:
		styling_config = StylingConfig(
			input_format="询问文档中知识的中文问题"
		)

		# 配置过滤参数，完全禁用质量评估
		filtration_config = FiltrationConfig(
			synthetic_input_quality_threshold=-1.0,  # 设置为负数，首次评测即达阈值
			max_quality_retries=0,                   # 至少评 1 次，保证 score 被赋值
			critic_model=custom_llm                  # 用你当前的 LLM 做 critic（或换为稳妥模型）
		)
        
		# 配置上下文构建参数，使用最简单的设置
		context_config = ContextConstructionConfig(
			embedder=custom_embedding,
			critic_model=custom_llm, #用于construct的model
			max_contexts_per_document=10,
			min_contexts_per_document=1,
			max_context_length=10,
			min_context_length=1,
			chunk_size=800,           # 适中切分，确保至少有 1 个 chunk
			chunk_overlap=0,
			max_retries=1,          # 和chunk数有关，为0报错
			context_quality_threshold=0.0,
			context_similarity_threshold=0.0,
		)
        
		# 初始化Synthesizer
		synthesizer = Synthesizer(
			styling_config=styling_config, 
			model=custom_llm,
			async_mode=False,
			max_concurrent=1
		)
        
        # 文档路径列表
		document_paths = ["../docs/ECIF-guimian.docx"]
		print(f"开始从文档生成Golden: {document_paths}")
        
		# 使用DeepEval的generate_goldens_from_docs方法
		goldens = synthesizer.generate_goldens_from_docs(
			document_paths=document_paths,
			include_expected_output=True,
			max_goldens_per_context=3,  
			context_construction_config=context_config
		)
        
		print(f"成功生成 {len(goldens)} 个Golden")
		# 显示生成的Golden
		for i, golden in enumerate(goldens[:5]):
			print(f"\n=== Golden {i+1} ===")
			print(f"问题: {golden.input}")
			print(f"期望输出: {golden.expected_output}")
			if hasattr(golden, 'context') and golden.context:
				print(f"上下文长度: {len(str(golden.context))} 字符")
        
		# 保存到CSV文件
		test_data = []
		for golden in goldens:
			test_data.append({
				'input': golden.input,
				'expected_output': golden.expected_output,
				'context': str(golden.context) if hasattr(golden, 'context') and golden.context else ''
			})
		df = pd.DataFrame(test_data)
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"generated_goldens_{timestamp}.csv"
		df.to_csv(filename, index=False, encoding='utf-8-sig')
		print(f"Golden数据已保存到: {filename}")
        
	except Exception as e:
		print(f"生成Golden时出错: {e}")
		import traceback
		traceback.print_exc()

