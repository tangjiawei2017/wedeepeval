import os
import json
from typing import Optional, List, Dict, Any
from openai import OpenAI

from deepeval.models import DeepEvalBaseEmbeddingModel
from qwen4deepeval import CustomOpenAIQwen


class CustomQwenEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "text-embedding-v1",  # Use a valid model name
        api_key: str = "sk-bc733bafbe7242e8bf13a057ac05adb0",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        batch_size: int = 10,  # Set batch size below DashScope limit
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.batch_size = batch_size
        
        if not self.api_key:
            raise ValueError(
                "API key is required. Set DASHSCOPE_API_KEY environment variable "
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
            
            # Try alternative models if the current one fails
            alternative_models = [
                "text-embedding-3-small",
                "text-embedding-3-large", 
                "text-embedding-ada-002"
            ]
            
            if self.model_name not in alternative_models:
                print(f"Trying alternative model: text-embedding-3-small")
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    embedding = response.data[0].embedding
                    print(f"Successfully generated embedding with alternative model")
                    return embedding
                except Exception as e2:
                    print(f"Alternative model also failed: {e2}")
            
            # Return a default embedding vector
            print("Returning default embedding vector")
            return [0.0] * 1536

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
                    dim = 1536
                    all_embeddings.extend([[0.0] * dim for _ in batch])
            print(f"All embeddings: len {len(all_embeddings)}")
            return all_embeddings
        except Exception as e:
            print(f"Error in embed_texts: {e}")
            return [[0.0] * 1536 for _ in texts]

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        # 简单同步包装，保持语义：调用方用 await 也能拿到列表
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return f"{self.model_name}"


if __name__ == "__main__":
    # Example usage with OpenAI-compatible API
    try:
        # Option 1: Using OpenAI-compatible API (recommended for most use cases)
        # Use one of the available models on DashScope
        custom_llm = CustomOpenAIQwen(
            model_name="text-embedding-ada-002",  # or "qwen-turbo", "qwen-max", etc.
            max_tokens=1000,
            temperature=0.7,
            enable_thinking=False
        )
        print("Using OpenAI-compatible API...")
        
        # Test embedding model
        embedding_model = CustomQwenEmbeddingModel(
            model_name="text-embedding-v1"
        )
        print("Testing embedding model...")
        test_embedding = embedding_model.embed_text("Hello, this is a test.")
        print(f"Embedding dimension: {len(test_embedding)}")
        
    except Exception as e:
        print(f"OpenAI API setup failed: {e}")
        print("Falling back to local model...")
    
    # Test the model
    try:
        response = custom_llm.generate("Hello! Please introduce yourself briefly.")
        print(f"Model: {custom_llm.get_model_name()}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Generation failed: {e}")