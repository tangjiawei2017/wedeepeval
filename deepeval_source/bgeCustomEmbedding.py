import os
import json
from typing import Optional, List, Dict, Any
import torch
from sentence_transformers import SentenceTransformer

from deepeval.models import DeepEvalBaseEmbeddingModel


class CustomBgeEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",  # Use BGE-small-zh model
        device: Optional[str] = None,
        batch_size: int = 32,  # Local model can handle larger batches
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Initialize the model
        try:
            print(f"Loading BGE model: {model_name} on {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"Successfully loaded BGE model with dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading BGE model: {e}")
            print("Please install sentence-transformers: pip install sentence-transformers")
            raise

    def load_model(self):
        return self.model

    def embed_text(self, text: str) -> List[float]:
        try:
            print(f"Embedding text: {text[:100]}...")  # Only print first 100 chars
            
            # Use the sentence transformer to generate embedding
            embedding = self.model.encode(
                text, 
                convert_to_tensor=False,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # Convert numpy array to list of floats
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            print(f"Successfully generated embedding of length: {len(embedding_list)}")
            return embedding_list
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a default embedding vector with the model's dimension
            default_dim = self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else 512
            print(f"Returning default embedding vector with dimension: {default_dim}")
            return [0.0] * default_dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            print(f"Embedding {len(texts)} texts...")
            if texts:
                print(f"First text sample: {texts[0][:100]}...")
            
            # Use sentence transformer's batch processing
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=False,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True
            )
            
            # Convert numpy array to list of lists
            if hasattr(embeddings, 'tolist'):
                embeddings_list = embeddings.tolist()
            else:
                embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
            
            print(f"Successfully generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            print(f"Error in embed_texts: {e}")
            # Return default embeddings with the model's dimension
            default_dim = self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else 512
            print(f"Returning default embeddings with dimension: {default_dim}")
            return [[0.0] * default_dim for _ in texts]

    async def a_embed_text(self, text: str) -> List[float]:
        # For local models, async is the same as sync
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        # For local models, async is the same as sync
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return f"BGE-Embedding-{self.model_name}"


# Alternative implementation using transformers directly (if you prefer)
class CustomBGEEmbeddingModelDirect(DeepEvalBaseEmbeddingModel):
    """Alternative implementation using transformers directly"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            print(f"Loading BGE model directly: {model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.shape[-1]
            
            print(f"Successfully loaded BGE model with dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"Error loading BGE model: {e}")
            print("Please install transformers: pip install transformers")
            raise

    def load_model(self):
        return self.model

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for BGE models"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_text(self, text: str) -> List[float]:
        try:
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                
                # Normalize (BGE models typically use normalized embeddings)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings[0].cpu().numpy().tolist()
                
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self.embedding_dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings_list = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                with torch.no_grad():
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    batch_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    
                    # Normalize
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Convert to list
                    batch_list = batch_embeddings.cpu().numpy().tolist()
                    embeddings_list.extend(batch_list)
            
            return embeddings_list
            
        except Exception as e:
            print(f"Error in embed_texts: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        return f"BGE-Direct-{self.model_name}"


if __name__ == "__main__":
    # Test the BGE embedding model
    try:
        print("Testing BGE-small-zh embedding model...")
        
        # Use the sentence-transformers version (recommended)
        embedding_model = CustomBgeEmbeddingModel(
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu"  # or "cuda" if you have GPU
        )
        
        # Test single text embedding
        test_text = "这是一个测试文本"
        test_embedding = embedding_model.embed_text(test_text)
        print(f"Single text embedding dimension: {len(test_embedding)}")
        
        # Test batch embedding
        test_texts = ["第一个文本", "第二个文本", "第三个文本"]
        test_embeddings = embedding_model.embed_texts(test_texts)
        print(f"Batch embeddings: {len(test_embeddings)} texts, each with {len(test_embeddings[0])} dimensions")
        
        print("BGE embedding model test completed successfully!")
        
    except Exception as e:
        print(f"BGE embedding model test failed: {e}")
        print("Please install required packages:")
        print("pip install sentence-transformers torch")