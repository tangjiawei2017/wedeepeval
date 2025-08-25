from typing import Optional, List, Dict, Union, Type
import os
import re

from deepeval.models.base_model import DeepEvalBaseEmbeddingModel

# check langchain availability
try:
    from langchain_core.documents import Document as LCDocument
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
    )
    from langchain_community.document_loaders.base import BaseLoader

    langchain_available = True
except ImportError:
    langchain_available = False

# check chromadb availability
try:
    import chromadb
    from chromadb import Metadata
    from chromadb.api.models.Collection import Collection

    chroma_db_available = True
except ImportError:
    chroma_db_available = False


# Define a helper function to check availability
def _check_chromadb_available():
    if not chroma_db_available:
        raise ImportError(
            "chromadb is required for this functionality. Install it via your package manager"
        )


def _check_langchain_available():
    if not langchain_available:
        raise ImportError(
            "langchain, langchain_community, and langchain_text_splitters are required for this functionality. Install it via your package manager"
        )


class CustomTextSplitter:
    """Custom text splitter that provides more control over chunking process"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks based on character count"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position for current chunk
            end = start + self.chunk_size
            
            # If this is not the first chunk, include overlap
            if start > 0 and self.chunk_overlap > 0:
                start = max(0, start - self.chunk_overlap)
            
            # Extract chunk
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move to next chunk
            start = end
            
            # If we've reached the end, break
            if end >= len(text):
                break
        
        return chunks
    
    def split_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """Split documents into chunks"""
        split_docs = []
        
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(text_chunks):
                # Create new document for each chunk
                new_doc = LCDocument(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                )
                split_docs.append(new_doc)
        
        return split_docs


class DocumentChunker:
    def __init__(
        self,
        embedder: DeepEvalBaseEmbeddingModel,
    ):
        _check_chromadb_available()
        _check_langchain_available()
        self.text_token_count: Optional[int] = None  # set later

        self.source_file: Optional[str] = None
        self.chunks: Optional["Collection"] = None
        self.sections: Optional[List[LCDocument]] = None
        self.embedder: DeepEvalBaseEmbeddingModel = embedder
        self.mean_embedding: Optional[float] = None

        # Mapping of file extensions to their respective loader classes
        self.loader_mapping: Dict[str, Type[BaseLoader]] = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
        }

    #########################################################
    ### Chunking Docs #######################################
    #########################################################

    async def a_chunk_doc(
        self, chunk_size: int = 1024, chunk_overlap: int = 0
    ) -> "Collection":
        _check_chromadb_available()

        # Raise error if chunk_doc is called before load_doc
        if self.sections is None or self.source_file is None:
            raise ValueError(
                "Document Chunker has yet to properly load documents"
            )

        # Create ChromaDB client
        full_document_path, _ = os.path.splitext(self.source_file)
        document_name = os.path.basename(full_document_path)
        client = chromadb.PersistentClient(path=f".vector_db/{document_name}")

        collection_name = f"processed_chunks_{chunk_size}_{chunk_overlap}"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            # Use custom text splitter instead of TokenTextSplitter
            text_splitter = CustomTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            # Collection doesn't exist, so create it and then add documents
            collection = client.create_collection(name=collection_name)

            langchain_chunks = text_splitter.split_documents(self.sections)
            contents = [rc.page_content for rc in langchain_chunks]
            embeddings = await self.embedder.a_embed_texts(contents)
            ids = [str(i) for i in range(len(contents))]

            max_batch_size = 5461  # Maximum batch size
            for i in range(0, len(contents), max_batch_size):
                batch_end = min(i + max_batch_size, len(contents))
                batch_contents = contents[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas: List["Metadata"] = [
                    {"source_file": self.source_file} for _ in batch_contents
                ]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
        return collection

    def chunk_doc(self, chunk_size: int = 1024, chunk_overlap: int = 0):
        _check_chromadb_available()

        # Raise error if chunk_doc is called before load_doc
        if self.sections is None or self.source_file is None:
            raise ValueError(
                "Document Chunker has yet to properly load documents"
            )

        # Create ChromaDB client
        full_document_path, _ = os.path.splitext(self.source_file)
        document_name = os.path.basename(full_document_path)
        client = chromadb.PersistentClient(path=f".vector_db/{document_name}")

        collection_name = f"processed_chunks_{chunk_size}_{chunk_overlap}"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            # Use custom text splitter instead of TokenTextSplitter
            text_splitter = CustomTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            # Collection doesn't exist, so create it and then add documents
            collection = client.create_collection(name=collection_name)

            langchain_chunks = text_splitter.split_documents(self.sections)
            contents = [rc.page_content for rc in langchain_chunks]
            embeddings = self.embedder.embed_texts(contents)
            ids = [str(i) for i in range(len(contents))]

            max_batch_size = 5461  # Maximum batch size
            for i in range(0, len(contents), max_batch_size):
                batch_end = min(i + max_batch_size, len(contents))
                batch_contents = contents[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = ids[i:batch_end]
                batch_metadatas: List["Metadata"] = [
                    {"source_file": self.source_file} for _ in batch_contents
                ]

                collection.add(
                    documents=batch_contents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
        return collection

    #########################################################
    ### Loading Docs ########################################
    #########################################################

    def get_loader(self, path: str, encoding: Optional[str]) -> "BaseLoader":
        # Find appropriate doc loader
        _, extension = os.path.splitext(path)
        extension = extension.lower()
        loader: Optional[type[BaseLoader]] = self.loader_mapping.get(extension)
        if loader is None:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load doc into sections and calculate total character count
        if loader is TextLoader:
            return loader(path, encoding=encoding, autodetect_encoding=True)
        elif loader is PyPDFLoader or loader is Docx2txtLoader:
            return loader(path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    async def a_load_doc(self, path: str, encoding: Optional[str]):
        loader = self.get_loader(path, encoding)
        self.sections = await loader.aload()
        self.text_token_count = self.count_tokens(self.sections)
        self.source_file = path

    def load_doc(self, path: str, encoding: Optional[str]):
        loader = self.get_loader(path, encoding)
        self.sections = loader.load()
        self.text_token_count = self.count_tokens(self.sections)
        self.source_file = path

    def count_tokens(self, chunks: List["LCDocument"]) -> int:
        """Custom token counting method using character-based approximation"""
        total_chars = 0
        for chunk in chunks:
            total_chars += len(chunk.page_content)
        
        # Approximate tokens as characters / 4 (rough estimate)
        # You can adjust this ratio based on your specific needs
        estimated_tokens = total_chars // 4
        return estimated_tokens
