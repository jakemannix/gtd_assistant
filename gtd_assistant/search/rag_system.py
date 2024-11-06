# gtd_assistant/app/search/rag_system.py

from typing import Dict, Type, List
import logging
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    Document,
    StorageContext,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.schema import NodeWithScore

from .obsidian_reader import ObsidianReader
from ..config import Config

logger = logging.getLogger('gtd_assistant')

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBED_MODELS: Dict[str, Type] = {
    "local": HuggingFaceEmbedding,
    "text-embedding-3-small": OpenAIEmbedding,
    "text-embedding-3-large": OpenAIEmbedding,
    "embed-english-v3.0": CohereEmbedding
}

class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.embed_model = self.get_embed_model(config.embed_model)
        self.llm = self.get_llm(config.model)
        
        # Use SentenceWindowNodeParser for better context
        self.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=config.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            llm=self.llm,
            node_parser=self.node_parser
        )
        
        # Initialize Redis storage
        self.vector_store = RedisVectorStore(
            redis_url=config.redis_url,
            namespace=config.redis_namespace or "gtd_assistant"
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=config.persist_dir
        )
        
        self.index = None
        self.obsidian_reader = ObsidianReader(config.vault_path)
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load the vector index"""
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                service_context=self.service_context
            )
        except Exception as e:
            logger.info(f"Creating new index: {e}")
            documents = self.obsidian_reader.load_data()
            self.index = VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context,
                storage_context=self.storage_context,
                show_progress=True
            )

    def query(self, query_text: str, first_k: int = 50, top_k: int = 5) -> str:
        """Query with context-aware retrieval"""
        colbert_reranker = ColbertRerank(top_n=top_k)
        query_engine = self.index.as_query_engine(
            similarity_top_k=first_k,
            node_postprocessors=[colbert_reranker],
            response_synthesizer=get_response_synthesizer(
                service_context=self.service_context,
                response_mode="compact"
            )
        )
        response = query_engine.query(query_text)
        return str(response)

    def retrieve_and_rerank(self, query_text: str, top_k: int = 5, first_k: int = 50) -> List[NodeWithScore]:
        """Retrieve and rerank nodes"""
        colbert_reranker = ColbertRerank(top_n=top_k)
        retriever = self.index.as_retriever(similarity_top_k=first_k)
        nodes = retriever.retrieve(query_text)
        reranked_nodes = colbert_reranker.postprocess_nodes(nodes=nodes, query_str=query_text)
        return reranked_nodes


    def update_documents(self, documents: List[Document]):
        """Update index with new/modified documents"""
        self.index.refresh_ref_docs(
            documents,
            update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
        )

    @staticmethod
    def get_embed_model(embed_model: str):
        if embed_model not in EMBED_MODELS:
            raise ValueError(f"Unsupported embedding model: {embed_model}")

        model_class = EMBED_MODELS[embed_model]
        if embed_model == "local":
            embed_model = LOCAL_EMBED_MODEL
        return model_class(model_name=embed_model)

    @staticmethod
    def get_llm(llm_model: str):
        """Get the appropriate LLM client based on model name pattern."""
        if llm_model.startswith("gpt-"):
            return OpenAI(model=llm_model)
        elif llm_model.startswith("claude-"):
            return Anthropic(model=llm_model)
        else:
            raise ValueError(f"Unsupported model pattern: {llm_model}. Must start with 'gpt-' or 'claude-'")

