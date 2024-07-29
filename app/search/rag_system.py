# gtd_assistant/app/search/rag_system.py

from typing import Dict, Type
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.cohere import CohereEmbedding

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBED_MODELS: Dict[str, Type] = {
    "local": HuggingFaceEmbedding,
    "text-embedding-3-small": OpenAIEmbedding,
    "text-embedding-3-large": OpenAIEmbedding,
    "embed-english-v3.0": CohereEmbedding
}

LLM_MODELS: Dict[str, Type] = {
    "gpt-3.5-turbo": OpenAI,
    "gpt-4o": OpenAI,
    "claude-3.5-sonnet": Anthropic,
    "claude-3-opus": Anthropic
}


class RAGSystem:
    def __init__(self, vault_path: str, embed_model: str = "text-embedding-3-small", llm_model: str = "gpt-4o"):
        self.vault_path = vault_path
        self.embed_model = self.get_embed_model(embed_model)
        self.llm = RAGSystem.get_llm(llm_model)
        self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=self.llm)
        self.index = None

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
        if llm_model not in LLM_MODELS:
            raise ValueError(f"Unsupported LLM model: {llm_model}")

        model_class = LLM_MODELS[llm_model]
        return model_class(model=llm_model)

    def build_index(self):
        documents = SimpleDirectoryReader(self.vault_path).load_data()
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        self.index = VectorStoreIndex(
            nodes,
            service_context=self.service_context,
        )

    def query(self, query_text: str) -> str:
        if self.index is None:
            raise ValueError("Index has not been built. Call build_index() first.")
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        return str(response)
