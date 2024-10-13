# gtd_assistant/app/search/rag_system.py

from typing import Dict, Type
import os
import logging
import json
import hashlib
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage, StorageContext, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.cohere import CohereEmbedding

from .obsidian_reader import ObsidianReader

logger = logging.getLogger('gtd_assistant')


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
    def __init__(self, vault_path: str, persist_dir: str, embed_model: str = "text-embedding-3-small", 
                 llm_model: str = "gpt-4o"):
        self.vault_path = vault_path
        self.embed_model = self.get_embed_model(embed_model)
        self.llm = self.get_llm(llm_model)
        self.service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=self.llm)
        self.persist_dir = persist_dir
        self.index = None
        self.obsidian_reader = ObsidianReader(self.vault_path)
        self.note_hashes: Dict[str, str] = {}
        self.hash_file = os.path.join(self.persist_dir, "note_hashes.json")
        self.load_note_hashes()

    def ensure_index_is_up_to_date(self):
        if self.index is None:
            self._load_or_create_index()

        documents = self.obsidian_reader.load_data()
        changed_docs = []
        current_file_paths = set()

        for doc in documents:
            file_path = doc.metadata.get('source')
            if not file_path:
                logger.warning(f"Document missing source metadata: {doc}")
                continue
            
            current_file_paths.add(file_path)
            current_hash = self.get_document_hash(doc)
            if file_path not in self.note_hashes or self.note_hashes[file_path] != current_hash:
                changed_docs.append(doc)
                self.note_hashes[file_path] = current_hash

        removed_docs = set(self.note_hashes.keys()) - current_file_paths

        if changed_docs or removed_docs:
            self._update_index(changed_docs, removed_docs)

        self.save_note_hashes()

    def _load_or_create_index(self):
        if os.path.exists(os.path.join(self.persist_dir, "docstore.json")):
            logger.info("Loading existing index")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new index")
            documents = self.obsidian_reader.load_data()
            logger.info(f"Loading {len(documents)} documents")
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(documents)
            self.index = VectorStoreIndex(nodes, service_context=self.service_context)
            self._persist_index()
            for doc in documents:
                if doc.metadata.get('source'):
                    self.note_hashes[doc.metadata['source']] = self.get_document_hash(doc)
                else:
                    logger.warning(f"Document missing file_path metadata: {doc}")
            self.save_note_hashes()

    def _update_index(self, changed_docs, removed_docs):
        logger.info(f"Updating index: {len(changed_docs)} changed documents, {len(removed_docs)} removed documents")
        
        # Remove documents
        for doc_path in removed_docs:
            self.index.delete_ref_doc(doc_path)
            del self.note_hashes[doc_path]

        # Add or update documents
        parser = SimpleNodeParser.from_defaults()
        for doc in changed_docs:
            file_path = doc.metadata.get('source')
            if not file_path:
                logger.warning(f"Document missing source metadata: {doc}")
                continue
            
            # First, remove the old version if it existed
            self.index.delete_ref_doc(file_path)
            
            # Then, add the new version
            nodes = parser.get_nodes_from_documents([doc])
            self.index.insert_nodes(nodes)

        self._persist_index()

    def _persist_index(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def query(self, query_text: str) -> str:
        if self.index is None:
            self.ensure_index_is_up_to_date()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query_text)
        return str(response)

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

    def load_note_hashes(self):
        if os.path.exists(self.hash_file):
            with open(self.hash_file, 'r') as f:
                self.note_hashes = json.load(f)

    def save_note_hashes(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.hash_file, 'w') as f:
            json.dump(self.note_hashes, f)

    def get_document_hash(self, document: Document) -> str:
        return hashlib.md5(document.text.encode()).hexdigest()
