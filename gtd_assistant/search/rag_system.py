# gtd_assistant/app/search/rag_system.py

from typing import Dict, Type, List, Union
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
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from .obsidian_reader import ObsidianReader

logger = logging.getLogger('gtd_assistant')


LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBED_MODELS: Dict[str, Type] = {
    "local": HuggingFaceEmbedding,
    "text-embedding-3-small": OpenAIEmbedding,
    "text-embedding-3-large": OpenAIEmbedding,
    "embed-english-v3.0": CohereEmbedding
}


# TODO: align better with how llama_index handles data + RAG.
class RAGSystem:
    def __init__(self, vault_path: str, persist_dir: str,
                 embed_model: str = "text-embedding-3-small",
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

    # TODO: wire in configurability of the reranking model.
    def query(self, query_text: str, first_k: int = 50, top_k: int = 5) -> str:
        """Standard query interface returning summarized response."""
        if self.index is None:
            self.ensure_index_is_up_to_date()
        # TODO: Test this out.
        colbert_reranker = ColbertRerank(top_n=top_k)
        query_engine = self.index.as_query_engine(
            similarity_top_k=first_k, 
            node_postprocessors=[colbert_reranker]
        )
        response = query_engine.query(query_text)
        return str(response)

    def retrieve_raw(self, query_text: str, top_k: int = 5) -> List[NodeWithScore]:
        """Get raw retrieval results with similarity scores."""
        if self.index is None:
            self.ensure_index_is_up_to_date()
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query_text)
    
    def rerank(self, nodes: List[NodeWithScore], top_k: int = 5) -> List[NodeWithScore]:
        """Rerank nodes using Colbert reranker."""
        colbert_reranker = ColbertRerank(top_n=top_k)
        post_processed_nodes = colbert_reranker.postprocess_nodes(nodes)
        return post_processed_nodes

    def debug_query(self, query_text: str, first_k: int = 50, top_k: int = 5) -> Dict:
        """
        Get detailed debug information about vector search results.
        
        Args:
            query_text (str): The query text
            first_k (int): Number of results to retrieve before reranking
            top_k (int): Number of top results to return
            
        Returns:
            Dict containing:
                - query information
                - embedding details
                - vector store info
                - detailed results with scores and sources, before and after reranking
        """
        nodes_with_scores = self.retrieve_raw(query_text, first_k)
        reranked_nodes = self.rerank(nodes_with_scores, top_k)
        query_embedding = self.embed_model.get_text_embedding(query_text)
        
        # Create lookup of all nodes by ID for easy reference
        node_lookup = {}
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            node_lookup[node.node_id] = {
                "text": node.text,
                "source": node.metadata.get("source", "Unknown source"), 
                "metadata": node.metadata,
                "embedding": getattr(node, "embedding", None)
            }
        
        debug_info = {
            "query": {
                "text": query_text,
                "embedding": query_embedding,
            },
            "system_info": {
                "embedding_model": str(self.embed_model),
                "vector_store_type": str(type(self.index._vector_store)),
                "first_k": first_k,
                "top_k": top_k
            },
            "raw_results": [
                {
                    "node_id": n.node.node_id,
                    "score": n.score
                } for n in nodes_with_scores
            ],
            "reranked_results": [
                {
                    "node_id": n.node.node_id,
                    "score": n.score
                } for n in reranked_nodes
            ],
            "nodes": node_lookup
        }
        return debug_info

    def inspect_node(self, node_id: str) -> Dict:
        """
        Get detailed information about a specific node by ID.
        Useful for investigating specific results from debug_query.
        """
        if self.index is None:
            self.ensure_index_is_up_to_date()
            
        try:
            node = self.index.docstore.get_node(node_id)
            return {
                "node_id": node.node_id,
                "text": node.text,
                "metadata": node.metadata,
                "embedding": getattr(node, "embedding", None),
                "relationships": getattr(node, "relationships", {}),
            }
        except KeyError:
            return {"error": f"Node {node_id} not found"}

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
        """
        Get the appropriate LLM client based on model name pattern.
        Args:
            llm_model (str): Name of the model (e.g., "gpt-4-turbo", "claude-3-opus")
        Returns:
            BaseLLM: An instance of the appropriate LLM client
        Raises:
            ValueError: If the model name pattern isn't recognized
        """
        if llm_model.startswith("gpt-"):
            return OpenAI(model=llm_model)
        elif llm_model.startswith("claude-"):
            return Anthropic(model=llm_model)
        else:
            # TODO: add more models, esp open source ones from Hugging Face
            raise ValueError(f"Unsupported model pattern: {llm_model}. Must start with 'gpt-' or 'claude-'")

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
