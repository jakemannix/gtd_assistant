from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import base64
from enum import Enum
from pydantic import Field
from redis import Redis

from llama_index.core.memory import BaseMemory
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import TextNode
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.base.llms.types import MessageRole

import logging
import time

logger = logging.getLogger('gtd_assistant')

MAX_TIMESTAMP = 10_000_000_000

def pack_entry(entry: Dict) -> str:
    json_str = json.dumps(entry)
    return base64.b64encode(json_str.encode()).decode()

def unpack_entry(packed_str: str) -> Dict:
    json_str = base64.b64decode(packed_str).decode()
    return json.loads(json_str)

class SortedRedisStore:
    def __init__(self, redis_client: Redis, key: str):
        self.redis_client = redis_client
        self.key = key

    def add_entry(self, node: Dict, score: float):
        packed_entry = pack_entry(node)
        result = self.redis_client.zadd(
            name=self.key,
            mapping={packed_entry: score}
        )
        logger.debug(f"Added entry to Redis with key {self.key}, score {score}. Result: {result}")

    def get_sorted_entries(self, max_timestamp: float, limit: int = 10) -> List[Dict]:
        packed_entries = self.redis_client.zrevrangebyscore(
            name=self.key,
            max=max_timestamp,
            min='-inf',
            start=0,
            num=limit
        )
        logger.debug(f"Retrieved {len(packed_entries)} packed entries from Redis")
        return [unpack_entry(entry) for entry in reversed(packed_entries)]

    def delete_old_entries(self, oldest_allowed_timestamp: float):
        result = self.redis_client.zremrangebyscore(
            name=self.key,
            min='-inf',
            max=oldest_allowed_timestamp
        )
        logger.debug(f"Deleted {result} old entries")

class SortedDocStore:
    def __init__(self, redis_client: Redis):
        self.redis_sorted_store = SortedRedisStore(redis_client, 'chat_entries')

    def add_entry(self, node: TextNode):
        logger.debug(f"Storing node in SortedDocStore: {node.dict()}")
        self.redis_sorted_store.add_entry(node.dict(), node.metadata['timestamp'])

    def get_recent_entries(self, max_datestamp: float, limit: int = 10) -> List[TextNode]:
        entries = self.redis_sorted_store.get_sorted_entries(max_datestamp, limit)
        logger.debug(f"Retrieved entries from SortedDocStore: {entries}")
        nodes = []
        for entry in entries:
            # No need to manually deserialize sub_dicts
            nodes.append(TextNode.parse_obj(entry))
        return nodes

    def delete_old_entries(self, oldest_allowed_timestamp: float):
        self.redis_sorted_store.delete_old_entries(oldest_allowed_timestamp)


class DurableSemanticMemory(BaseMemory):
    vector_store: RedisVectorStore = Field(...)
    vector_index: VectorStoreIndex = Field(...)
    doc_store: SortedDocStore = Field(...)
    retriever_kwargs: Dict[str, Any] = Field(default_factory=dict)
    max_recent_memories: int = Field(default=100)
    max_memory_age: timedelta = Field(default=timedelta(days=7))
    batch_by_user_message: bool = Field(default=True)
    cur_batch_textnode: Optional[TextNode] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        redis_client: Redis,
        embed_model: Any,
        max_recent_memories: int = 100,
        max_memory_age: timedelta = timedelta(days=7),
        **kwargs: Any
    ):
        super().__init__()
        self.vector_store = RedisVectorStore(redis_client=redis_client)
        self.vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store, 
            embed_model=embed_model
        )
        self.doc_store = SortedDocStore(redis_client)
        self.max_recent_memories = max_recent_memories
        self.max_memory_age = max_memory_age
        self.batch_by_user_message = True
        self.cur_batch_textnode = None

    def put(self, message: ChatMessage) -> None:
        # Use the timestamp from the message metadata if available
        current_time = message.additional_kwargs.get('timestamp', datetime.now().timestamp())
        
        if not self.batch_by_user_message or message.role in [MessageRole.USER, MessageRole.SYSTEM]:
            self.flush()
            self.cur_batch_textnode = TextNode(
                text="",
                metadata={
                    "sub_dicts": [],
                    "timestamp": current_time,
                    "start_timestamp": current_time
                }
            )

        serialized_message = json.loads(message.json())

        # Only set the timestamp if it's not already present
        serialized_message["timestamp"] = serialized_message.get("timestamp", current_time)

        # Handle potential list content (unlikely but kept for consistency)
        if isinstance(serialized_message["content"], list):
            serialized_message["content"] = " ".join(
                item.get('text', '') 
                for item in serialized_message["content"] 
                if isinstance(item, dict) and 'text' in item
            )

        self.cur_batch_textnode.text += f"{message.role}: {serialized_message['content']}\n"
        self.cur_batch_textnode.metadata["sub_dicts"].append(serialized_message)
        self.cur_batch_textnode.metadata["timestamp"] = current_time

    def get_all(self) -> List[ChatMessage]:
        self.flush()
        current_time = datetime.now().timestamp()
        oldest_allowed_time = current_time - self.max_memory_age.total_seconds()
        logger.debug(f"Current time: {current_time}, Oldest allowed time: {oldest_allowed_time}")
        
        nodes = self.doc_store.get_recent_entries(current_time, self.max_recent_memories)
        logger.info(f"Retrieved {len(nodes)} nodes from doc store")
        
        chat_messages = []
        for node in nodes:
            logger.debug(f"Node timestamp: {node.metadata['timestamp']}")
            if node.metadata['timestamp'] >= oldest_allowed_time:
                sub_dicts = json.loads(node.metadata['sub_dicts'])
                logger.debug(f"Parsed sub_dicts: {sub_dicts}")
                
                for sub_dict in sub_dicts:
                    # Parse the ChatMessage
                    chat_message = ChatMessage.parse_obj(sub_dict)
                    extra_data = {
                        k: v
                        for k, v in sub_dict.items()
                        if k not in {'role', 'content', 'additional_kwargs'} and not isinstance(v, ChatMessage)
                    }
                    chat_message.additional_kwargs.update(extra_data)
                    chat_messages.append(chat_message)
        
        logger.debug(f"Retrieved chat messages: {[msg.dict() for msg in chat_messages]}")
        return chat_messages

    def get(self, input: Optional[str] = None, **kwargs: Any) -> List[ChatMessage]:
        # For now, this just returns all messages
        # In the future, you might want to implement filtering based on the input
        return self.get_all()

    def set(self, messages: List[ChatMessage]) -> None:
        for message in messages:
            self.put(message)

    def reset(self) -> None:
        self.vector_store.clear()

    def flush(self) -> None:        
        if self.cur_batch_textnode and self.cur_batch_textnode.text:
            # For vector index, ensure sub_dicts is a string
            vector_node = self.cur_batch_textnode.copy()
            vector_node.metadata["sub_dicts"] = json.dumps(vector_node.metadata["sub_dicts"])
            self.vector_index.insert_nodes([vector_node])
            
            # Add the original entry to the doc store
            self.doc_store.add_entry(self.cur_batch_textnode)
            
            self.cur_batch_textnode = None


    @classmethod
    def from_defaults(
        cls,
        collection_name: str,
        host: str,
        port: int,
        embed_model: Any,
        max_recent_memories: int = 100,
        max_memory_age: timedelta = timedelta(days=7),
        **kwargs: Any
    ) -> "DurableSemanticMemory":
        return cls(
            collection_name=collection_name,
            host=host,
            port=port,
            embed_model=embed_model,
            max_recent_memories=max_recent_memories,
            max_memory_age=max_memory_age,
            **kwargs
        )
