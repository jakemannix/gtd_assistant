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
        return [TextNode.parse_obj(entry) for entry in entries]

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
        current_time = datetime.now().timestamp()

        if not self.batch_by_user_message or message.role in [MessageRole.USER, MessageRole.SYSTEM]:
            self._commit_node()
            self.cur_batch_textnode = TextNode(
                text="",
                metadata={
                    "sub_dicts": [],
                    "timestamp": current_time,
                    "start_timestamp": current_time
                }
            )

        # Serialize the entire message using Pydantic's json method
        serialized_message = message.json()
        msg_dict = json.loads(serialized_message)
        
        # Add timestamp to the message
        msg_dict["timestamp"] = current_time
        
        # Update the TextNode
        self.cur_batch_textnode.text += f"{message.role}: {message.content}\n"
        self.cur_batch_textnode.metadata["sub_dicts"].append(json.dumps(msg_dict))
        self.cur_batch_textnode.metadata["timestamp"] = current_time

    def _commit_node(self) -> None:
        if self.cur_batch_textnode and self.cur_batch_textnode.text:
            # Insert the node into the vector index
            self.vector_index.insert_nodes([self.cur_batch_textnode])
            
            # Add the entry to the doc store
            self.doc_store.add_entry(self.cur_batch_textnode)
            
            # Reset the current batch text node
            self.cur_batch_textnode = None

    def get_all(self) -> List[ChatMessage]:
        current_time = datetime.now().timestamp()
        oldest_allowed_time = current_time - self.max_memory_age.total_seconds()
        
        nodes = self.doc_store.get_recent_entries(current_time, self.max_recent_memories)
        logger.info(f"Retrieved {len(nodes)} nodes from doc store")
        
        chat_messages = []
        for node in nodes:
            if node.metadata['timestamp'] >= oldest_allowed_time:
                for serialized_msg in node.metadata['sub_dicts']:
                    # Deserialize the message using Pydantic's parse_raw method
                    chat_message = ChatMessage.parse_raw(serialized_msg)
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
        self._commit_node()

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
