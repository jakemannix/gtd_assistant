from llama_index.core.memory.vector_memory import VectorMemory, _stringify_obj
from llama_index.core.base.llms.types import ChatMessage
from typing import Dict, Optional
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.embeddings.utils import EmbedType


def stringify_complex_chat_message(msg: ChatMessage) -> Dict:
    """Improved utility function to convert chatmessage to serializable dict."""
    msg_dict = msg.dict()
    msg_dict["additional_kwargs"] = _stringify_obj(msg_dict["additional_kwargs"])
    
    # Handle the case where content is a list
    if isinstance(msg_dict["content"], list):
        msg_dict["content"] = " ".join(
            item.get('text', '') 
            for item in msg_dict["content"] 
            if isinstance(item, dict) and 'text' in item
        )
    elif not isinstance(msg_dict["content"], str):
        msg_dict["content"] = str(msg_dict["content"])
    
    return msg_dict