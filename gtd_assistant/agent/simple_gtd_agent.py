from typing import List, Dict, Type, Optional
import logging
from llama_index.core.agent import AgentRunner
from llama_index.core.memory import VectorMemory, ChatMemoryBuffer, SimpleComposableMemory
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.multi_modal_llms.anthropic.utils import generate_anthropic_multi_modal_chat_message
from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.agent.react.prompts import CONTEXT_REACT_CHAT_SYSTEM_HEADER
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.schema import ImageDocument
from redis import Redis
from datetime import timedelta
from ..actions.gtd_tools import GTDTools
from ..search.rag_system import RAGSystem
from .memory import DurableSemanticMemory
from .constants import GTD_PROMPT
from ..config import Config


logger = logging.getLogger('gtd_assistant')

class SimpleGTDAgent:
    MODEL_CLIENTS: Dict[str, Type] = {
        "gpt-4o": OpenAIMultiModal,
        "claude-3.5-sonnet": AnthropicMultiModal,
    }
    def __init__(self, config: Config):
        self.config = config
        if config.model not in self.MODEL_CLIENTS:
            raise ValueError(f"Unsupported model: {config.model}")
        logger.info(f"Initializing SimpleGTDAgent with model: {config.model}, embed_model: {config.embed_model}")
        self.model = config.model        
        self.debug = config.debug
        self.embed_model_name = config.embed_model
        self.mm_llm = self.MODEL_CLIENTS[config.model](model=config.model, max_new_tokens=1000)
        self.memory = self.setup_memory(config.redis_url)
        self.tools = GTDTools(config=config, memory=self.memory).get_tools()
        self.agent = self.setup_agent()

    def setup_memory(self, redis_url: str):
        embed_model = RAGSystem.get_embed_model(self.embed_model_name)
        redis_client = Redis.from_url(redis_url)
        vector_memory = DurableSemanticMemory(
            redis_client=redis_client,
            embed_model=embed_model,
            max_memory_age=timedelta(hours=1),
            max_recent_memories=10,
            batch_by_user_message=True,
        )
        return vector_memory

    def setup_agent(self):
        if self.debug:
            chat_formatter = DebugReActChatFormatter(
                system_header=CONTEXT_REACT_CHAT_SYSTEM_HEADER, 
                context=GTD_PROMPT
            )
        else:
            chat_formatter = ReActChatFormatter.from_defaults(
                system_header=CONTEXT_REACT_CHAT_SYSTEM_HEADER, 
                context=GTD_PROMPT
            )
        
        worker = MultimodalReActAgentWorker.from_tools(
            tools=self.tools,
            multi_modal_llm=self.mm_llm,
            react_chat_formatter=chat_formatter,
            verbose=True,
            max_iterations=50,
            memory=self.memory
        )
        return AgentRunner(agent_worker=worker, memory=self.memory)

    def run(self, user_input: str, image_urls: Optional[List[str]] = None):
        try:
            # Prepare image documents if image URLs are provided
            image_docs = []
            if image_urls:
                for url in image_urls:
                    image_docs.append(ImageDocument(image_url=url))

            # Create a task with the user input and image docs in extra_state
            task = self.agent.create_task(
                input=user_input,
                extra_state={"image_docs": image_docs}
            )
            
            # Run steps until the agent is done
            step_output = None
            while True:
                step_output = self.agent.run_step(task_id=task.task_id)
                if step_output.is_last:
                    break
            
            # Finalize and return the response
            return self.agent.finalize_response(task_id=task.task_id, step_output=step_output)
            
        except Exception as e:
            logger.error(f"Error in SimpleGTDAgent.run: {str(e)}", exc_info=True)
            # Handle the error appropriately
            raise


class DebugReActChatFormatter(ReActChatFormatter):
    def format(self, *args, **kwargs):
        formatted_messages = super().format(*args, **kwargs)
        logger.info("Formatted messages:")
        for msg in formatted_messages:
            logger.info(f"{msg.role}: {msg.content}\n")
        return formatted_messages