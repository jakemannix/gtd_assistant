# gtd_assistant/app/agent/gtd_agent.py

import re
import requests
import base64
from typing import Dict, Type, List
from pathlib import Path
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument
from llama_index.core.agent import AgentRunner, Task
from llama_index.core.memory import VectorMemory, ChatMemoryBuffer, SimpleComposableMemory
from ..actions.gtd_tools import GTDTools
from ..search.rag_system import RAGSystem
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GTDAgent:
    MODEL_CLIENTS: Dict[str, Type] = {
        "gpt-4o": OpenAIMultiModal,
        "claude-3.5-sonnet": AnthropicMultiModal,
    }

    def __init__(self, vault_path: str, model: str = "gpt-4o", embed_model: str = "local"):
        self.vault_path = vault_path
        self.model = model
        self.embed_model = embed_model
        self.mm_llm = self.get_client()
        self.tools = GTDTools(vault_path, embed_model, model)
        self.memory = self.setup_memory()
        self.agent = self.setup_agent()

    def get_client(self):
        if self.model not in self.MODEL_CLIENTS:
            raise ValueError(f"Unsupported model: {self.model}")

        client_class = self.MODEL_CLIENTS[self.model]
        return client_class(model=self.model, max_new_tokens=1000)

    def setup_memory(self):
        vector_memory = VectorMemory.from_defaults(
            vector_store=None,
            embed_model=RAGSystem.get_embed_model(self.embed_model),
            retriever_kwargs={"similarity_top_k": 5},
        )

        chat_memory_buffer = ChatMemoryBuffer.from_defaults()

        return SimpleComposableMemory.from_defaults(
            primary_memory=chat_memory_buffer,
            secondary_memory_sources=[vector_memory],
        )

    def setup_agent(self):
        tools = self.tools.get_tools()
        worker = MultimodalReActAgentWorker.from_tools(
            tools,
            multi_modal_llm=self.mm_llm,
            verbose=True,
        )

        agent = AgentRunner(
            agent_worker=worker,
            memory=self.memory,
            llm=self.mm_llm,
            callback_manager=worker.callback_manager,
            verbose=True
        )
        return agent

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        url_pattern = r'(https?://\S+|file://\S+)'
        return re.findall(url_pattern, text)

    @staticmethod
    def is_image_url(url: str) -> bool:
        try:
            if url.startswith('file://'):
                # TODO: check file extension here as well
                return Path(url[7:]).is_file()
            else:
                response = requests.head(url, allow_redirects=True, timeout=5)
                content_type = response.headers.get('Content-Type', '')
                return content_type.startswith('image/')
        except (requests.RequestException, OSError):
            return False

    @staticmethod
    def load_image(url: str) -> ImageDocument:
        if url.startswith('file://'):
            with open(url[7:], 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            return ImageDocument(image_data=image_data)
        else:
            return ImageDocument(image_url=url)

    def execute_step(self, agent: AgentRunner, task: Task):
        logger.debug(f"Executing step for task: {task.task_id}")
        step_output = agent.run_step(task.task_id)
        if step_output.is_last:
            logger.debug("Final step reached")
            response = agent.finalize_response(task.task_id)
            return response
        else:
            logger.debug("Continuing to next step")
            return None

    def execute_steps(self, agent: AgentRunner, task: Task):
        response = self.execute_step(agent, task)
        while response is None:
            response = self.execute_step(agent, task)
        return response

    def run(self, user_input: str) -> str:
        try:
            all_urls = self.extract_urls(user_input)
            image_urls = [url for url in all_urls if self.is_image_url(url)]

            image_documents = [self.load_image(url) for url in image_urls]

            query_str = f"Analyze the following input in the context of a GTD (Getting Things Done) " \
                        f"system and the user's Obsidian vault. Provide relevant actions, organization " \
                        f"suggestions, or insights based on the GTD methodology. If there are any images, " \
                        f"describe their content and relate it to GTD concepts: {user_input}"

            logger.debug(f"Creating task with query: {query_str}")
            task = self.agent.create_task(
                query_str,
                extra_state={"image_docs": image_documents},
            )
            logger.debug("Executing steps")
            response = self.execute_steps(self.agent, task)
            logger.debug(f"Response: {response}")
            return str(response)

        except Exception as e:
            return f"Error: {str(e)}"
