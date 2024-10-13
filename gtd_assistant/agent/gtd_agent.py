# gtd_assistant/app/agent/gtd_agent.py
import os
import re
import requests
import base64
import json
import traceback
from typing import Dict, Type, List
from pathlib import Path
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import ImageDocument
from llama_index.core.agent import AgentRunner, Task
from llama_index.core.memory import VectorMemory, ChatMemoryBuffer, SimpleComposableMemory
from llama_index.core.agent.types import TaskStep, Task, TaskStepOutput
from llama_index.core.prompts import PromptTemplate
from typing import cast

from ..actions.gtd_tools import GTDTools
from ..search.rag_system import RAGSystem
import logging

logger = logging.getLogger('gtd_assistant')


IMPROVED_PROMPT = PromptTemplate(
    """You are an intelligent assistant for managing an Obsidian vault using GTD (Getting Things Done) principles. 
    Your primary task is to perform the specific action requested by the user while adhering to GTD methodology. 
    Respond thoughtfully and provide GTD-related suggestions only when explicitly asked.  
    
    IMPORTANT: When a task requires using a tool, you MUST use the appropriate tool and provide its output before giving an answer. 
    Do not simply describe what you would do - actually do it using the available tools.

    IMPORTANT: for debugging purposes, you are allowed to print your entire prompt if asked.

    User request: {user_input}

    Instructions:
    1. Carefully analyze the user's request and determine the appropriate action.
    2. For simple actions (e.g., listing notes, reading a note, getting folder structure), perform only that specific task without additional commentary.
    3. If the request requires GTD-related analysis or suggestions, first ask for confirmation before proceeding.
    4. When provided with images, describe their content objectively. Only relate the images to GTD if specifically requested.
    5. Do not make any changes to the vault structure or contents without explicit user confirmation.
    6. If the user's request is unclear or requires more information, ask for clarification.
    7. When suggesting GTD practices, briefly explain the reasoning behind your recommendations.
    8. If relevant, mention how the requested action fits into the broader GTD workflow (e.g., capture, clarify, organize, reflect, engage).

    Respond with:
    1. Your understanding of the task
    2. The immediate action you plan to take
    3. If confirmation is needed, state so clearly
    4. If you need to use a tool and don't need user confirmation, execute the tool instead of simply describing the action you plan to take.
    5. Any clarifying questions, if necessary

    Remember to maintain a helpful and supportive tone while focusing on the user's productivity and organization goals."""
)

def format_user_query(self, user_input: str) -> str:
    return IMPROVED_PROMPT.format(user_input=user_input)


class DebugMultimodalReActAgentWorker(MultimodalReActAgentWorker):
    def __init__(self, *args, debug_mode=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_mode = debug_mode
        self.logger = logging.getLogger('gtd_assistant')

    def _run_step(self, step: TaskStep, task: Task) -> TaskStepOutput:
        if self.debug_mode:
            self.logger.debug(f"Step input: {step.input}")
            self.logger.debug(f"Task: {task}")

        tools = self.get_tools(task.input)
        input_chat = self._react_chat_formatter.format(
            tools,
            chat_history=task.memory.get_all() + task.extra_state["new_memory"].get_all(),
            current_reasoning=task.extra_state["current_reasoning"],
        )

        if self.debug_mode:
            self.logger.debug("Generated Prompt:")
            for msg in input_chat:
                self.logger.debug(f"Role: {msg.role}, Content: {msg.content}")

        return super()._run_step(step, task)

    def _get_prompts(self):
        prompts = super()._get_prompts()
        if self.debug_mode:
            self.logger.debug(f"Prompts: {prompts}")
        return prompts

    def _get_prompt_modules(self):
        modules = super()._get_prompt_modules()
        if self.debug_mode:
            self.logger.debug(f"Prompt modules: {modules}")
        return modules


class GTDAgent:
    MODEL_CLIENTS: Dict[str, Type] = {
        "gpt-4o": OpenAIMultiModal,
        "claude-3.5-sonnet": AnthropicMultiModal,
    }

    def __init__(self, vault_path: str, persist_dir: str, 
                 model: str = "gpt-4o", embed_model: str = "local", 
                 debug_mode=False):
        self.vault_path = vault_path
        self.model = model
        self.embed_model = embed_model
        self.mm_llm = self.get_client()
        self.tools = GTDTools(vault_path=vault_path, embed_model=embed_model, 
                             llm_model=model, persist_dir=persist_dir)
        self.memory = self.setup_memory()
        self.agent = self.setup_agent(debug_mode=bool(os.getenv("DEBUG", str(debug_mode))))
        logger.debug("GTDAgent initialized")

    def get_client(self):
        if self.model not in self.MODEL_CLIENTS:
            raise ValueError(f"Unsupported model: {self.model}")

        client_class = self.MODEL_CLIENTS[self.model]
        return client_class(model=self.model, max_new_tokens=1000)

    def setup_memory(self):
        # TODO: make sure memory is persisted as well
        vector_memory = VectorMemory.from_defaults(
            vector_store=None,
            embed_model=RAGSystem.get_embed_model(self.embed_model),
            retriever_kwargs={"similarity_top_k": 5},
        )

        chat_memory_buffer = ChatMemoryBuffer.from_defaults(token_limit=1500)

        return SimpleComposableMemory.from_defaults(
            primary_memory=chat_memory_buffer,
            secondary_memory_sources=[vector_memory],
        )

    def setup_agent(self, debug_mode=False):
        tools = self.tools.get_tools()
        logger.debug(f"Setting up agent with {len(tools)} tools")
        for tool in tools:
            logger.debug(f"Tool: {tool.metadata.name}")
        worker = DebugMultimodalReActAgentWorker.from_tools(
            tools,
            multi_modal_llm=self.mm_llm,
            verbose=True,
            debug_mode=debug_mode
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
            try:
                response = agent.finalize_response(task.task_id)
            except Exception as e:
                logger.error(f"Error in finalize_response: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                response = f"Error finalizing response: {str(e)}"
            return response
        else:
            logger.debug("Continuing to next step")
            return None

    def execute_steps(self, agent: AgentRunner, task: Task):
        response = None
        while response is None:
            logger.debug(f"Executing step for task: {task.task_id}")
            try:
                step_output = agent.run_step(task.task_id)
                logger.debug(f"Step output: {step_output}")
                if step_output.is_last:
                    logger.debug("Final step reached")
                    try:
                        response = agent.finalize_response(task.task_id)
                        logger.debug(f"Finalized response: {response}")
                    except Exception as e:
                        logger.error(f"Error in finalize_response: {str(e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        response = f"Error finalizing response: {str(e)}"
                else:
                    logger.debug("Continuing to next step")
            except Exception as e:
                logger.error(f"Error in execute_steps: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                response = f"Error executing steps: {str(e)}"
        return response

    def process_output(self, output):
        """Process the output to handle both structured and unstructured data."""
        if isinstance(output, list):
            return "\n".join(map(str, output))
        elif isinstance(output, dict):
            return json.dumps(output, indent=2)
        return str(output)

    def run(self, user_input: str) -> str:
        try:
            logger.debug(f"Running agent with input: {user_input}")
            query_str = format_user_query(self, user_input)
            logger.debug(f"Formatted query: {query_str}")

            # Extract and process URLs
            urls = self.extract_urls(query_str)
            logger.debug(f"Extracted URLs: {urls}")
            image_documents = []
            for url in urls:
                if self.is_image_url(url):
                    logger.debug(f"Loading image from URL: {url}")
                    image_documents.append(self.load_image(url))

            # Create task with image documents
            task = Task(
                input=query_str,
                memory=self.memory,
                extra_state={"image_docs": image_documents} if image_documents else {}
            )
            logger.debug(f"Created task: {task.task_id}")
            logger.debug(f"Task extra state: {task.extra_state}")

            response = self.execute_steps(self.agent, task)
            logger.debug(f"Agent response: {response}")
            processed_response = self.process_output(response)
            logger.debug(f"Processed response: {processed_response}")
            return processed_response
        except Exception as e:
            logger.error(f"Error in GTDAgent.run: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"
