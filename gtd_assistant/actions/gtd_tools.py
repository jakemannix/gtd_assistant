# gtd_assistant/app/actions/gtd_tools.py

from typing import List, Dict
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage
from ..file_system.obsidian_interface import ObsidianVault
from ..search.rag_system import RAGSystem
from ..analysis.spider import Spider
from ..agent.memory import DurableSemanticMemory
import logging
import os
import traceback
from pathlib import Path

logger = logging.getLogger('gtd_assistant')

# TODO: all actions reading / writing to the filesystem need to assume the caller is an LLM, who makes mistakes
# thus we need to expect exceptions, and return an error message, possibly also listing the contents of the directory
# above the path that failed, to help the LLM understand the file system layout and self correct.
class GTDTools:
    def __init__(self, vault_path: str, embed_model: str, llm_model: str, persist_dir: str,
     memory: DurableSemanticMemory):
        self.vault = ObsidianVault(vault_path)
        self.rag = RAGSystem(vault_path=vault_path, persist_dir=persist_dir,
                             embed_model=embed_model, llm_model=llm_model)
        self.spider = Spider()
        self.memory = memory
        # self.rag.ensure_index_is_up_to_date()

    def search_memory(self, query: str) -> List[ChatMessage]:
        """
        Search long-term chat memory for information relevant to the query.
        Args:
            query (str): The query to search for.
        Returns:
            List[ChatMessage]: A list of ChatMessage's that are relevant to the query.
        """
        return self.memory.get(query)

    def get_folder_structure(self) -> Dict[str, List[str]]:
        """
        Check the current contents of the Obsidian vault.
        Returns:
            Dict[str, List[str]]: A dictionary with keys being the folder path and
            values being a list of all markdown files in that folder.
        """
        logger.debug("Attempting to get current Obsidian vault folder structure")
        return self.vault.get_folder_structure()

    def query(self, question: str) -> str:
        """
        Query the RAG system with a question.
        Args:
            question (str): The question to ask the RAG system.
        Returns:
            str: The answer from the RAG system.
        """
        return self.rag.query(question)

    def list_notes(self, folder_path: str) -> List[str]:
        """
        List notes and folders in a specific folder.
        Args:
            folder_path (str): The path to the folder.
        Returns:
            List[str]: A list of note names and subfolders in the folder.
        """
        logger.debug(f"Attempting to list notes and folders in: {folder_path}")
        try:
            contents = self.vault.list_notes_in_folder(folder_path)
            logger.debug(f"Contents found: {contents}")
            return contents
        except Exception as e:
            logger.error(f"Error listing notes and folders: {str(e)}")
            return []

    def read_note(self, note_path: str) -> str:
        """
        Read the content of a specific note.
        Args:
            note_path (str): The path to the note.
        Returns:
            str: The content of the note or an error message with additional context.
        """
        try:
            return self.vault.read_note(note_path)
        except Exception as e:
            parent_dir = os.path.dirname(note_path)
            parent_contents = self.list_notes(parent_dir)
            
            error_message = (
                f"Error reading note '{note_path}': {str(e)}\n\n"
                f"Stack trace:\n{traceback.format_exc()}\n\n"
                f"Contents of parent directory '{parent_dir}':\n{parent_contents}"
            )
            logger.error(error_message)
            return error_message

    def write_note(self, note_path: str, content: str) -> str:
        """
        Write a new note or overwrite an existing one.
        Args:
            note_path (str): The path where to write the note.
            content (str): The content of the note.
        Returns:
            str: Confirmation message.
        """
        self.vault.write_note(note_path, content)
        return f"Note written to {note_path}"

    def move_note(self, source_path: str, dest_path: str) -> str:
        """
        Move a note from one location to another.
        Args:
            source_path (str): The current path of the note.
            dest_path (str): The destination path for the note.
        Returns:
            str: Confirmation message.
        """
        self.vault.move_note(source_path, dest_path)
        return f"Note moved from {source_path} to {dest_path}"

    def fetch_links_from_note(self, source_path: str) -> str:
        """
        Extract all http/https links from a note and store a cached copy of the contents of each link in a
        subdirectory next to the source note.
        
        Args:
            source_path (str): Path to a note to extract links from, relative to the Obsidian vault root.
        
        Returns:
            str: Confirmation message.
        
        Note:
            The linked pages will be stored in a new directory named '{note_name}_linked_pages',
            located in the same directory as the source note. For example:
            - If the source note is at 'Projects/AI/LLMs/MoD.md',
            - The linked pages will be stored in 'Projects/AI/LLMs/MoD_linked_pages/'.
            Each linked page will be saved as an HTML file, with the filename being '{md5 hash of the URL}.html'.
        """
        logger.debug(f"Attempting to fetch HTML content from links found in {source_path}")
        
        # Get the full path of the source note
        source_full_path = Path(self.vault.vault_path) / source_path
        
        # Create a subdirectory for linked pages
        source_dir = source_full_path.parent
        source_name = source_full_path.stem  # Get filename without extension
        dest_dir = source_dir / f"{source_name}_linked_pages"
        
        content = self.read_note(source_path)
        
        self.spider.fetch_all_html_from_string(content, str(dest_dir))
        
        return f"Links fetched and stored in {dest_dir}"

    def get_tools(self):
        logger.debug("get_tools called")
        return [
            FunctionTool.from_defaults(fn=self.get_folder_structure, name="get_folder_structure"),
            FunctionTool.from_defaults(fn=self.query, name="query_rag"),
            FunctionTool.from_defaults(fn=self.list_notes, name="list_notes"),
            FunctionTool.from_defaults(fn=self.read_note, name="read_note"),
            FunctionTool.from_defaults(fn=self.write_note, name="write_note"),
            FunctionTool.from_defaults(fn=self.move_note, name="move_note"),
            FunctionTool.from_defaults(fn=self.fetch_links_from_note, name="fetch_links_from_note"),
            FunctionTool.from_defaults(fn=self.search_memory, name="search_memory"),
        ]
