# gtd_assistant/app/actions/gtd_tools.py

from typing import List, Dict, Callable, Any
from inspect import signature
from llama_index.core.tools import FunctionTool
from llama_index.core.base.llms.types import ChatMessage
from ..file_system.obsidian_interface import ObsidianVault
from ..search.rag_system import RAGSystem
from ..analysis.spider import Spider
from ..agent.memory import DurableSemanticMemory
from ..config import Config
import logging
import os
import traceback
from pathlib import Path

logger = logging.getLogger('gtd_assistant')


class GTDTools:
    def __init__(self, config: Config, memory: DurableSemanticMemory):
        self.vault = ObsidianVault(config.vault_path)
        self.rag = RAGSystem(config=config)
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
        
        Example tool call:

        Action: search_memory
        Action Input: {"query": "What did we discuss about AI last week?"}
        """
        return self.memory.get(query)

    def get_folder_structure(self) -> Dict[str, List[str]]:
        """
        Check the current contents of the Obsidian vault.
        Returns:
            Dict[str, List[str]]: A dictionary with keys being the folder path and
            values being a list of the filenames of all markdown files in that folder.
        
        Generally useful to get a high-level view of the entire set of notes, by note title
        and folder structure.

        Example tool call:

        Action: get_folder_structure
        Action Input: {}
        """
        logger.debug("Attempting to get current Obsidian vault folder structure")
        return self.vault.get_folder_structure()

    def search_gtd_notes(self, question: str) -> List[Dict[str, Any]]:
        """
        Query the GTD index system with a question.
        Args:
            question (str): The question to ask the GTD system.
        Returns:
            List[Dict[str, Any]]: A list of search results, each containing:
                - score: float, similarity score
                - source: str, source file path
                - text: str, content of the note
                - metadata: Dict, any additional metadata
        
        Note: information about the names of notes and their location in the filesystem are
        best retrieved using the get_folder_structure and list_notes tools.  Searching based
        on the note content is best done using this tool.

        Example tool calls:

        if the user is asking "I've been trying to remember what kinds of LLMs I've been researching. 
        What LLM architecture variations do I have notes on?", you
        could call this tool with the following input:

        Action: search_gtd_notes
        Action Input: {"question": "LLM architecture variations"}

        if the user is asking "I'm going shopping soon.What's in my grocery list?", you could call this tool with
        the following input:

        Action: search_gtd_notes
        Action Input: {"question": "grocery list"}
        """
        # nodes = self.rag.retrieve_raw(question)
        # reranked_nodes = self.rag.rerank(query_text=question, nodes=nodes)

        nodes = self.rag.retrieve_and_rerank(question) # TODO: parametrize via the top_k and first_k parameters

        return [{
            'score': node.score,
            'source': node.node.metadata.get('source', 'Unknown'),
            'text': node.node.text,
            'metadata': node.node.metadata
        } for node in nodes]

    def list_notes(self, folder_path: str) -> List[str]:
        """
        List notes and folders in a specific folder.
        Args:
            folder_path (str): The path to the folder.
        Returns:
            List[str]: A list of note names and subfolders in the folder.
        
        Example tool call:

        Action: list_notes
        Action Input: {"folder_path": "GTD/Inbox"}
        """
        logger.debug(f"Attempting to list notes and folders in: {folder_path}")
        try:
            contents = self.vault.list_notes_in_folder(folder_path)
            logger.debug(f"Contents found: {contents}")
            return contents
        except Exception as e:
            logger.error(f"Error listing notes and folders: {str(e)}")
            return []

    # TODO: allow a list of notes to be read at once, to enable the LLM to batch this together
    def read_note(self, note_path: str) -> str:
        """
        Read the content of a specific note.
        Args:
            note_path (str): The path to the note.
        Returns:
            str: The content of the note or an error message with additional context.
        
        Example tool call:

        Action: read_note
        Action Input: {"note_path": "GTD/Next Actions/Emails/Referrals for Adil.md"}
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
            str: Confirmation message or error message with context.
        
        Example tool call:

        Action: write_note
        Action Input: {
            "note_path": "GTD/Next Actions/Errands/20240510_1015_groceries.md",
            "content": "Buy groceries: milk, bread, eggs, and cheese"
        }
        """
        try:
            self.vault.write_note(note_path, content)
            return f"Note written to {note_path}"
        except Exception as e:
            parent_dir = os.path.dirname(note_path)
            try:
                parent_contents = self.list_notes(parent_dir)
            except:
                parent_dir = os.path.dirname(parent_dir)
                parent_contents = self.list_notes(parent_dir)
            
            error_message = (
                f"Error writing note '{note_path}': {str(e)}\n\n"
                f"Stack trace:\n{traceback.format_exc()}\n\n"
                f"Contents of parent directory '{parent_dir}':\n{parent_contents}"
            )
            logger.error(error_message)
            return error_message

    def move_note(self, source_path: str, dest_path: str) -> str:
        """
        Move a note from one location to another.
        Args:
            source_path (str): The current path of the note.
            dest_path (str): The destination path for the note.
        Returns:
            str: Confirmation message or error message with context.
        
        Example tool call:

        Action: move_note
        Action Input: {
            "source_path": "GTD/Inbox/task.md",
            "dest_path": "GTD/Next Actions/task.md"
        }
        """
        try:
            self.vault.move_note(source_path, dest_path)
            return f"Note moved from {source_path} to {dest_path}"
        except Exception as e:
            source_parent = os.path.dirname(source_path)
            dest_parent = os.path.dirname(dest_path)
            try:
                source_contents = self.list_notes(source_parent)
                dest_contents = self.list_notes(dest_parent)
            except:
                source_parent = os.path.dirname(source_parent)
                dest_parent = os.path.dirname(dest_parent)
                source_contents = self.list_notes(source_parent)
                dest_contents = self.list_notes(dest_parent)
            
            error_message = (
                f"Error moving note from '{source_path}' to '{dest_path}': {str(e)}\n\n"
                f"Stack trace:\n{traceback.format_exc()}\n\n"
                f"Contents of source directory '{source_parent}':\n{source_contents}\n\n"
                f"Contents of destination directory '{dest_parent}':\n{dest_contents}"
            )
            logger.error(error_message)
            return error_message

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
        
        Example tool call:

        Action: fetch_links_from_note
        Action Input: {"source_path": "Projects/AI/LLMs/MoD.md"}

        Action: fetch_links_from_note
        Action Input: {"source_path": "GTD/Inbox/20241023_1015_Today from Twitter.md"}
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
            FunctionTool.from_defaults(fn=self.get_folder_structure),
            FunctionTool.from_defaults(fn=self.search_gtd_notes),
            FunctionTool.from_defaults(fn=self.list_notes),
            FunctionTool.from_defaults(fn=self.read_note),
            FunctionTool.from_defaults(fn=self.write_note),
            FunctionTool.from_defaults(fn=self.move_note),
            FunctionTool.from_defaults(fn=self.fetch_links_from_note),
            FunctionTool.from_defaults(fn=self.search_memory),
        ]
