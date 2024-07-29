# gtd_assistant/app/actions/gtd_tools.py

from typing import List, Dict
from llama_index.core.tools import FunctionTool
from ..file_system.obsidian_interface import ObsidianVault
from ..search.rag_system import RAGSystem
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GTDTools:
    def __init__(self, vault_path: str, embed_model: str, llm_model: str):
        self.vault = ObsidianVault(vault_path)
        self.rag = RAGSystem(vault_path, embed_model, llm_model)
        self.rag.build_index()

    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        return self.rag.query(question)

    def list_notes(self, folder_path: str) -> List[str]:
        """List notes and folders in a specific folder."""
        logger.debug(f"Attempting to list notes and folders in: {folder_path}")
        try:
            contents = self.vault.list_notes_in_folder(folder_path)
            logger.debug(f"Contents found: {contents}")
            return contents
        except Exception as e:
            logger.error(f"Error listing notes and folders: {str(e)}")
            return []

    def get_folder_structure(self) -> Dict[str, List[str]]:
        # TODO: add comments so the agent knows what this method does.
        logger.debug("Attempting to get current Obsidian vault folder structure")
        structure: Dict[str, List[str]] = self.vault.get_folder_structure()
        return structure

    def fetch_links_from_note(self, note_path):
        logger.debug(f"Attempting to fetch HTML content from links found in {note_path}")
        # FIXME: we are duplicating logic across gtd_tools, gtd_actions, and obsidian_interface
        # FIXME: we don't currently expose an easy way to call gtd_actions.fetch_links_from_note.
        pass

    def read_note(self, note_path: str) -> str:
        """Read the content of a specific note."""
        return self.vault.read_note(note_path)

    def write_note(self, note_path: str, content: str) -> str:
        """Write a new note or overwrite an existing one."""
        self.vault.write_note(note_path, content)
        return f"Note written to {note_path}"

    def move_note(self, source_path: str, dest_path: str) -> str:
        """Move a note from one location to another."""
        self.vault.move_note(source_path, dest_path)
        return f"Note moved from {source_path} to {dest_path}"

    def get_tools(self):
        # TODO: perhaps add in ToolMetadata for each of these?
        return [
            FunctionTool.from_defaults(fn=self.query),
            FunctionTool.from_defaults(fn=self.list_notes),
            FunctionTool.from_defaults(fn=self.read_note),
            FunctionTool.from_defaults(fn=self.write_note),
            FunctionTool.from_defaults(fn=self.move_note),
            FunctionTool.from_defaults(fn=self.get_folder_structure),
        ]
