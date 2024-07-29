# gtd_assistant/app/actions/gtd_actions.py

from typing import List, Dict, Any
from ..file_system.obsidian_interface import ObsidianVault
from ..search.rag_system import RAGSystem
from ..analysis.spider import *


class GTDActions:
    def __init__(self, vault_path: str):
        self.vault = ObsidianVault(vault_path)
        self.rag = RAGSystem(vault_path)
        self.rag.build_index()

    def get_folder_structure(self) -> Dict[str, List[str]]:
        """
        Check the current contents of the Obsidian vault
        :return: dictionary with keys being the folder path (like "Next Actions/Emails" or "Projects/AI/LLMs") and the
        value is a list of all markdown files (ending with ".md") in that folder.
        """
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
        List notes in a specific folder.
        Args:
            folder_path (str): The path to the folder.
        Returns:
            List[str]: A list of note names in the folder.
        """
        return self.vault.list_notes_in_folder(folder_path)

    def read_note(self, note_path: str) -> str:
        """
        Read the content of a specific note.
        Args:
            note_path (str): The path to the note.
        Returns:
            str: The content of the note.
        """
        return self.vault.read_note(note_path)

    def write_note(self, note_path: str, content: str) -> None:
        """
        Write a new note or overwrite an existing one.
        Args:
            note_path (str): The path where to write the note.
            content (str): The content of the note.
        """
        self.vault.write_note(note_path, content)

    def move_note(self, source_path: str, dest_path: str) -> None:
        """
        Move a note from one location to another.
        Args:
            source_path (str): The current path of the note.
            dest_path (str): The destination path for the note.
        """
        self.vault.move_note(source_path, dest_path)

    def fetch_links_from_note(self, source_path: str, dest_dir: str) -> None:
        """
        Extract all http/https links from a note and store a cached copy of the contents of each link in the
        specified folder.
        :param source_path: Path to a note to extract links from
        :param dest_dir: The destination folder to store the cached local copies of pages linked from the note
        :return:
        """
        content = self.read_note(source_path)
        fetch_all_html_from_string(content, dest_dir)
