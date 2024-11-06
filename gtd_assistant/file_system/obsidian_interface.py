# gtd_assistant/app/file_system/obsidian_interface.py

import os
from pathlib import Path
from typing import List, Dict
import logging
import os

logger = logging.getLogger('gtd_assistant')


class ObsidianVault:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")
        logger.debug(f"Initialized ObsidianVault with path: {self.vault_path}")

    # TODO: add support for non-markdown files
    def get_folder_structure(self) -> Dict[str, List[str]]:
        structure = {}
        for root, dirs, files in os.walk(self.vault_path):
            relative_path = Path(root).relative_to(self.vault_path)
            textfiles = [f for f in files if f.endswith('.md')]
            if len(textfiles) > 0:
                structure[str(relative_path)] = textfiles
        return structure

    def read_note(self, note_path: str) -> str:
        full_path = self.vault_path / note_path
        if not full_path.exists():
            raise FileNotFoundError(f"Note not found: {note_path}")
        with open(full_path, 'r', encoding='utf-8') as file:
            return file.read()

    def write_note(self, note_path: str, content: str):
        full_path = self.vault_path / note_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as file:
            file.write(content)

    def move_note(self, source_path: str, dest_path: str):
        source = self.vault_path / source_path
        dest = self.vault_path / dest_path
        if not source.exists():
            raise FileNotFoundError(f"Source note not found: {source_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)

    def list_notes_in_folder(self, folder_path: str) -> List[str]:
        full_path = os.path.join(self.vault_path, folder_path)
        logger.debug(f"Attempting to list notes and folders in: {full_path}")
        if not os.path.exists(full_path):
            logger.error(f"Folder not found: {full_path}")
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        contents = []
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)
            if os.path.isdir(item_path):
                contents.append(f"Folder: {item}")
            elif os.path.isfile(item_path):
                contents.append(f"File: {item}")

        logger.debug(f"Contents found: {contents}")
        return contents


