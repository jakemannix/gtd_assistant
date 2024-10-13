# gtd_assistant/app/search/obsidian_reader.py

from typing import List, Iterator
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from ..file_system.obsidian_interface import ObsidianVault


class ObsidianReader(BaseReader):
    def __init__(self, vault_path: str):
        self.vault = ObsidianVault(vault_path)

    def lazy_load_data(self, *args, **kwargs) -> Iterator[Document]:
        folder_structure = self.vault.get_folder_structure()

        for folder, files in folder_structure.items():
            for file in files:
                file_path = f"{folder}/{file}"
                content = self.vault.read_note(file_path)
                yield Document(text=content, metadata={"source": file_path})

    def load_data(self, *args, **kwargs) -> List[Document]:
        return list(self.lazy_load_data(*args, **kwargs))

