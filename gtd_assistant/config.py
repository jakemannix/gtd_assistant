import os
import argparse
from pydantic import BaseModel, Field

class Config(BaseModel):
    vault_path: str = Field(..., description="Path to the Obsidian vault")
    persist_dir: str = Field(..., description="Path to the persistence directory")
    model: str | None = Field(default=None, description="Model to use")
    embed_model: str | None = Field(default=None, description="Embedding model to use")
    debug: bool = Field(default=False, description="Enable debug mode")
    redis_url: str | None = Field(default=None, description="Redis URL")
    redis_namespace: str | None = Field(default="gtd_assistant", description="Redis namespace for vector store")
    window_size: int = Field(default=2, description="Number of sentences for context window in RAG")

    @classmethod
    def from_args_and_env(cls):
        parser = argparse.ArgumentParser(description="GTD Assistant")
        parser.add_argument("--vault_path", help="Path to the Obsidian vault")
        parser.add_argument("--persist_dir", help="Path to the persistence directory")
        parser.add_argument("--model", help="Model to use")
        parser.add_argument("--embed_model", help="Embedding model to use")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument("--redis_url", help="Redis URL")
        parser.add_argument("--redis_namespace", help="Redis namespace for vector store")
        parser.add_argument("--window_size", type=int, default=2, help="Number of sentences for context window in RAG")
        args = parser.parse_args()

        return cls(
            vault_path=args.vault_path or os.getenv("VAULT_PATH"),
            persist_dir=args.persist_dir or os.getenv("RAG_PERSIST_DIR"),
            model=args.model or os.getenv("MODEL"),
            embed_model=args.embed_model or os.getenv("EMBED_MODEL"),
            debug=args.debug or os.getenv("DEBUG", "False").lower() in ("true", "1", "yes", "on"),
            redis_url=args.redis_url or os.getenv("REDIS_URL"),
            redis_namespace=args.redis_namespace or os.getenv("REDIS_NAMESPACE"),
            window_size=args.window_size or 2
        ) 