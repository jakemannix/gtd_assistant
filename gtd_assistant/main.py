# gtd_assistant/app/main.py

import os
import argparse
import logging
from dotenv import load_dotenv
from gtd_assistant.ui.chat_interface import start_chat_interface

# import debugpy

# Allow the debugger to attach to this process
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached")


def configure_logging():
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING)

    # Create a custom logger for your application
    logger = logging.getLogger('gtd_assistant')
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    # Create a console handler and set its level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a formatter that includes the logger name
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the console handler
    ch.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(ch)

    return logger

def main():
    # Load environment variables
    load_dotenv()

    # Configure logging
    logger = configure_logging()

    parser = argparse.ArgumentParser(description="GTD Assistant")
    parser.add_argument("--vault_path", help="Path to the Obsidian vault")
    parser.add_argument("--persist_dir", help="Path to the persistence directory")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--embed_model", help="Embedding model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--redis_url", help="Redis URL")
    args = parser.parse_args()

    # Use environment variables as defaults, override with command-line args if provided
    vault_path = args.vault_path or os.getenv("VAULT_PATH")
    persist_dir = args.persist_dir or os.getenv("RAG_PERSIST_DIR")
    model = args.model or os.getenv("MODEL")
    embed_model = args.embed_model or os.getenv("EMBED_MODEL")
    debug = args.debug or os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    redis_url = args.redis_url or os.getenv("REDIS_URL")

    if not vault_path or not persist_dir:
        logger.error("VAULT_PATH and RAG_PERSIST_DIR must be set in .env file or provided as command-line arguments")
        return

    start_chat_interface(
        vault_path=vault_path,
        persist_dir=persist_dir,
        model=model,
        embed_model=embed_model,
        debug=debug,
        redis_url=redis_url
    )

if __name__ == "__main__":
    main()
