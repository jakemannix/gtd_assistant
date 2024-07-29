# gtd_assistant/app/main.py

import os
import argparse
from dotenv import load_dotenv
from io import StringIO
from .ui.chat_interface import start_chat_interface


def main():
    parser = argparse.ArgumentParser(description="GTD Assistant")
    parser.add_argument("--env", help="Path to .env file")
    parser.add_argument("--env-string", help="Environment variables as a string")
    args = parser.parse_args()

    if args.env_string:
        # Load environment variables from string
        load_dotenv(stream=StringIO(args.env_string), override=True)
    elif args.env:
        # Load environment variables from specified file
        load_dotenv(dotenv_path=args.env, override=True)
    else:
        # Load from default .env file
        load_dotenv(override=True)

    vault_path = os.getenv("VAULT_PATH", os.path.expanduser("~/obsidian_vaults/GTD"))
    model = os.getenv("MODEL", "gpt-4o")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    start_chat_interface(vault_path, model, embed_model)


if __name__ == "__main__":
    main()
