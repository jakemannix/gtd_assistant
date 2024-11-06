# gtd_assistant/app/main.py

import os
import argparse
import logging
from dotenv import load_dotenv
from gtd_assistant.config import Config
from gtd_assistant.ui.chat_interface import GTDAssistant

# import debugpy

# Allow the debugger to attach to this process
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached")


def configure_logging(config: Config) -> logging.Logger:
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('gtd_assistant')
    logger.setLevel(logging.DEBUG if config.debug else logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if config.debug else logging.INFO)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    return logger

def main():
    load_dotenv()

    try:
        config = Config.from_args_and_env()
        logger = configure_logging(config)
    except ValueError as e:
        # Fallback logger for configuration errors
        logger = logging.getLogger('gtd_assistant')
        logger.error(f"Configuration error: {e}")
        return

    GTDAssistant(config).cmdloop()

if __name__ == "__main__":
    main()
