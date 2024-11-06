# gtd_assistant/app/ui/chat_interface.py

import cmd
import logging
from ..agent.gtd_agent import GTDAgent
from ..agent.simple_gtd_agent import SimpleGTDAgent
from ..config import Config

logger = logging.getLogger('gtd_assistant')


class GTDAssistant(cmd.Cmd):
    def __init__(self, config: Config):
        super().__init__()
        self.agent = SimpleGTDAgent(config)
        self.intro = "Welcome to your GTD Assistant. Type 'help' or '?' to list commands."
        self.prompt = "(GTD) "

    def process_input(self, line):
        """Process input, handling special commands or passing to default."""
        logger.debug(f"Processing input: {line}")
        command = line.strip().lower()
        
        if command == 'exit' or command == 'quit':
            return self.do_exit(None)
        elif command == 'help' or command == '?':
            return self.do_help(None)
        else:
            return self.default(line)

    def onecmd(self, line):
        """Override onecmd to use our custom input processing."""
        return self.process_input(line)

    def default(self, line):
        """Handle any input not recognized as a special command."""
        logger.debug(f"Running agent with input: {line}")
        response = self.agent.run(line)
        print(response)

    def do_exit(self, arg):
        """Exit the GTD Assistant"""
        print("Thank you for using GTD Assistant. Goodbye!")
        return True

    def do_help(self, arg):
        """Display help information."""
        print("Available commands:")
        print("  exit or quit - Exit the GTD Assistant")
        print("  help or ? - Display this help message")
        print("For any other input, type your request in natural language.")

    # Alias 'quit' to 'exit'
    do_quit = do_exit

    # Alias '?' to 'help'
    do_question = do_help
