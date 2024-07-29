# gtd_assistant/app/ui/chat_interface.py

import cmd
from ..agent.gtd_agent import GTDAgent


class GTDAssistant(cmd.Cmd):
    intro = "Welcome to your GTD Assistant. Type 'help' or '?' to list commands."
    prompt = "(GTD) "

    def __init__(self, vault_path: str, model: str, embed_model: str):
        super().__init__()
        self.agent = GTDAgent(vault_path, model, embed_model)

    def default(self, line):
        """Handle any input not recognized as a command."""
        response = self.agent.run(line)
        print(response)

    def do_exit(self, arg):
        """Exit the GTD Assistant"""
        print("Thank you for using GTD Assistant. Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the GTD Assistant"""
        return self.do_exit(arg)


def start_chat_interface(vault_path: str, model: str, embed_model: str):
    GTDAssistant(vault_path, model, embed_model).cmdloop()
