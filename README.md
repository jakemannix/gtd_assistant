# GTD Assistant

GTD Assistant is a powerful tool that integrates with your Obsidian vault to help manage your tasks and projects using the Getting Things Done (GTD) methodology. It uses advanced language models and retrieval-augmented generation to provide intelligent assistance with your GTD workflow.

## Quickstart

Follow these steps to get the GTD Assistant up and running quickly:

1. Clone the repository:
```commandline
git clone https://github.com/yourusername/gtd-assistant.git
cd gtd-assistant
```
2. Create and activate a virtual environment:
```commandline
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```
3. Install the required dependencies:
```commandline
pip install -r requirements.txt
```
4. Create a `.env` file in the project root and add your API keys:
```commandline
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
5. Set up your Obsidian vault path in the `.env` file:
```commandline
VAULT_PATH=/path/to/your/obsidian/vault
```
6. Run the GTD Assistant:
```commandline
python -m gtd_assistant.app.main
```
7. Start interacting with your GTD Assistant! Try commands like:
```commandline
(GTD) List all notes in my Projects folder
(GTD) Create a new note titled "New Project Idea" in the Projects folder
(GTD) Summarize the content of the note "Projects/Ongoing Project"
```

## Features

- Integrates with your Obsidian vault
- Uses advanced language models for intelligent assistance
- Supports multimodal interactions (text and images)
- Helps organize projects, manage next actions, and process inbox items

## Configuration

You can configure the GTD Assistant by modifying the following environment variables in your `.env` file:

- `MODEL`: Choose between "gpt-4o" (default) or "claude-3.5-sonnet"
- `EMBED_MODEL`: Set to "text-embedding-3-small" (default), "text-embedding-3-large", or "local" for offline embedding

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.