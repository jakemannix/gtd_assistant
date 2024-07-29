import requests
import re
import hashlib
from pathlib import Path
import os
from bs4 import BeautifulSoup

# Regular expression to find URLs
URL_PATTERN = re.compile(r'https?://[^\s|\"]+')


def generate_hash(url: str) -> str:
    """Generate SHA-256 hash of the URL."""
    hash_object = hashlib.sha256(url.encode())
    return hash_object.hexdigest()


# Function to fetch and save HTML content
def fetch_and_save_html(url: str, save_dir: str) -> None:
    file_name = generate_hash(url)
    if Path(os.path.join(save_dir, file_name, '.html')).is_file():
        return  # already fetched, contents in target directory
    try:
        text = fetch_html(url)
        # Save the text content in a file with the original URL on the first line, then the rest of the contents
        with open(os.path.join(save_dir, file_name, '.html'), 'w', encoding='utf-8') as file:
            # We should really ensure we're not simply breaking the HTML this way, by inserting it as a link, or
            # something like creating an index of URL <-> cachefile names that we update
            file.write(url + "\n")
            file.write(text)
        print(f"Saved {file_name}")
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {str(e)}")


def fetch_html(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    raw_text = response.text
    soup = BeautifulSoup(raw_text, 'html.parser')
    text = soup.get_text(separator=' ')
    text = ' '.join(text.split())
    return text


def fetch_all_html_from_string(text: str, save_dir: str) -> None:
    for url in URL_PATTERN.findall(text):
        fetch_and_save_html(url, save_dir)


def fetch_all_html_from_file(text_file: str, save_dir: str) -> None:
    with open(text_file, 'r', encoding='utf-8') as file:
        for line in file:
            urls = URL_PATTERN.findall(line)
            for url in urls:
                fetch_and_save_html(url, save_dir)
