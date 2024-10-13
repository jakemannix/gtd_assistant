import requests
import re
import hashlib
from pathlib import Path
import os
from bs4 import BeautifulSoup
from typing import Callable

# Regular expression to find URLs
URL_PATTERN = re.compile(r'https?://[^\s|\"]+')

class Spider:
    def __init__(self, fetch_html_func: Callable[[str], str] = None):
        self.fetch_html = fetch_html_func or self._default_fetch_html

    @staticmethod
    def generate_hash(url: str) -> str:
        """Generate SHA-256 hash of the URL."""
        hash_object = hashlib.sha256(url.encode())
        return hash_object.hexdigest()

    def fetch_and_save_html(self, url: str, save_dir: str) -> None:
        file_name = self.generate_hash(url)
        save_path = Path(save_dir)
        file_path = save_path / f"{file_name}.html"
        
        # Create the directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)
        
        if file_path.is_file():
            return  # already fetched, contents in target directory
        try:
            text = self.fetch_html(url)
            # Save the text content in a file with the original URL on the first line, then the rest of the contents
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"{url}\n{text}")
            print(f"Saved {file_name}")
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {str(e)}")

    @staticmethod
    def _default_fetch_html(url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        raw_text = response.text
        soup = BeautifulSoup(raw_text, 'html.parser')
        text = soup.get_text(separator=' ')
        text = ' '.join(text.split())
        return text

    def fetch_all_html_from_string(self, text: str, save_dir: str) -> None:
        """
        Fetch and save HTML content for all URLs found in the given text string.

        This method searches for URLs in the provided text using a regular expression
        pattern. For each URL found, it calls the fetch_and_save_html method to
        retrieve the HTML content and save it to the specified directory.

        Args:
            text (str): The input text string to search for URLs.
            save_dir (str): The directory path where the HTML content will be saved.
                            If the directory doesn't exist, it will be created.

        Returns:
            None

        Note:
            - The method uses the global URL_PATTERN regular expression to find URLs.
            - Each URL's content is saved in a separate file within the save_dir.
            - If a URL has already been fetched, it will not be fetched again.
        """
        for url in URL_PATTERN.findall(text):
            self.fetch_and_save_html(url, save_dir)

    def fetch_all_html_from_file(self, text_file: str, save_dir: str) -> None:
        with open(text_file, 'r', encoding='utf-8') as file:
            for line in file:
                urls = URL_PATTERN.findall(line)
                for url in urls:
                    self.fetch_and_save_html(url, save_dir)

# Example usage:
# spider = Spider()
# spider.fetch_all_html_from_string("Here's a URL: https://example.com", "/tmp/save_dir")
