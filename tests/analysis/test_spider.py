import unittest
import tempfile
import shutil
from pathlib import Path
from gtd_assistant.analysis.spider import Spider, URL_PATTERN

class TestSpider(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp(dir='/tmp/foo', prefix="test_spider_")
        
        # Set up the fake fetcher and spider
        self.fake_html_content = "<html><body>Test content</body></html>"
        self.fake_fetcher = lambda url: self.fake_html_content
        self.spider = Spider(fetch_html_func=self.fake_fetcher)

    def tearDown(self):
        pass
        # Remove the temporary directory and its contents
        # shutil.rmtree(self.temp_dir)

    def test_generate_hash(self):
        url = "https://example.com"
        expected_hash = "100680ad546ce6a577f42f52df33b4cfdca75685"
        self.assertEqual(Spider.generate_hash(url)[:40], expected_hash)

    def test_fetch_and_save_html_new_file(self):
        url = "https://example.com"
        self.spider.fetch_and_save_html(url, self.temp_dir)
        expected_file = Path(self.temp_dir) / f"{Spider.generate_hash(url)}.html"
        self.assertTrue(expected_file.is_file())
        
        # Verify file contents
        with open(expected_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, f"{url}\n{self.fake_html_content}")

    def test_fetch_and_save_html_existing_file(self):
        url = "https://example.com"
        self.spider.fetch_and_save_html(url, self.temp_dir)
        
        # Attempt to fetch again
        self.spider.fetch_and_save_html(url, self.temp_dir)
        
        expected_file = Path(self.temp_dir) / f"{Spider.generate_hash(url)}.html"
        self.assertTrue(expected_file.is_file())
        
        # Verify file contents haven't changed
        with open(expected_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, f"{url}\n{self.fake_html_content}")

    def test_fetch_all_html_from_string_single_url(self):
        text = "Here's a URL: https://example.com"
        self.spider.fetch_all_html_from_string(text, self.temp_dir)
        expected_file = Path(self.temp_dir) / f"{Spider.generate_hash('https://example.com')}.html"
        self.assertTrue(expected_file.is_file())

    def test_fetch_all_html_from_string_multiple_urls(self):
        text = "Multiple URLs: https://example.com http://test.org https://another.net"
        self.spider.fetch_all_html_from_string(text, self.temp_dir)
        expected_files = [
            Path(self.temp_dir) / f"{Spider.generate_hash('https://example.com')}.html",
            Path(self.temp_dir) / f"{Spider.generate_hash('http://test.org')}.html",
            Path(self.temp_dir) / f"{Spider.generate_hash('https://another.net')}.html"
        ]
        for file in expected_files:
            self.assertTrue(file.is_file())

    
if __name__ == '__main__':
    unittest.main()