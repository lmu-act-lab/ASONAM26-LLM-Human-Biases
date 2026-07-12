from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

from constants import Constants


class ArticleFetcher:
    """Handles data cleaning, article scraping, and managing intermediate data files."""
    
    def __init__(self, input_file=Constants.DEFAULT_INPUT_FILE, output_dir='data'):
        """Initializes the fetcher with input/output paths."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def _export_data(self, data: pd.DataFrame, file_path: str | Path):
        """Internal helper to export a DataFrame to a CSV file."""
        print(f"Exporting data to {file_path}")
        data.to_csv(Path(file_path), index=False)

    def clean_data(self):
        """Loads the original dataset and preprocesses it by selecting and renaming columns."""
        print(f"Starting data cleaning from {self.input_file}...")
        try:
            data = pd.read_csv(self.input_file)
        except FileNotFoundError:
            print(f"Error: Input file {self.input_file} not found.")
            return

        rename_map = {
            'Answer.age': 'age', 
            'Answer.articleNumber': 'articleNumber',
            'Answer.batch': 'batch', 
            'Answer.bias-question': 'bias-question', 
            'Answer.country': 'country', 
            'Answer.gender': 'gender', 
            'Answer.language1': 'language', 
            'Answer.newsOutlet': 'source', 
            'Answer.politics': 'politics', 
            'Answer.url': 'url'
        }
        
        # Filter and rename safely without slice/copy warnings
        clean_data = data[list(rename_map.keys())].copy().reset_index()
        clean_data.rename(columns=rename_map, inplace=True)
        
        self.data = clean_data
        self._export_data(self.data, Constants.CLEAN_DATA_FILE)
        print("Data cleaning complete.")
        
    def _filter_necessary_text(self, paragraph) -> str:
        """Filters out footnotes and generic recurring news network footers."""
        if 'footnote' in paragraph.get('class', []):
            return ""
            
        text = paragraph.text.strip()
        banned_phrases = {"CLICK HERE TO GET THE FOX NEWS APP", "Stay tuned for all the latest details."}
        return "" if text in banned_phrases else text

    def _get_article_details(self, url: str) -> tuple[str, str]:
        """Scrapes the title and content from a given URL with robust text-fallback parsing."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch {url}. Status code: {response.status_code}")
                return "", ""

            # Leverage built-in chardet-style fallback encoding detection safely
            response.encoding = response.apparent_encoding or 'utf-8'
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Decompose boilerplate and marketing blocks
            for unwanted in soup.find_all("div", attrs={"data-component": "links-block"}):
                unwanted.decompose()

            exclusion_classes = [
                'article-footer', 'site-footer', 'sidebar', 'advertisement', 
                'ad-banner', 'related-articles', 'footnote', 'caption', 'author-bio'
            ]
            for class_name in exclusion_classes:
                for unwanted_element in soup.find_all(class_=class_name):
                    unwanted_element.decompose()

            title_tag = soup.find("h1")
            title = title_tag.text.strip() if title_tag else "No Title Found"
            
            content = []
            body = soup.find("article")
            paragraphs = body.find_all("p") if body else soup.find_all("p", limit=10)
            
            for paragraph in paragraphs:
                text = self._filter_necessary_text(paragraph)
                if text:
                    content.append(text)

            full_content = " ".join(content).strip() if content else "No Content Found"
            return title, full_content
        
        except requests.exceptions.Timeout:
            print(f"Error: Timeout fetching URL: {url}")
            return "", ""
        except Exception as e:
            print(f"Error occurred during scraping {url}: {e}")
            return "", ""

    def fetch_article_info(self) -> pd.DataFrame | None:
        """Scrapes metadata descriptions for unique links and merges them back."""
        clean_file = Path(Constants.CLEAN_DATA_FILE)
        
        if self.data is None and clean_file.exists():
            self.data = pd.read_csv(clean_file)
        elif self.data is None:
            print("Clean data not found. Please run clean_data() first.")
            return None

        print("Starting article detail fetching...")
        data = self.data.dropna(subset=['url']).copy()
        data['article_id'] = data.groupby('url').ngroup()
        
        unique_articles = data[['url', 'article_id']].drop_duplicates().reset_index(drop=True)
        total_unique = len(unique_articles)
        
        scraped_rows = []
        for _, row in unique_articles.iterrows():
            url = row['url']
            article_id = row['article_id']
            
            print(f"Fetching article {article_id + 1}/{total_unique}: {url}")
            title, content = self._get_article_details(url)
            
            scraped_rows.append({
                'article_id': article_id,
                'article_title': title,
                'article_content': content,
                'url': url
            })

        article_info_df = pd.DataFrame(scraped_rows)
        self.data = pd.merge(data, article_info_df, on=['article_id', 'url'], how='left', validate="one_to_one")
        
        self._export_data(article_info_df, Constants.ARTICLES_INFO_FILE)
        self._export_data(self.data, Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
        print("Article detail fetching complete.")
        
        return self.data

    def get_data(self) -> pd.DataFrame | None:
        """Returns the current internal DataFrame."""
        info_file = Path(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
        if self.data is None and info_file.exists():
            self.data = pd.read_csv(info_file)
        return self.data