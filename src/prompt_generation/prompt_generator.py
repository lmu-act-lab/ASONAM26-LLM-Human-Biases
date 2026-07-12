import argparse
from pathlib import Path
import pandas as pd

from article_fetcher import ArticleFetcher
from constants import Constants


class ArticlePromptGenerator:
    """Generates various LLM prompts based on processed article data from original data arrays."""

    ARTICLE_INFO_TEMPLATE = "Article Title: {title}\nArticle Content: {content}\n"
    VIEWPOINT_PROMPT = "Assume the viewpoint of an individual who is **{politics}**."
    PII_PROMPT = "**Demographics:** A {age}-year-old {gender} {language} speaker from {country}."
    SOURCE_AWARENESS_PROMPT = "**Source Awareness:** Aware this article is **from the publication or organization {source}**."
    CONTEXT_START = "Analyze the following article based on this reader context."
    CONTEXT_START_BLANK = "Assume the viewpoint of an **unspecified reader**. Analyze the following article."
    
    def __init__(self, output_dir='data', context_length=2048):
        """Initializes the generator with context length bounds and destination directories."""
        self.output_dir = Path(output_dir)
        self.context_length = context_length

    def _export_data(self, data: pd.DataFrame, file_path: Path):
        """Internal helper to export a DataFrame to a CSV file."""
        print(f"Exporting data to {file_path}")
        data.to_csv(file_path, index=False)

    def _create_article_info_text(self, row: dict) -> str:
        """Helper to create the Article Title/Content block."""
        return self.ARTICLE_INFO_TEMPLATE.format(
            title=row.get('article_title', ''),
            content=row.get('article_content', '')
        )
        
    def _create_pii_context_text(self, row: dict) -> str:
        """Helper to create the Reader PII info block."""
        return self.PII_PROMPT.format(
            age=row.get('age', ''),
            gender=row.get('gender', ''),
            language=row.get('language', ''),
            country=row.get('country', '')
        )
    
    def _build_user_prompt_context(self, row: dict, include_source=False, include_politics=False, include_pii=False) -> str:
        """Helper to build the introductory context part of the User Prompt."""
        if not (include_source or include_politics or include_pii):
            return self.CONTEXT_START_BLANK
        
        context_parts = []
        
        if include_politics:
            context_parts.append(self.VIEWPOINT_PROMPT.format(politics=row.get('politics', '')))
        else:
            context_parts.append("Assume the viewpoint of a reader with **unspecified politics**.")
        
        if include_source:
            context_parts.append(self.SOURCE_AWARENESS_PROMPT.format(source=row.get('source', '')))
            
        if include_pii:
            context_parts.append(self._create_pii_context_text(row))
            
        context_parts.append(self.CONTEXT_START)
        return "\n".join(context_parts).strip()

    def _generate_prompts_with_context(self, data: pd.DataFrame, file_name: str, context_config: dict):
        """Internal helper to safely iterate over text structures and output configuration batches."""
        active_contexts = [k for k, v in context_config.items() if v and k != 'output_cols']
        print(f"Generating {file_name} with context: {active_contexts}...")
        
        prompt_data = []
        include_source = context_config.get('include_source', False)
        include_politics = context_config.get('include_politics', False)
        include_pii = context_config.get('include_pii', False)
        output_cols = context_config.get('output_cols', [])

        for idx, row in zip(data.index, data.to_dict(orient='records')):
            user_context = self._build_user_prompt_context(
                row, include_source=include_source, include_politics=include_politics, include_pii=include_pii
            )
            prompt = f"{user_context}\n{self._create_article_info_text(row)}"

            if self.context_length > 0 and len(prompt) > self.context_length:
                prompt = prompt[:self.context_length]
            
            output_row = {
                'article_id': row.get('article_id'),
                'index': idx,
                'prompt': prompt,
            }
            for col in output_cols:
                output_row[col] = row.get(col)

            prompt_data.append(output_row)

        output_path = self.output_dir / file_name
        self._export_data(pd.DataFrame(prompt_data), output_path)

    # --- Matrix Config Declarations Mapping ---

    def generate_article_info_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_ARTICLE_INFO_FILE, {
            'include_source': False, 'include_politics': False, 'include_pii': False, 'output_cols': []
        })

    def generate_source_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_SOURCE_FILE, {
            'include_source': True, 'include_politics': False, 'include_pii': False, 'output_cols': ['source']
        })
    
    def generate_politics_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_POLITICS_FILE, {
            'include_source': False, 'include_politics': True, 'include_pii': False, 'output_cols': ['politics']
        })
        
    def generate_source_politics_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_SOURCE_POLITICS_FILE, {
            'include_source': True, 'include_politics': True, 'include_pii': False, 'output_cols': ['source', 'politics']
        })
    
    def generate_source_pii_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_SOURCE_PII_FILE, {
            'include_source': True, 'include_politics': False, 'include_pii': True, 'output_cols': ['source', 'age', 'gender']
        })
        
    def generate_politics_pii_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_POLITICS_PII_FILE, {
            'include_source': False, 'include_politics': True, 'include_pii': True, 'output_cols': ['politics', 'age', 'gender']
        })
        
    def generate_pii_combined_all_prompts(self, data):
        self._generate_prompts_with_context(data, Constants.DEFAULT_PROMPT_PII_COMBINED_ALL_FILE, {
            'include_source': True, 'include_politics': True, 'include_pii': True, 'output_cols': ['age', 'gender', 'source', 'politics']
        })

    def generate_all_prompts(self, data):
        """Generates all seven custom variations of LLM prompts."""
        print("Starting prompt generation and segmentation into files...")
        self.generate_article_info_prompts(data)
        self.generate_politics_prompts(data)
        self.generate_source_prompts(data)
        self.generate_pii_combined_all_prompts(data)
        self.generate_source_politics_prompts(data)
        self.generate_source_pii_prompts(data)
        self.generate_politics_pii_prompts(data)
        print("All prompt files generated successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate LLM prompts from news bias data with selective execution.")
    parser.add_argument('--clean', action='store_true', help="Run data cleaning step.")
    parser.add_argument('--fetch', action='store_true', help="Run article fetching step.")
    parser.add_argument('--input-file', type=str, default=Constants.DEFAULT_INPUT_FILE, help="Path to input CSV.")
    parser.add_argument('--output-dir', type=str, default=Constants.DEFAULT_PROMPT_DIR, help="Base output path directory.")
    parser.add_argument('--version', type=str, default='v5', help="Version target layout suffix.")
    parser.add_argument('--context-length', type=int, default=-1, help="Max context string length cut thresholds.")
    
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument('--all-prompts', action='store_true', help="Generate all prompt files.")
    prompt_group.add_argument('--prompts', nargs='+', choices=[
        'articles_info', 'politics', 'sources', 'pii_combined_all', 'politics_pii', 'source_pii', 'source_politics'
    ], help="Specify subset targets.")
    
    args = parser.parse_args()
    resolved_output_dir = Path(args.output_dir) / args.version

    fetcher = ArticleFetcher(input_file=args.input_file, output_dir=resolved_output_dir)
    generator = ArticlePromptGenerator(output_dir=resolved_output_dir, context_length=args.context_length)

    if args.clean:
        fetcher.clean_data()
        
    if args.fetch:
        fetcher.fetch_article_info()
    
    data = fetcher.get_data()
    
    if args.prompts or args.all_prompts:
        if data is None:
            print("Data execution state array is empty. Please verify dependencies via execution of --clean and --fetch.")
            exit(1)

        prompt_methods_map = {
            'articles_info': generator.generate_article_info_prompts,
            'politics': generator.generate_politics_prompts,
            'sources': generator.generate_source_prompts,
            'pii_combined_all': generator.generate_pii_combined_all_prompts,
            'politics_pii': generator.generate_politics_pii_prompts,
            'source_pii': generator.generate_source_pii_prompts,
            'source_politics': generator.generate_source_politics_prompts,
        }

        if args.all_prompts:
            generator.generate_all_prompts(data)
        elif args.prompts:
            print("Starting prompt generation based on explicit parameters...")
            for method_key in args.prompts:
                prompt_methods_map[method_key](data)
            print("Selected prompt targets built successfully.")