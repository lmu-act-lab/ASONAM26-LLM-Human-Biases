#!/usr/bin/env python3
"""
Generate cot prompts for sampled_rows_10pct.csv (make sure to run sample_row_selection.py first).

This script:
1. Loads the clean data with article info
2. Filters to the sampled indexes
3. Generates all 7 prompt variants (article_only, +source, +politics, +all, etc.)
4. Exports to data/prompts/cot/
"""

from pathlib import Path
import pandas as pd

from prompt_generator import ArticlePromptGenerator
from constants import Constants


def load_indexes(sample_file=Constants.SAMPLE_ROWS_DATA_FILE) -> list:
    """Load conflict article indexes from analysis output."""
    sample_path = Path(sample_file)
    if not sample_path.exists():
        print(f"ERROR: Conflict file not found: {sample_path}")
        raise FileNotFoundError(f"Missing required metadata dependency: {sample_path}")
    
    conflict_df = pd.read_csv(sample_path)
    indexes = conflict_df['index'].tolist()
    print(f"✓ Loaded {len(indexes)} article indexes from {sample_path}")
    return indexes


def load_file(indexes=None, sample_file=None) -> pd.DataFrame:
    """Load and filter clean data to top conflict articles."""
    print("Loading clean data with article info...")
    
    data_file_path = Path(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)
    if not data_file_path.exists():
        print(f"ERROR: File not found: {data_file_path}")
        raise FileNotFoundError(f"Missing primary corpus dependency: {data_file_path}")
    
    if indexes is None:
        indexes = load_indexes(sample_file or Constants.SAMPLE_ROWS_DATA_FILE)
    
    df = pd.read_csv(data_file_path)
    print(f"Loaded {len(df)} total rows")
    
    filtered_df = df[df['index'].isin(indexes)].copy()
    filtered_df.sort_values('index', inplace=True)
    
    print(f"Filtered to {len(filtered_df)} rows for {len(indexes)} conflict articles")
    print(f"Article indexes: {sorted(filtered_df['index'].unique())}")
    
    return filtered_df


def generate_sample_prompts(data: pd.DataFrame, output_dir='../../data/prompts/cot'):
    """Generate all 7 prompt variants for cot."""
    output_path = Path(output_dir)
    print(f"\nGenerating cot prompts to {output_path}...")
    
    output_path.mkdir(parents=True, exist_ok=True)
    generator = ArticlePromptGenerator(output_dir=output_path, context_length=-1)
    
    generation_pipelines = {
        "Article Info": generator.generate_article_info_prompts,
        "Politics Variant": generator.generate_politics_prompts,
        "Source Variant": generator.generate_source_prompts,
        "PII Combined All": generator.generate_pii_combined_all_prompts,
        "Source + Politics Variant": generator.generate_source_politics_prompts,
        "Source + PII Variant": generator.generate_source_pii_prompts,
        "Politics + PII Variant": generator.generate_politics_pii_prompts
    }
    
    for title, run_pipeline in generation_pipelines.items():
        print(f"\n--- Generating {title} Prompts ---")
        run_pipeline(data)
        
    print("\n✓ All cot prompt files generated successfully!")
    
    print("\nVerifying generated files:")
    for filepath in sorted(output_path.glob('*.csv')):
        df = pd.read_csv(filepath)
        print(f"  {filepath.name}: {len(df)} rows")


if __name__ == '__main__':
    print("=" * 70)
    print("Prompt generation by indexes")
    print("=" * 70)
    
    conflict_data = load_file()
    generate_sample_prompts(conflict_data)
    
    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)