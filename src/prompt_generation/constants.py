class Constants:
    NEW_SOURCES = ['BBC', 'Fox News', 'CNN']
    NEW_POLITICS = ['Conservative', 'Liberal', 'Independent', 'Other']

    # Default file paths
    DATA_DIR = '../../data/'
    DEFAULT_INPUT_FILE = f'{DATA_DIR}news_bias_full_data.csv'
    CLEAN_DATA_FILE = f'{DATA_DIR}clean_original_data.csv'
    CLEAN_DATA_FILE_WITH_ARTICLE_INFO = f'{DATA_DIR}clean_data_with_article_info.csv'
    SAMPLE_ROWS_DATA_FILE = f'{DATA_DIR}sampled_rows_10pct.csv'
    ARTICLES_INFO_FILE = f'{DATA_DIR}data_articles_info.csv'
    DEFAULT_PROMPT_DIR = f'{DATA_DIR}prompts/'
    DEFAULT_OUTPUT_DIR = '../../results/'
    DEFAULT_LLM_OUTPUT_FOLDER = "llm_outputs/"
    DEFAULT_PROMPT_ARTICLE_INFO_FILE = 'prompt_article_info.csv'
    DEFAULT_PROMPT_POLITICS_FILE = 'prompt_politics_variants.csv'
    DEFAULT_PROMPT_SOURCE_FILE = 'prompt_source_variants.csv'
    DEFAULT_PROMPT_SOURCE_POLITICS_FILE = 'prompt_source_politics_variants.csv'
    DEFAULT_PROMPT_POLITICS_PII_FILE = 'prompt_politics_pii_variants.csv'
    DEFAULT_PROMPT_SOURCE_PII_FILE = 'prompt_source_pii_variants.csv'
    DEFAULT_PROMPT_PII_COMBINED_ALL_FILE = 'prompt_pii_combined_variants.csv'
    DEFAULT_PROMPT_TEST_TWO_QUERIES_FILE = 'prompt_test_two_queries.csv'
    
    DEFAULT_ANALYSIS_FOLDER = 'analysis_reports/'
    
    LLM_RESULT_FILE_PREFIX = 'llm_output'
    PROMPT_FILE_MAP = {
        'articles_info': DEFAULT_PROMPT_ARTICLE_INFO_FILE,
        'politics_variants': DEFAULT_PROMPT_POLITICS_FILE,
        'sources': DEFAULT_PROMPT_SOURCE_FILE,
        'source_politics': DEFAULT_PROMPT_SOURCE_POLITICS_FILE,
        'pii_combined_all': DEFAULT_PROMPT_PII_COMBINED_ALL_FILE,
        'politics_pii': DEFAULT_PROMPT_POLITICS_PII_FILE,
        'source_pii': DEFAULT_PROMPT_SOURCE_PII_FILE,
        'test_two_queries': DEFAULT_PROMPT_TEST_TWO_QUERIES_FILE,
    }
    
    MODEL_NAME = "llama3.1:8b"
    VERSION = 'v5'
    BATCH_SIZE = 4
    TOKEN_LIMIT = 4096
    CONTEXT_LENGTH = 4096
    
    SAMPLE_FRACTION = 0.10
    RANDOM_SEED = 42