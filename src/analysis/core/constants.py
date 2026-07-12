from pathlib import Path


class Constants:
    CHAINED_COT = "Chained CoT"

    DATA_FILE = Path("../../data/clean_data_with_article_info.csv")
    
    ANALYSIS_FOLDER_DIR = Path("../../analysis_reports")
    FIGURE_FOLDER = "figures"
    CSV_FOLDER = "csv"
    
    
    LABEL_MAP = {
        "is-biased": 1,
        "is-not-biased": 0,
    }
    
    BIAS_LABELS_DESCRIPTIONS = [
        "Not Biased",
        "Biased",
    ]
    
    COLORS = {
        "Default": "#4E79A7",
        "Biased": "#4E79A7",
        "Not Biased": "#C7D3E3",
        "Conservative": "#4E79A7",
        "Liberal": "#F28E2B",
        "Independent": "#59A14F",
        "Edge": "black",
        "Direct": "#4E79A7",
        "CoT": "#F28E2B",
        CHAINED_COT: "#59A14F",
        "CoT vs Direct": "#F28E2B",
        f"{CHAINED_COT} vs Direct": "#59A14F",
        f"{CHAINED_COT} vs CoT": "#B07AA1",
        "Not biased → Biased": "#F28E2B",
        "Biased → Not biased": "#4E79A7",
        
        # Metadata conditions
        "source": "#4E79A7",
        "politics": "#F28E2B",
        "source_politics": "#59A14F",
        "source_demographic": "#76B7B2",
        "politics_demographic": "#E15759",
        "source_politics_demographic": "#B07AA1",
    }
    
    PLOT_STYLE = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,

        "figure.dpi": 300,
        "savefig.dpi": 300,

        "ps.fonttype": 42,
    }
    
    MODEL_NAME_MAP = {
        "gemini-2.5-flash-lite": "Gemini 2.5",
        "gemma3_27b": "Gemma3 27B",
        "gpt-4o-mini": "GPT-4o mini",
        "gpt-5.4-mini": "GPT-5.4 mini",
        "llama3.2_3b": "Llama 3.2 3B",
        "llama4_scout": "Llama 4 Scout",
        "phi4-mini": "Phi4 mini",
        "qwen3_30b": "Qwen3 30B",
        "qwen3_4b": "Qwen3 4B",
        "r1-1776_latest": "R1-1776",
        "majority_class_baseline": "Majority Baseline",
        "random_baseline": "Random Baseline",
    }
    
    POLITICS_GROUPS = ["Conservative", "Liberal", "Independent"]

    REASONING_CONDITION_PATHS = {
        "Direct": Path("../../results/direct/direct.1/"),
        "CoT": Path("../../results/cot/cot.1/"),
        CHAINED_COT: Path("../../results/cot/cot.chained/"),
    }

    REASONING_CONDITION_ORDER = ["Direct", "CoT", CHAINED_COT]

    REASONING_COMPARISONS = {
        "cot_vs_direct": "CoT vs Direct",
        "chained_vs_direct": f"{CHAINED_COT} vs Direct",
        "chained_vs_cot": f"{CHAINED_COT} vs CoT",
    }

    REASONING_COMPARISON_ORDER = [
        REASONING_COMPARISONS["cot_vs_direct"],
        REASONING_COMPARISONS["chained_vs_direct"],
        REASONING_COMPARISONS["chained_vs_cot"],
    ]

    REASONING_COMPARISON_COLUMNS = {
        REASONING_COMPARISONS["cot_vs_direct"]: ("direct_output", "cot_output"),
        REASONING_COMPARISONS["chained_vs_direct"]: ("direct_output", "chained_cot_output"),
        REASONING_COMPARISONS["chained_vs_cot"]: ("cot_output", "chained_cot_output"),
    }

    LEGEND_LOCATIONS = {
        "lower_right": "lower right",
    }