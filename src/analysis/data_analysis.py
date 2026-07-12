from pathlib import Path
from itertools import combinations

from src.analysis.core.constants import Constants
from src.analysis.core.utils import Utils

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

matplotlib.use("Agg")
plt.rcParams.update(Constants.PLOT_STYLE)

utils = Utils("dataset")
BAR_WIDTH = 0.65
human_df = utils.load_human_data()


def compute_inter_rater_reliability(data_path):
    """Computes pooled and per-pair inter-rater agreement (Cohen's Kappa)."""
    raw_cols = ["WorkerId", "Answer.articleNumber", "Answer.bias-question"]
    df = pd.read_csv(data_path, usecols=raw_cols).rename(
        columns={"Answer.articleNumber": "article_number", "Answer.bias-question": "bias_question"}
    )
    
    df["bias_question"] = df["bias_question"].astype(str).str.strip().str.lower()
    df["label"] = df["bias_question"].map(Constants.LABEL_MAP)
    df = df.dropna(subset=["WorkerId", "article_number", "label"]).copy()
    df["label"] = df["label"].astype(int)
    
    df = df.drop_duplicates(subset=["WorkerId", "article_number"], keep="first")
    
    pair_rows = []
    for _, sub in df.groupby("article_number"):
        if len(sub) < 2:
            continue
        records = sub[["WorkerId", "label"]].to_records(index=False)
        for (w_a, l_a), (w_b, l_b) in combinations(records, 2):
            w_a, w_b = sorted([w_a, w_b])
            pair_rows.append({"WorkerId_A": w_a, "WorkerId_B": w_b, "label_a": int(l_a), "label_b": int(l_b)})
            
    if not pair_rows:
        return pd.DataFrame([{
            "N_Workers": df["WorkerId"].nunique(), "N_Articles": df["article_number"].nunique(),
            "N_Annotations": len(df), "N_Annotation_Pairs": 0, "Pooled_Raw_Agreement": np.nan,
            "Pooled_Cohen_Kappa": np.nan, "Worker_Pairs_With_Overlap": 0, "Worker_Pairs_With_Kappa": 0
        }]), pd.DataFrame(columns=["WorkerId_A", "WorkerId_B", "N_Shared", "Cohen_Kappa", "Raw_Agreement"])

    pair_df = pd.DataFrame(pair_rows)
    pooled_raw = (pair_df["label_a"] == pair_df["label_b"]).mean()
    pooled_kappa = cohen_kappa_score(pair_df["label_a"], pair_df["label_b"])
    
    worker_pair_rows = []
    for (w_a, w_b), sub in pair_df.groupby(["WorkerId_A", "WorkerId_B"]):
        n_shared = len(sub)
        raw_agree = (sub["label_a"] == sub["label_b"]).mean()

        has_variance = sub["label_a"].nunique() >= 2 and sub["label_b"].nunique() >= 2
        kappa = cohen_kappa_score(sub["label_a"], sub["label_b"]) if (n_shared >= 2 and has_variance) else np.nan
        
        worker_pair_rows.append({
            "WorkerId_A": w_a, "WorkerId_B": w_b, "N_Shared": n_shared, "Cohen_Kappa": kappa, "Raw_Agreement": raw_agree
        })
        
    wp_kappa_df = pd.DataFrame(worker_pair_rows)
    summary_df = pd.DataFrame([{
        "N_Workers": df["WorkerId"].nunique(), "N_Articles": df["article_number"].nunique(),
        "N_Annotations": len(df), "N_Annotation_Pairs": len(pair_df),
        "Pooled_Raw_Agreement": pooled_raw, "Pooled_Cohen_Kappa": pooled_kappa,
        "Worker_Pairs_With_Overlap": len(wp_kappa_df), "Worker_Pairs_With_Kappa": wp_kappa_df["Cohen_Kappa"].notna().sum()
    }])
    
    return summary_df, wp_kappa_df


# =========================================================
# Distribution of raters per article
# =========================================================
voters_per_article = human_df.groupby("article_id")["human_label"].count()

fig, ax = plt.subplots(figsize=(3.35, 2.4))
ax.hist(voters_per_article, bins=15, color=Constants.COLORS["Default"], edgecolor=Constants.COLORS["Edge"], linewidth=0.6)
utils.finalize_plot(ax, xlabel="Number of Raters", ylabel="Number of Articles")
utils.save_figure(fig, "voters_distribution")

# =========================================================
# Human label distribution
# =========================================================
rating_counts = human_df["human_label"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(2.5, 2.5))
colors = [Constants.COLORS["Not Biased"], Constants.COLORS["Biased"]]
utils.plot_pie_chart(fig, ax, rating_counts, Constants.BIAS_LABELS_DESCRIPTIONS, 2, colors, "human_label_distribution")

# =========================================================
# Article-level majority distribution
# =========================================================
article_level = human_df.groupby("article_id")["human_label"].mean().reset_index(name="human_bias_rate")
article_level["human_majority"] = (article_level["human_bias_rate"] > 0.5).astype(int)
majority_counts = article_level["human_majority"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(2.4, 2.4))
bars = ax.bar(Constants.BIAS_LABELS_DESCRIPTIONS, majority_counts.values, color=Constants.COLORS["Default"], edgecolor=Constants.COLORS["Edge"], linewidth=0.6, width=BAR_WIDTH)
utils.add_bar_percent_labels(ax, bars, majority_counts.values)
utils.finalize_plot(ax, ylabel="Number of Articles")
utils.save_figure(fig, "article_majority_distribution")

# =========================================================
# Article-level bias rate distribution
# =========================================================
fig, ax = plt.subplots(figsize=(3.35, 2.4))
ax.hist(article_level["human_bias_rate"], bins=12, color=Constants.COLORS["Default"], edgecolor=Constants.COLORS["Edge"], linewidth=0.6)
utils.finalize_plot(ax, xlabel="Proportion labeled biased", ylabel="Number of Articles")
utils.save_figure(fig, "article_bias_rate_distribution")

# =========================================================
# Human consensus distribution
# =========================================================
consensus_df = human_df.groupby("article_id").agg(num_raters=("human_label", "count"), bias_rate=("human_label", "mean")).reset_index()
consensus_df = consensus_df[consensus_df["num_raters"] >= 2].copy()
consensus_df["consensus_rate"] = consensus_df["bias_rate"].apply(lambda x: max(x, 1 - x))

fig, ax = plt.subplots(figsize=(3.35, 2.4))
ax.hist(consensus_df["consensus_rate"], bins=10, color=Constants.COLORS["Default"], edgecolor=Constants.COLORS["Edge"], linewidth=0.6)
ax.set_xlim(0.5, 1.0)
utils.finalize_plot(ax, xlabel="Proportion agreeing with majority label", ylabel="Number of Articles")
utils.save_figure(fig, "human_consensus_distribution")

# =========================================================
# Consensus category distribution
# =========================================================
consensus_df["consensus_category"] = pd.cut(
    consensus_df["consensus_rate"], 
    bins=[0.0, 0.5, 0.60, 0.80, 1.0], 
    labels=["Tie", "Low", "Moderate", "High"], 
    include_lowest=True
)

category_order = ["Tie", "Low", "Moderate", "High"]
consensus_counts = consensus_df["consensus_category"].value_counts().reindex(category_order).fillna(0)

fig, ax = plt.subplots(figsize=(3.35, 2.4))
category_colors = ["#D9D9D9", "#C7D3E3", "#7FA6D6", "#2F5D9B"]
utils.plot_pie_chart(fig, ax, consensus_counts, category_order, 2, category_colors, "human_consensus_categories")

# =========================================================
# Political group distribution
# =========================================================
politics_clean = human_df["politics"].fillna("No response").astype(str).str.strip()
politics_clean = politics_clean.replace(["", "nan", "NaN", "None"], "No response")

politics_order = ["Conservative", "Liberal", "Independent", "No response"]
politics_counts = politics_clean.value_counts().reindex(politics_order).dropna()

fig, ax = plt.subplots(figsize=(3.35, 2.4))
bars = ax.bar(
    politics_counts.index, politics_counts.values,
    color=[Constants.COLORS.get(x, "#999999") for x in politics_counts.index],
    edgecolor=Constants.COLORS["Edge"], linewidth=0.6, width=BAR_WIDTH
)
utils.add_bar_percent_labels(ax, bars, politics_counts.values)
utils.finalize_plot(ax, xlabel="", ylabel="Number of Annotations", rotate_x=30)
utils.save_figure(fig, "political_group_distribution")

# =========================================================
# Human bias rate by political group
# =========================================================
politics_df = human_df.copy()
politics_df["politics_clean"] = politics_clean
politics_groups = ["Conservative", "Liberal", "Independent"]

human_bias_by_politics = (
    politics_df[politics_df["politics_clean"].isin(politics_groups)]
    .groupby("politics_clean")["human_label"].mean()
    .reindex(politics_groups)
)

fig, ax = plt.subplots(figsize=(3.35, 2.4))
bars = ax.bar(human_bias_by_politics.index, human_bias_by_politics.values, color=Constants.COLORS["Default"], edgecolor=Constants.COLORS["Edge"], linewidth=0.6, width=BAR_WIDTH)

for bar, value in zip(bars, human_bias_by_politics.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}", ha="center", va="bottom")

ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
utils.finalize_plot(ax, ylabel="Human Bias Detection Rate")
utils.save_figure(fig, "human_bias_rate_by_politics")

# =========================================================
# Human crowdworker agreement (Cohen's kappa) & File Saves
# =========================================================
summary_row, worker_pair_kappa = compute_inter_rater_reliability(Path("../../data/news_bias_full_data.csv"))

# Save CSV metrics
utils.save_csv(voters_per_article.describe(), "voters_summary.csv")
utils.save_csv(article_level, "article_level_bias_rates.csv", index=False)
utils.save_csv(consensus_df, "human_consensus_by_article.csv", index=False)
utils.save_csv(consensus_counts, "human_consensus_category_counts.csv", header=["count"])
utils.save_csv(politics_counts, "political_group_counts.csv", header=["count"])
utils.save_csv(human_bias_by_politics, "human_bias_rate_by_politics.csv", header=["bias_rate"])
utils.save_csv(summary_row, "human_crowdworker_kappa_summary.csv", index=False)
utils.save_csv(worker_pair_kappa, "human_crowdworker_worker_pair_kappa.csv", index=False)

# Console Output Logs
print("\nHuman crowdworker agreement:")
print(summary_row.to_string(index=False))

if worker_pair_kappa["Cohen_Kappa"].notna().sum() == 0:
    print("No worker pair had enough shared items for per-pair Cohen's kappa; saved pooled pairwise agreement only.")

print(f"\nSaved figures to:\n{utils.figure_dir}")
print(f"\nSaved CSV files to:\n{utils.csv_dir}")