from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from core.constants import Constants
from core.utils import Utils

matplotlib.use("Agg")
plt.rcParams.update(Constants.PLOT_STYLE)

utils = Utils("dataset")
BAR_WIDTH = 0.65
human_df = utils.load_human_data()


def compute_inter_rater_reliability(data_path, n_simulations=10_000, random_state=42):
    """Estimate an article-balanced sampled human-human Cohen's kappa using vectorized simulation."""
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    raw_cols = ["WorkerId", "article_id", "Answer.bias-question"]
    df = pd.read_csv(data_path, usecols=raw_cols).rename(columns={"Answer.bias-question": "bias_question"})

    df["WorkerId"] = df["WorkerId"].astype("string").str.strip()
    df["article_id"] = df["article_id"].astype("string").str.strip()
    df["bias_question"] = df["bias_question"].astype("string").str.strip().str.lower()

    norm_label_map = {str(k).strip().lower(): int(v) for k, v in Constants.LABEL_MAP.items()}
    df["label"] = df["bias_question"].map(norm_label_map)
    df.dropna(subset=["WorkerId", "article_id", "label"], inplace=True)
    df = df[df["WorkerId"].ne("") & df["article_id"].ne("")].copy()
    df["label"] = df["label"].astype(np.int8)

    dup_mask = df.duplicated(subset=["WorkerId", "article_id"], keep=False)
    n_duplicate_rows = int(dup_mask.sum())
    conflicting_dup_groups = df.loc[dup_mask].groupby(["WorkerId", "article_id"])["label"].nunique().gt(1).sum()
    df.drop_duplicates(subset=["WorkerId", "article_id"], keep="first", inplace=True)

    article_df = df.groupby("article_id")["label"].agg(N_Raters="size", N_Biased="sum").reset_index()
    article_df["N_Not_Biased"] = article_df["N_Raters"] - article_df["N_Biased"]
    article_df["Human_Bias_Rate"] = article_df["N_Biased"] / article_df["N_Raters"]
    article_df["Consensus_Rate"] = article_df[["N_Biased", "N_Not_Biased"]].max(axis=1) / article_df["N_Raters"]
    article_df["Eligible_For_Sampled_Kappa"] = article_df["N_Raters"] >= 2

    num = article_df["N_Biased"] * (article_df["N_Biased"] - 1) + article_df["N_Not_Biased"] * (article_df["N_Not_Biased"] - 1)
    den = article_df["N_Raters"] * (article_df["N_Raters"] - 1)
    article_df["Pairwise_Raw_Agreement"] = np.where(den > 0, num / den, np.nan)

    n_workers = df["WorkerId"].nunique()
    n_total_articles = df["article_id"].nunique()
    eligible_df = article_df[article_df["Eligible_For_Sampled_Kappa"]]
    n_eligible_articles = len(eligible_df)
    n_excluded_articles = n_total_articles - n_eligible_articles

    simulation_columns = ["Simulation", "Cohen_Kappa", "Raw_Agreement", "Pseudo_Rater_A_Bias_Rate", "Pseudo_Rater_B_Bias_Rate", "Expected_Agreement"]

    if n_eligible_articles < 2:
        summary_df = pd.DataFrame([{
            "N_Workers": n_workers, "N_Total_Articles": n_total_articles, "N_Eligible_Articles": n_eligible_articles,
            "N_Excluded_Articles": n_excluded_articles, "N_Annotations": len(df), "N_Duplicate_Rows_Found": n_duplicate_rows,
            "N_Conflicting_Duplicate_Groups": int(conflicting_dup_groups), "N_Simulations_Requested": n_simulations,
            "N_Valid_Simulations": 0, "Mean_Cohen_Kappa": np.nan, "Median_Cohen_Kappa": np.nan, "SD_Cohen_Kappa": np.nan,
            "Kappa_2.5_Percentile": np.nan, "Kappa_25_Percentile": np.nan, "Kappa_75_Percentile": np.nan, "Kappa_97.5_Percentile": np.nan,
            "Mean_Sampled_Raw_Agreement": np.nan, "Mean_Exact_Article_Pairwise_Agreement": np.nan, "Eligible_Article_Bias_Rate": np.nan, "Random_State": random_state
        }])
        return summary_df, pd.DataFrame(columns=simulation_columns), article_df

    eligible_ids = set(eligible_df["article_id"])
    grouped = df[df["article_id"].isin(eligible_ids)].groupby("article_id")
    labels_list = [group["label"].to_numpy(dtype=np.int8) for _, group in sorted(grouped, key=lambda x: x[0])]

    rng = np.random.default_rng(random_state)
    sim_bias_a, sim_bias_b= [], []

    for labels in labels_list:
        n_lbls = len(labels)
        idx_matrix = np.array([rng.choice(n_lbls, size=2, replace=False) for _ in range(n_simulations)])
        lbl_a, lbl_b = labels[idx_matrix[:, 0]], labels[idx_matrix[:, 1]]
        
        # Symmetrical balance randomizer shuffle mask
        swap_mask = rng.random(n_simulations) < 0.5
        lbl_a[swap_mask], lbl_b[swap_mask] = lbl_b[swap_mask], lbl_a[swap_mask]
        
        sim_bias_a.append(lbl_a)
        sim_bias_b.append(lbl_b)

    arr_a = np.column_stack(sim_bias_a)
    arr_b = np.column_stack(sim_bias_b)

    raw_agreements = (arr_a == arr_b).mean(axis=1)
    bias_rates_a = arr_a.mean(axis=1)
    bias_rates_b = arr_b.mean(axis=1)
    expected_agreements = bias_rates_a * bias_rates_b + (1.0 - bias_rates_a) * (1.0 - bias_rates_b)
    
    kappa_denominators = 1.0 - expected_agreements
    kappas = np.where(np.isclose(kappa_denominators, 0.0), np.nan, (raw_agreements - expected_agreements) / kappa_denominators)

    simulation_df = pd.DataFrame({
        "Simulation": np.arange(1, n_simulations + 1), "Cohen_Kappa": kappas, "Raw_Agreement": raw_agreements,
        "Pseudo_Rater_A_Bias_Rate": bias_rates_a, "Pseudo_Rater_B_Bias_Rate": bias_rates_b, "Expected_Agreement": expected_agreements
    })

    valid_kappas = simulation_df["Cohen_Kappa"].dropna()
    eligible_bias_rate = df[df["article_id"].isin(eligible_ids)]["label"].mean()

    summary_df = pd.DataFrame([{
        "N_Workers": n_workers, "N_Total_Articles": n_total_articles, "N_Eligible_Articles": n_eligible_articles,
        "N_Excluded_Articles": n_excluded_articles, "N_Annotations": len(df), "N_Duplicate_Rows_Found": n_duplicate_rows,
        "N_Conflicting_Duplicate_Groups": int(conflicting_dup_groups), "N_Simulations_Requested": n_simulations, "N_Valid_Simulations": len(valid_kappas),
        "Mean_Cohen_Kappa": float(valid_kappas.mean()) if not valid_kappas.empty else np.nan,
        "Median_Cohen_Kappa": float(valid_kappas.median()) if not valid_kappas.empty else np.nan,
        "SD_Cohen_Kappa": float(valid_kappas.std(ddof=1)) if not valid_kappas.empty else np.nan,
        "Kappa_2.5_Percentile": float(valid_kappas.quantile(0.025)) if not valid_kappas.empty else np.nan,
        "Kappa_25_Percentile": float(valid_kappas.quantile(0.25)) if not valid_kappas.empty else np.nan,
        "Kappa_75_Percentile": float(valid_kappas.quantile(0.75)) if not valid_kappas.empty else np.nan,
        "Kappa_97.5_Percentile": float(valid_kappas.quantile(0.975)) if not valid_kappas.empty else np.nan,
        "Mean_Sampled_Raw_Agreement": float(simulation_df["Raw_Agreement"].mean()),
        "Mean_Exact_Article_Pairwise_Agreement": float(article_df[article_df["Eligible_For_Sampled_Kappa"]]["Pairwise_Raw_Agreement"].mean()),
        "Mean_Pseudo_Rater_A_Bias_Rate": float(bias_rates_a.mean()), "Mean_Pseudo_Rater_B_Bias_Rate": float(bias_rates_b.mean()),
        "Eligible_Article_Bias_Rate": float(eligible_bias_rate), "Random_State": random_state
    }])

    return summary_df, simulation_df, article_df


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
# Human crowdworker agreement (Cohen's kappa)
# =========================================================
summary_row, sampled_human_kappa, article_kappa_details = compute_inter_rater_reliability(
    Path("../../data/news_bias_full_data.csv"), n_simulations=10_000, random_state=42
)

utils.save_csv(voters_per_article.describe(), "voters_summary.csv")
utils.save_csv(article_level, "article_level_bias_rates.csv", index=False)
utils.save_csv(consensus_df, "human_consensus_by_article.csv", index=False)
utils.save_csv(consensus_counts, "human_consensus_category_counts.csv", header=["count"])
utils.save_csv(politics_counts, "political_group_counts.csv", header=["count"])
utils.save_csv(human_bias_by_politics, "human_bias_rate_by_politics.csv", header=["bias_rate"])
utils.save_csv(summary_row, "sampled_human_human_kappa_summary.csv", index=False)
utils.save_csv(sampled_human_kappa, "sampled_human_human_kappa_simulations.csv", index=False)
utils.save_csv(article_kappa_details, "sampled_human_human_kappa_articles.csv", index=False)

print("\nSampled human-human Cohen's kappa:")
print(f"\nEligible articles: {int(summary_row.loc[0, 'N_Eligible_Articles'])}")
print(f"Excluded single-rater articles: {int(summary_row.loc[0, 'N_Excluded_Articles'])}")
print(f"Mean sampled human-human κ: {summary_row.loc[0, 'Mean_Cohen_Kappa']:.4f}")
print(f"Median sampled human-human κ: {summary_row.loc[0, 'Median_Cohen_Kappa']:.4f}")
print(f"95% simulation interval: [{summary_row.loc[0, 'Kappa_2.5_Percentile']:.4f}, {summary_row.loc[0, 'Kappa_97.5_Percentile']:.4f}]")
print(f"Mean sampled raw agreement: {summary_row.loc[0, 'Mean_Sampled_Raw_Agreement']:.4f}")

print(f"\nSaved figures to:\n{utils.figure_dir}")
print(f"\nSaved CSV files to:\n{utils.csv_dir}")