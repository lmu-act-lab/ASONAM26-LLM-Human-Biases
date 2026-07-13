from pathlib import Path
from itertools import combinations

from core.constants import Constants
from core.utils import Utils

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


def compute_inter_rater_reliability(
    data_path,
    n_simulations=10_000,
    random_state=42,
):
    """
    Estimate an article-balanced human-human Cohen's kappa baseline.

    In each simulation:
      1. Keep articles with at least two valid human annotations.
      2. Sample two distinct annotations from each eligible article.
      3. Treat the sampled labels as ratings from two pseudo-raters.
      4. Compute Cohen's kappa across eligible articles.

    Each article contributes exactly one human-human comparison per
    simulation, regardless of whether it has 2 or 45 annotators.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Path to the raw human annotation CSV.

    n_simulations : int, default=10_000
        Number of random pseudo-rater simulations.

    random_state : int, default=42
        Seed used for reproducible sampling.

    Returns
    -------
    summary_df : pandas.DataFrame
        One-row summary of the sampled human-human kappa distribution.

    simulation_df : pandas.DataFrame
        One row per simulation, including Cohen's kappa, raw agreement,
        and the two pseudo-raters' bias prediction rates.
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be at least 1.")

    raw_cols = [
        "WorkerId",
        "Answer.articleNumber",
        "Answer.bias-question",
    ]

    df = (
        pd.read_csv(data_path, usecols=raw_cols)
        .rename(
            columns={
                "Answer.articleNumber": "article_number",
                "Answer.bias-question": "bias_question",
            }
        )
    )

    df["bias_question"] = (
        df["bias_question"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    normalized_label_map = {
        str(label_name).strip().lower(): label_value
        for label_name, label_value in Constants.LABEL_MAP.items()
    }

    df["label"] = df["bias_question"].map(normalized_label_map)

    df = df.dropna(
        subset=[
            "WorkerId",
            "article_number",
            "label",
        ]
    ).copy()

    df["label"] = df["label"].astype(int)

    unexpected_labels = set(df["label"].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(
            "Human labels must be binary values 0 and 1. "
            f"Unexpected values: {sorted(unexpected_labels)}"
        )

    df = df.drop_duplicates(
        subset=["WorkerId", "article_number"],
        keep="first",
    )

    n_workers = df["WorkerId"].nunique()
    n_total_articles = df["article_number"].nunique()
    n_annotations = len(df)

    labels_by_article = {
        article_number: sub["label"].to_numpy(dtype=np.int8)
        for article_number, sub in df.groupby(
            "article_number",
            sort=True,
        )
        if len(sub) >= 2
    }

    n_eligible_articles = len(labels_by_article)
    n_single_rater_articles = (
        n_total_articles - n_eligible_articles
    )

    if n_eligible_articles < 2:
        empty_summary = pd.DataFrame(
            [
                {
                    "N_Workers": n_workers,
                    "N_Total_Articles": n_total_articles,
                    "N_Eligible_Articles": n_eligible_articles,
                    "N_Excluded_Single_Rater_Articles": (
                        n_single_rater_articles
                    ),
                    "N_Annotations": n_annotations,
                    "N_Simulations": n_simulations,
                    "N_Valid_Simulations": 0,
                    "Mean_Cohen_Kappa": np.nan,
                    "Median_Cohen_Kappa": np.nan,
                    "SD_Cohen_Kappa": np.nan,
                    "Kappa_2.5_Percentile": np.nan,
                    "Kappa_25_Percentile": np.nan,
                    "Kappa_75_Percentile": np.nan,
                    "Kappa_97.5_Percentile": np.nan,
                    "Mean_Raw_Agreement": np.nan,
                    "Mean_Pseudo_Rater_A_Bias_Rate": np.nan,
                    "Mean_Pseudo_Rater_B_Bias_Rate": np.nan,
                }
            ]
        )

        empty_simulations = pd.DataFrame(
            columns=[
                "Simulation",
                "Cohen_Kappa",
                "Raw_Agreement",
                "Pseudo_Rater_A_Bias_Rate",
                "Pseudo_Rater_B_Bias_Rate",
            ]
        )

        return empty_summary, empty_simulations

    article_ids = list(labels_by_article.keys())
    rng = np.random.default_rng(random_state)

    simulation_rows = []

    # ---------------------------------------------------------
    # Repeatedly construct two pseudo-human raters
    # ---------------------------------------------------------
    for simulation_number in range(1, n_simulations + 1):
        pseudo_rater_a = np.empty(
            n_eligible_articles,
            dtype=np.int8,
        )
        pseudo_rater_b = np.empty(
            n_eligible_articles,
            dtype=np.int8,
        )

        for article_position, article_id in enumerate(article_ids):
            article_labels = labels_by_article[article_id]

            # Sampling without replacement guarantees that the two
            # labels come from two distinct human annotation rows.
            selected_indices = rng.choice(
                len(article_labels),
                size=2,
                replace=False,
            )

            pseudo_rater_a[article_position] = article_labels[
                selected_indices[0]
            ]
            pseudo_rater_b[article_position] = article_labels[
                selected_indices[1]
            ]

        raw_agreement = np.mean(
            pseudo_rater_a == pseudo_rater_b
        )

        kappa = cohen_kappa_score(
            pseudo_rater_a,
            pseudo_rater_b,
        )

        # Kappa can be undefined when expected agreement is exactly 1,
        # such as when both sampled raters use only one label.
        if not np.isfinite(kappa):
            kappa = np.nan

        simulation_rows.append(
            {
                "Simulation": simulation_number,
                "Cohen_Kappa": kappa,
                "Raw_Agreement": raw_agreement,
                "Pseudo_Rater_A_Bias_Rate": (
                    pseudo_rater_a.mean()
                ),
                "Pseudo_Rater_B_Bias_Rate": (
                    pseudo_rater_b.mean()
                ),
            }
        )

    simulation_df = pd.DataFrame(simulation_rows)
    valid_kappas = simulation_df["Cohen_Kappa"].dropna()

    # ---------------------------------------------------------
    # Exact article-balanced pairwise raw agreement
    # ---------------------------------------------------------
    #
    # For article i:
    #
    #   agreement_i =
    #       [n0(n0 - 1) + n1(n1 - 1)] /
    #       [n(n - 1)]
    #
    # This is the probability that two distinct randomly selected
    # annotations from the article agree.
    article_agreements = []

    for article_labels in labels_by_article.values():
        n_raters = len(article_labels)
        n_biased = int(article_labels.sum())
        n_not_biased = n_raters - n_biased

        article_pairwise_agreement = (
            n_biased * (n_biased - 1)
            + n_not_biased * (n_not_biased - 1)
        ) / (n_raters * (n_raters - 1))

        article_agreements.append(article_pairwise_agreement)

    mean_exact_pairwise_agreement = float(
        np.mean(article_agreements)
    )

    if valid_kappas.empty:
        mean_kappa = np.nan
        median_kappa = np.nan
        sd_kappa = np.nan
        q025 = np.nan
        q25 = np.nan
        q75 = np.nan
        q975 = np.nan
    else:
        mean_kappa = valid_kappas.mean()
        median_kappa = valid_kappas.median()
        sd_kappa = valid_kappas.std(ddof=1)
        q025 = valid_kappas.quantile(0.025)
        q25 = valid_kappas.quantile(0.25)
        q75 = valid_kappas.quantile(0.75)
        q975 = valid_kappas.quantile(0.975)

    summary_df = pd.DataFrame(
        [
            {
                "N_Workers": n_workers,
                "N_Total_Articles": n_total_articles,
                "N_Eligible_Articles": n_eligible_articles,
                "N_Excluded_Single_Rater_Articles": (
                    n_single_rater_articles
                ),
                "N_Annotations": n_annotations,
                "N_Simulations": n_simulations,
                "N_Valid_Simulations": len(valid_kappas),
                "Mean_Cohen_Kappa": mean_kappa,
                "Median_Cohen_Kappa": median_kappa,
                "SD_Cohen_Kappa": sd_kappa,
                "Kappa_2.5_Percentile": q025,
                "Kappa_25_Percentile": q25,
                "Kappa_75_Percentile": q75,
                "Kappa_97.5_Percentile": q975,
                "Mean_Sampled_Raw_Agreement": (
                    simulation_df["Raw_Agreement"].mean()
                ),
                "Mean_Exact_Pairwise_Raw_Agreement": (
                    mean_exact_pairwise_agreement
                ),
                "Mean_Pseudo_Rater_A_Bias_Rate": (
                    simulation_df[
                        "Pseudo_Rater_A_Bias_Rate"
                    ].mean()
                ),
                "Mean_Pseudo_Rater_B_Bias_Rate": (
                    simulation_df[
                        "Pseudo_Rater_B_Bias_Rate"
                    ].mean()
                ),
                "Random_State": random_state,
            }
        ]
    )

    return summary_df, simulation_df


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
summary_row, sampled_human_kappa = compute_inter_rater_reliability(
    Path("../../data/news_bias_full_data.csv"),
    n_simulations=10_000,
    random_state=42,
)

# Save CSV metrics
utils.save_csv(voters_per_article.describe(), "voters_summary.csv")
utils.save_csv(article_level, "article_level_bias_rates.csv", index=False)
utils.save_csv(consensus_df, "human_consensus_by_article.csv", index=False)
utils.save_csv(consensus_counts, "human_consensus_category_counts.csv", header=["count"])
utils.save_csv(politics_counts, "political_group_counts.csv", header=["count"])
utils.save_csv(human_bias_by_politics, "human_bias_rate_by_politics.csv", header=["bias_rate"])
utils.save_csv(summary_row, "sampled_human_human_kappa_summary.csv",index=False)
utils.save_csv(sampled_human_kappa, "sampled_human_human_kappa_simulations.csv",index=False)

print("\nSampled human-human agreement:")
print(summary_row.to_string(index=False))

valid_simulations = sampled_human_kappa[
    "Cohen_Kappa"
].notna().sum()

if valid_simulations == 0:
    print(
        "Cohen's kappa was undefined in all pseudo-rater "
        "simulations."
    )
else:
    mean_kappa = summary_row.loc[
        0,
        "Mean_Cohen_Kappa",
    ]
    median_kappa = summary_row.loc[
        0,
        "Median_Cohen_Kappa",
    ]
    lower = summary_row.loc[
        0,
        "Kappa_2.5_Percentile",
    ]
    upper = summary_row.loc[
        0,
        "Kappa_97.5_Percentile",
    ]
    eligible = summary_row.loc[
        0,
        "N_Eligible_Articles",
    ]

    print(
        f"\nHuman-human sampled Cohen's kappa "
        f"across {eligible} eligible articles:"
    )
    print(f"  Mean: {mean_kappa:.4f}")
    print(f"  Median: {median_kappa:.4f}")
    print(
        f"  95% simulation interval: "
        f"[{lower:.4f}, {upper:.4f}]"
    )

print(f"\nSaved figures to:\n{utils.figure_dir}")
print(f"\nSaved CSV files to:\n{utils.csv_dir}")