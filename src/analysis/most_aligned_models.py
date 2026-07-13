from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import statsmodels.formula.api as smf

from core.constants import Constants
from core.utils import Utils

matplotlib.use("Agg")
plt.rcParams.update(Constants.PLOT_STYLE)

utils = Utils("question1/")
output_parent_dir = utils.csv_dir.parent


def _render_horizontal_barplot(metric_series, xlabel, filename, value_fmt="{:.2f}", offset=0.006, zero_line=False, xlim=None):
    """Unified engine to build organized paper-ready horizontal chart plots."""
    df = metric_series.reset_index()
    df.columns = ["Model", "Metric"]
    df["Model_Label"] = df["Model"].map(Constants.MODEL_NAME_MAP).fillna(df["Model"])
    df.sort_values("Metric", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    bars = ax.barh(
        df["Model_Label"], df["Metric"],
        color=Constants.COLORS["Default"], edgecolor=Constants.COLORS["Edge"], linewidth=0.7
    )

    if zero_line:
        ax.axvline(0, linestyle="--", linewidth=0.8, color="black")

    for bar in bars:
        width = bar.get_width()
        x_pos = width + (offset if width >= 0 else -offset)
        ha = "left" if width >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, value_fmt.format(width), va="center", ha=ha, fontsize=8)

    ax.set_xlabel(xlabel)
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", length=3, width=0.8)

    if xlim:
        ax.set_xlim(xlim)
    else:
        xmin = min(df["Metric"].min() - 0.03, -0.05) if df["Metric"].min() < 0 else 0
        ax.set_xlim(xmin, df["Metric"].max() + 0.06)

    plt.tight_layout(pad=0.4)
    utils.save_figure(fig, filename)


# =========================================================
# 1. Load Datasets & Add Baseline Models
# =========================================================
human_df = utils.load_human_data()
llm_df = utils.load_model_outputs(Constants.REASONING_CONDITION_PATHS["Direct"], "prompt_article_info.csv")

# Create distinct reference baseline indices maps
index_map = human_df[["index", "article_id"]].drop_duplicates(subset=["index"]).copy()
majority_label = int(human_df["human_label"].mode().iloc[0])
human_biased_rate = float(human_df["human_label"].mean())

rng = np.random.default_rng(seed=42)
random_labels = rng.binomial(1, human_biased_rate, size=len(index_map))
label_to_text = {1: "is-biased", 0: "is-not-biased"}

majority_baseline = index_map.assign(
    llm_label=majority_label,
    llm_assessment=label_to_text[majority_label],
    llm_confidence=np.nan,
    Model="majority_class_baseline"
)

random_baseline = index_map.assign(
    llm_label=random_labels,
    llm_assessment=lambda d: d["llm_label"].map(label_to_text),
    llm_confidence=np.nan,
    Model="random_baseline"
)

llm_df = pd.concat([llm_df, majority_baseline, random_baseline], ignore_index=True)

# =========================================================
# 2. Alignment Diagnostics Data Processing
# =========================================================
row_level_df = human_df.merge(llm_df, on="index", how="inner", suffixes=("_human", "_llm"))
row_level_df["article_id"] = row_level_df["article_id_human"]
row_level_df["Aligned"] = (row_level_df["llm_label"] == row_level_df["human_label"]).astype(int)

# Inject consensus baseline features metrics
human_article_stats = human_df.groupby("article_id")["human_label"].agg(
    human_bias_rate="mean", num_raters="count"
).reset_index()
row_level_df = row_level_df.merge(human_article_stats, on="article_id", how="left")

# =========================================================
# 3. Logistic Regression Statistical Modeling
# =========================================================
print("\n" + "=" * 60 + "\nFitting Logit Models...")
model_simple = smf.logit("Aligned ~ C(Model)", data=row_level_df).fit()
model_with_human_bias_rate = smf.logit("Aligned ~ C(Model) + human_bias_rate", data=row_level_df).fit()

# Write summaries text directly via Path
(output_parent_dir / "q1_row_level_model_only_summary.txt").write_text(str(model_simple.summary()))
(output_parent_dir / "q1_row_level_model_human_bias_summary.txt").write_text(str(model_with_human_bias_rate.summary()))

# =========================================================
# 4. Compute Metrics Evaluations & Export Summary Plots
# =========================================================
# Raw accuracy levels
ranking = row_level_df.groupby("Model")["Aligned"].mean().sort_values(ascending=False)

weighted_acc = row_level_df.groupby("Model").apply(
    lambda x: np.average(x["Aligned"], weights=x["num_raters"]), include_groups=False
).sort_values(ascending=False)

kappa_scores = row_level_df.groupby("Model").apply(
    lambda x: cohen_kappa_score(x["llm_label"], x["human_label"]), include_groups=False
).sort_values(ascending=False)

model_bias_rate = row_level_df.groupby("Model")["llm_label"].mean().sort_values(ascending=False)

utils.save_csv(row_level_df, "q1_row_level_merged.csv", index=False)
utils.save_csv(ranking, "q1_row_level_ranking.csv")
utils.save_csv(weighted_acc, "q1_row_level_weighted_ranking.csv")
utils.save_csv(kappa_scores, "q1_row_level_kappa.csv")
utils.save_csv(model_bias_rate, "q1_row_level_bias_prediction_rate.csv", header=["bias_prediction_rate"])

_render_horizontal_barplot(ranking, "Accuracy", "q1_accuracy_paper")
_render_horizontal_barplot(weighted_acc, "Weighted accuracy", "q1_weighted_accuracy_paper")
_render_horizontal_barplot(kappa_scores, "Cohen's kappa", "q1_kappa_paper", zero_line=True)
_render_horizontal_barplot(model_bias_rate, "Proportion predicted as biased", "q1_bias_prediction_rate_paper", xlim=(0, 1))

print(f"\nExecution success!\nSaved logs to: {output_parent_dir}\nSaved figures to: {utils.figure_dir}")