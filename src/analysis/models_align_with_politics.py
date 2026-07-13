from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2
import statsmodels.formula.api as smf

from core.constants import Constants
from core.utils import Utils

# Setup plotting engine
matplotlib.use("Agg")
plt.rcParams.update(Constants.PLOT_STYLE)

# Initialization
PROMPT_FILE = "prompt_article_info.csv"
BAR_WIDTH = 0.65
HIST_COLOR = "0.65"

utils = Utils("question2")
output_dir = utils.csv_dir

# ============================================================
# 1. Load and Clean Datasets
# ============================================================
human_df = utils.load_human_data()
human_df = human_df[human_df["politics"].isin(Constants.POLITICS_GROUPS)].copy()

llm_df = utils.load_model_outputs(Constants.REASONING_CONDITION_PATHS["Direct"], PROMPT_FILE)

merged_df = human_df.merge(llm_df, on="index", how="inner", suffixes=("_human", "_llm"))
merged_df["article_id"] = merged_df["article_id_human"]
merged_df.dropna(subset=["human_label", "llm_label"], inplace=True)

merged_df["human_label"] = merged_df["human_label"].astype(int)
merged_df["llm_label"] = merged_df["llm_label"].astype(int)
merged_df["Aligned"] = (merged_df["human_label"] == merged_df["llm_label"]).astype(int)
merged_df["politics"] = pd.Categorical(merged_df["politics"], categories=Constants.POLITICS_GROUPS, ordered=True)
merged_df["Model_Display"] = merged_df["Model"].map(utils.pretty_model_name)

# ============================================================
# 2. Extract Stratified Aggregations and Kappa Metrics
# ============================================================
summary = merged_df.groupby(["Model", "Model_Display", "politics"], observed=True).agg(
    N=("Aligned", "count"),
    Accuracy=("Aligned", "mean"),
    Bias_Prediction_Rate=("llm_label", "mean"),
    Human_Bias_Rate=("human_label", "mean"),
).reset_index()

kappa_df = merged_df.groupby(["Model", "politics"], observed=True).apply(
    lambda x: utils.safe_kappa(x), include_groups=False
).reset_index(name="Kappa")

summary = summary.merge(kappa_df, on=["Model", "politics"], how="left")

# ============================================================
# 3. Logistic Regression Metrics Pipeline
# ============================================================
coef_rows, overall_rows = [], []
term_to_group = {
    "Intercept": ("Conservative", "Conservative baseline"),
    "C(politics, Treatment(reference='Conservative'))[T.Liberal]": ("Liberal", "Liberal vs Conservative"),
    "C(politics, Treatment(reference='Conservative'))[T.Independent]": ("Independent", "Independent vs Conservative")
}

for model_name, df_model in merged_df.groupby("Model"):
    df_model = df_model.copy()
    df_model["politics"] = pd.Categorical(df_model["politics"], categories=Constants.POLITICS_GROUPS, ordered=True)

    full_model = smf.logit("Aligned ~ C(politics, Treatment(reference='Conservative'))", data=df_model).fit(disp=0)
    null_model = smf.logit("Aligned ~ 1", data=df_model).fit(disp=0)

    lr_stat = 2 * (full_model.llf - null_model.llf)
    df_diff = full_model.df_model - null_model.df_model
    overall_p = chi2.sf(lr_stat, df_diff)

    overall_rows.append({
        "Model": model_name,
        "Model_Display": utils.pretty_model_name(model_name),
        "LR_statistic": lr_stat,
        "df": df_diff,
        "overall_p_value": overall_p,
        "overall_significance": utils.significance_star(overall_p),
        "interpretation": "Accuracy differs by political group" if overall_p < 0.05 else "No significant overall political-group difference"
    })

    Path(output_dir / f"logit_summary_{model_name}.txt").write_text(str(full_model.summary()))

    coef_table = full_model.summary2().tables[1]
    for term, row in coef_table.iterrows():
        group, comparison = term_to_group.get(term, (term, term))
        
        coef_rows.append({
            "Model": model_name,
            "Model_Display": utils.pretty_model_name(model_name),
            "Group": group,
            "Comparison": comparison,
            "Coefficient_log_odds": row["Coef."],
            "Odds_Ratio": np.exp(row["Coef."]),
            "OR_CI_2.5": np.exp(row["[0.025"]),
            "OR_CI_97.5": np.exp(row["0.975]"]),
            "p_value": row["P>|z|"],
            "significance": utils.significance_star(row["P>|z|"]),
            "direction": "higher accuracy than Conservative" if row["Coef."] > 0 else "lower accuracy than Conservative"
        })

logit_coef_df = pd.DataFrame(coef_rows)
logit_overall_df = pd.DataFrame(overall_rows)

# ============================================================
# 4. Accuracy Grouped Bar Chart Visual Construction
# ============================================================
plot_df = summary.copy()
sig_lookup = logit_coef_df.set_index(["Model", "Group"])["significance"].to_dict()

plot_df["Sig_vs_Conservative"] = plot_df.apply(
    lambda r: "" if r["politics"] == "Conservative" else sig_lookup.get((r["Model"], r["politics"]), ""), axis=1
)

model_order = summary.groupby("Model")["Accuracy"].mean().sort_values(ascending=False).index.tolist()
model_display_order = [utils.pretty_model_name(m) for m in model_order]

x = np.arange(len(model_order))
bar_width = 0.30
fig, ax = plt.subplots(figsize=(10, 4.8))

for i, group in enumerate(Constants.POLITICS_GROUPS):
    group_data = plot_df[plot_df["politics"] == group].set_index("Model").reindex(model_order).reset_index()
    offset = (i - 1) * bar_width

    bars = ax.bar(
        x + offset, group_data["Accuracy"], width=bar_width, label=group,
        color=Constants.COLORS[group], edgecolor="black", linewidth=0.4
    )

    for bar, (_, row) in zip(bars, group_data.iterrows()):
        value = row["Accuracy"]
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value*100:.1f}", ha="center", va="bottom", fontsize=10)

        if group != "Conservative" and row["Sig_vs_Conservative"] in ["*", "**", "***"]:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, row["Sig_vs_Conservative"], ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.legend(title="Political group", ncol=1, loc="upper right")
ax.text(0.5, 1, "* p < .05, ** p < .01, *** p < .001\nStars compare each group with Conservative baseline.", transform=ax.transAxes, fontsize=12, va="top", ha="center")

ax.set_xticks(x)
ax.set_xticklabels(model_display_order, rotation=35, ha="right")
utils.finalize_plot(ax, ylabel="Accuracy", rotate_x=35, ylim=(0, 0.75))
utils.save_figure(fig, "q2_accuracy_by_politics")

# ============================================================
# 5. Export Heatmaps & Manifest Metric Long Tables
# ============================================================
heatmap_configs = [
    ("Kappa", "q2_kappa_heatmap_by_politics", -0.10, 0.20, "Blues", ".2f"),
    ("Bias_Prediction_Rate", "q2_bias_prediction_rate_heatmap_by_politics", 0, 1, "Blues", "percent_no_symbol")
]

for metric, filename, vmin, vmax, cmap, fmt in heatmap_configs:
    pivot_df = summary.pivot(index="Model", columns="politics", values=metric)
    utils.plot_heatmap(
        fig_size=(3.2, 3.6), df=pivot_df, model_order=model_order, cbar_label="",
        filename=filename, vmin=vmin, vmax=vmax, cmap=cmap, fmt=fmt, column_order=Constants.POLITICS_GROUPS
    )
    utils.save_csv(pivot_df.reindex(index=model_order, columns=Constants.POLITICS_GROUPS), f"q2_{metric.lower()}_by_model_and_politics.csv")

# Export manifest execution tables
utils.save_csv(merged_df, "q2_logistic_long_data.csv", index=False)
utils.save_csv(summary, "q2_summary_accuracy_kappa_biasrate.csv", index=False)
utils.save_csv(logit_coef_df, "q2_logistic_coefficients_vs_conservative.csv", index=False)
utils.save_csv(logit_overall_df, "q2_logistic_overall_politics_effect.csv", index=False)

print(f"\nExecution Success!\nSaved CSV logs to: {utils.csv_dir}\nSaved figures to: {utils.figure_dir}")