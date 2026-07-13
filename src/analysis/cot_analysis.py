from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from core.constants import Constants
from core.utils import Utils

matplotlib.use("Agg")
plt.rcParams.update(Constants.PLOT_STYLE)

utils = Utils("question5")
PROMPT_FILE = "prompt_pii_combined_variants.csv"

# Label mapping constants configuration
COND_COLS_MAP = {
    "Direct": ("direct_output", "direct_aligned", "direct"),
    "CoT": ("cot_output", "cot_aligned", "cot"),
    "Chained CoT": ("chained_cot_output", "chained_cot_aligned", "chained_cot")
}


def _plot_horizontal_bars(
    df,
    columns,
    value_col,
    filename,
    xlabel,
    xlim=None,
    zero_line=False,
    pct_formatter=False,
    sort_col=None,
):
    """
    Generate a horizontal grouped bar plot.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    columns : list[str]
        Ordered conditions or comparisons to display.
    value_col : str
        Exact numeric metric to plot, such as:
        - Accuracy
        - Kappa
        - Bias_Prediction_Rate
        - Delta_Accuracy
    """
    pivot_col = "Condition" if "Condition" in df.columns else "Comparison"

    required_cols = {"Model", pivot_col, value_col}
    missing_cols = required_cols.difference(df.columns)

    if missing_cols:
        raise ValueError(
            f"Cannot plot {value_col}. Missing columns: "
            f"{sorted(missing_cols)}"
        )

    plot_df = (
        df.pivot(
            index="Model",
            columns=pivot_col,
            values=value_col,
        )
        .reindex(columns=columns)
    )

    if sort_col is not None:
        if sort_col not in plot_df.columns:
            raise ValueError(
                f"sort_col={sort_col!r} is not present in plot columns."
            )

        ordered_index = (
            plot_df[sort_col]
            .sort_values(ascending=True)
            .index
        )
    else:
        ordered_index = (
            plot_df.mean(axis=1)
            .sort_values(ascending=True)
            .index
        )

    plot_df = plot_df.loc[ordered_index]

    # Convert raw model IDs to display names only after sorting.
    plot_df.index = plot_df.index.map(utils.pretty_model_name)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    plot_df.plot(
        kind="barh",
        ax=ax,
        color=[Constants.COLORS[c] for c in columns],
        edgecolor="black",
        linewidth=0.6,
        width=0.8,
    )

    if zero_line:
        ax.axvline(
            0,
            color="black",
            linewidth=1,
            linestyle="--",
        )

    if pct_formatter:
        ax.xaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.legend(
        title="",
        loc=Constants.LEGEND_LOCATIONS["lower_right"],
    )

    plt.tight_layout()
    utils.save_figure(fig, filename)


# ============================================================
# 1. LOAD & CONCATENATE DATASETS
# ============================================================
human_row = utils.load_human_data()

all_conditions = []
for cond_name, path in Constants.REASONING_CONDITION_PATHS.items():
    cond_df = utils.load_model_outputs(path, PROMPT_FILE)
    if not cond_df.empty:
        all_conditions.append(cond_df.assign(Condition=cond_name))

all_llm = pd.concat(all_conditions, ignore_index=True)
utils.save_csv(all_llm, "q5_reasoning_all_outputs_long.csv", index=False)

# Keep only rows matched across all conditions
req_conds = len(Constants.REASONING_CONDITION_PATHS)
match_counts = all_llm.groupby(["Model", "article_id", "index"])["Condition"].transform("nunique")
all_llm_matched = all_llm[match_counts == req_conds].copy()

# ============================================================
# 2. BUILD ROW-LEVEL WIDE TABLE
# ============================================================
wide_parts = []
for cond in Constants.REASONING_CONDITION_ORDER:
    safe_name = cond.lower().replace(" ", "_")
    part = all_llm_matched[all_llm_matched["Condition"] == cond][["Model", "article_id", "index", "llm_label", "llm_confidence"]].copy()
    part.rename(columns={"llm_label": f"{safe_name}_output", "llm_confidence": f"{safe_name}_confidence"}, inplace=True)
    wide_parts.append(part.set_index(["Model", "article_id", "index"]))

wide_df = pd.concat(wide_parts, axis=1, join="inner").reset_index()
wide_df = wide_df.merge(human_row, on=["article_id", "index"], how="left").dropna(subset=["human_label"])
wide_df["human_label"] = wide_df["human_label"].astype(int)

for cond, (out_col, align_col, _) in COND_COLS_MAP.items():
    wide_df[align_col] = (wide_df[out_col] == wide_df["human_label"]).astype(int)

utils.save_csv(wide_df, "q5_reasoning_row_level_wide.csv", index=False)

# ============================================================
# 3. PERFORMANCE & DELTA METRICS COMPUTATION
# ============================================================
perf_rows, delta_rows = [], []

for model_name, sub in wide_df.groupby("Model"):
    model_disp = utils.pretty_model_name(model_name)
    metrics_cache = {}

    unique_article_sub = sub.drop_duplicates(subset=["article_id", "index"])

    for cond in Constants.REASONING_CONDITION_ORDER:
        out_col, align_col, safe_lbl = COND_COLS_MAP[cond]
        
        acc = sub[align_col].mean()
        kappa = utils.safe_kappa_pair(unique_article_sub["human_label"], unique_article_sub[out_col])
        bias = unique_article_sub[out_col].mean()
        conf = unique_article_sub[f"{safe_lbl}_confidence"].mean()

        metrics_cache[cond] = {"acc": acc, "kappa": kappa, "bias": bias}
        
        perf_rows.append({
            "Model": model_name, "Model_Display": model_disp, "Condition": cond, "N": len(sub),
            "Accuracy": acc, "Kappa": kappa, "Bias_Prediction_Rate": bias, "Mean_Confidence": conf
        })

    comparisons = [
        (Constants.REASONING_COMPARISONS["cot_vs_direct"], "CoT", "Direct"),
        (Constants.REASONING_COMPARISONS["chained_vs_direct"], "Chained CoT", "Direct"),
        (Constants.REASONING_COMPARISONS["chained_vs_cot"], "Chained CoT", "CoT")
    ]
    for comp_name, aft, bef in comparisons:
        delta_rows.append({
            "Model": model_name, "Model_Display": model_disp, "Comparison": comp_name,
            "Before_Accuracy": metrics_cache[bef]["acc"], "After_Accuracy": metrics_cache[aft]["acc"], "Delta_Accuracy": metrics_cache[aft]["acc"] - metrics_cache[bef]["acc"],
            "Before_Kappa": metrics_cache[bef]["kappa"], "After_Kappa": metrics_cache[aft]["kappa"], "Delta_Kappa": metrics_cache[aft]["kappa"] - metrics_cache[bef]["kappa"],
            "Before_Bias_Rate": metrics_cache[bef]["bias"], "After_Bias_Rate": metrics_cache[aft]["bias"], "Delta_Bias_Rate": metrics_cache[aft]["bias"] - metrics_cache[bef]["bias"]
        })

performance_df, delta_df = pd.DataFrame(perf_rows), pd.DataFrame(delta_rows)

# ============================================================
# 4. HYPOTHESIS FLIP ANALYSIS & MCNEMAR PIPELINE RUNS
# ============================================================
comp_configs = [
    (Constants.REASONING_COMPARISONS["cot_vs_direct"], "direct_output", "cot_output", "direct_aligned", "cot_aligned"),
    (Constants.REASONING_COMPARISONS["chained_vs_direct"], "direct_output", "chained_cot_output", "direct_aligned", "chained_cot_aligned"),
    (Constants.REASONING_COMPARISONS["chained_vs_cot"], "cot_output", "chained_cot_output", "cot_aligned", "chained_cot_aligned")
]

flips, pred_mc, align_chg, align_mc = [], [], [], []
for comp, b_out, a_out, b_alg, a_alg in comp_configs:
    flips.append(utils.summarize_prediction_flips(wide_df, b_out, a_out, context_fields={"Comparison": comp}))
    pred_mc.append(utils.run_mcnemar_by_model(wide_df, b_out, a_out, context_fields={"Comparison": comp}, test_type="Prediction change"))
    align_chg.append(utils.summarize_alignment_change(wide_df, b_alg, a_alg, context_fields={"Comparison": comp}))
    align_mc.append(utils.run_mcnemar_by_model(wide_df, b_alg, a_alg, context_fields={"Comparison": comp}, test_type="Alignment change"))

flip_df = pd.concat(flips, ignore_index=True)
alignment_mcnemar_df = pd.concat(align_mc, ignore_index=True)
all_mcnemar_df = pd.concat([pd.concat(pred_mc, ignore_index=True), alignment_mcnemar_df], ignore_index=True)

utils.save_csv(performance_df, "q5_reasoning_performance_by_condition_row_level.csv", index=False)
utils.save_csv(delta_df, "q5_reasoning_delta_metrics_row_level.csv", index=False)
utils.save_csv(flip_df, "q5_reasoning_prediction_flip_summary_row_level.csv", index=False)
utils.save_csv(pd.concat(align_chg, ignore_index=True), "q5_reasoning_alignment_change_summary_row_level.csv", index=False)
utils.save_csv(all_mcnemar_df, "q5_reasoning_all_mcnemar_tests_row_level.csv", index=False)

utils.save_csv(performance_df.groupby("Condition")[["Accuracy", "Kappa", "Bias_Prediction_Rate", "Mean_Confidence"]].mean().reindex(Constants.REASONING_CONDITION_ORDER), "q5_reasoning_overall_performance_row_level.csv")
utils.save_csv(flip_df.groupby("Comparison")[["Flip_Rate", "To_Biased_Rate", "To_Not_Biased_Rate"]].mean().reindex(Constants.REASONING_COMPARISON_ORDER), "q5_reasoning_overall_flip_summary_row_level.csv")
utils.save_csv(delta_df.groupby("Comparison")[["Delta_Accuracy", "Delta_Kappa", "Delta_Bias_Rate"]].mean().reindex(Constants.REASONING_COMPARISON_ORDER), "q5_reasoning_overall_delta_metrics_row_level.csv")

# ============================================================
# 5. PAPER PLOTS RENDERERS
# ============================================================

# Figure 1: Average flip rate chart (Fixed Tick Labels Mismatch warning)
avg_flip = flip_df.groupby("Comparison")["Flip_Rate"].mean().reindex(Constants.REASONING_COMPARISON_ORDER)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(avg_flip.index, avg_flip.values, color=[Constants.COLORS[x] for x in avg_flip.index], edgecolor="black", linewidth=0.7, width=0.65)
for bar, val in zip(bars, avg_flip.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val * 100:.1f}%", ha="center", va="bottom")
ax.set_ylabel("Prediction Flip Rate")
ax.set_ylim(0, max(0.5, avg_flip.max() + 0.08))
ax.set_xticks(range(len(avg_flip.index)))
ax.set_xticklabels(avg_flip.index, rotation=20, ha="right")
plt.tight_layout()
utils.save_figure(fig, "q5_avg_prediction_flip_rate_row_level")

# Figure 2: Directional flips stacked chart
dir_overall = flip_df.groupby("Comparison")[["To_Biased_Rate", "To_Not_Biased_Rate"]].mean().reindex(Constants.REASONING_COMPARISON_ORDER)
fig, ax = plt.subplots(figsize=(6.5, 4.4))
x_axis = np.arange(len(dir_overall.index))
ax.bar(x_axis, dir_overall["To_Biased_Rate"], label="Not biased → Biased", color=Constants.COLORS["Not biased → Biased"], edgecolor="black", linewidth=0.7)
ax.bar(x_axis, dir_overall["To_Not_Biased_Rate"], bottom=dir_overall["To_Biased_Rate"], label="Biased → Not biased", color=Constants.COLORS["Biased → Not biased"], edgecolor="black", linewidth=0.7)
for idx, row in enumerate(dir_overall.itertuples()):
    tot = row.To_Biased_Rate + row.To_Not_Biased_Rate
    ax.text(idx, tot + 0.01, f"{tot * 100:.1f}%", ha="center", va="bottom")
ax.set_xticks(x_axis)
ax.set_xticklabels(dir_overall.index, rotation=20, ha="right")
ax.set_ylabel("Directional Flip Rates")
ax.legend(frameon=True)
ax.set_ylim(0, max(0.5, dir_overall.sum(axis=1).max() + 0.08))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()
utils.save_figure(fig, "q5_directional_prediction_flips_row_level")

# Figure 3: Accuracy by condition
_plot_horizontal_bars(
    df=performance_df,
    columns=Constants.REASONING_CONDITION_ORDER,
    value_col="Accuracy",
    filename="q5_accuracy_by_reasoning_condition_row_level",
    xlabel="Accuracy",
    xlim=(0, max(0.75, performance_df["Accuracy"].max() + 0.05)),
)

# Figure 4: Cohen's kappa by condition
_plot_horizontal_bars(
    df=performance_df,
    columns=Constants.REASONING_CONDITION_ORDER,
    value_col="Kappa",
    filename="q5_kappa_by_reasoning_condition_row_level",
    xlabel="Cohen's kappa",
    zero_line=True,
)

# Figure 6: Delta accuracy
_plot_horizontal_bars(
    df=delta_df,
    columns=Constants.REASONING_COMPARISON_ORDER,
    value_col="Delta_Accuracy",
    filename="q5_delta_accuracy_by_reasoning_condition_row_level",
    xlabel="Δ accuracy",
    zero_line=True,
    sort_col=Constants.REASONING_COMPARISONS["chained_vs_direct"],
)

# Figure 7: Bias prediction rate
_plot_horizontal_bars(
    df=performance_df,
    columns=Constants.REASONING_CONDITION_ORDER,
    value_col="Bias_Prediction_Rate",
    filename="q5_bias_prediction_rate_by_reasoning_condition_row_level",
    xlabel="Biased Prediction Rate",
    xlim=(0, 1),
    pct_formatter=True,
)

# Figure 5: Delta kappa chart supplemented with McNemar statistical flags (Fixed Axis Value Matching Bug)
delta_kappa_plot = delta_df.pivot(index="Model", columns="Comparison", values="Delta_Kappa").reindex(columns=Constants.REASONING_COMPARISON_ORDER)
delta_kappa_plot = delta_kappa_plot.loc[delta_kappa_plot["Chained CoT vs Direct"].sort_values(ascending=True).index]
sig_lookup = alignment_mcnemar_df.set_index(["Model", "Comparison"])["significance"].to_dict()

ordered_model_keys = list(delta_kappa_plot.index)
pretty_model_labels = [utils.pretty_model_name(m) for m in ordered_model_keys]

fig, ax = plt.subplots(figsize=(8.0, 5.8))
delta_kappa_plot.index = pretty_model_labels
delta_kappa_plot.plot(kind="barh", ax=ax, color=[Constants.COLORS[c] for c in delta_kappa_plot.columns], edgecolor="black", linewidth=0.6, width=0.82)
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel(r"$\Delta$ Cohen's kappa")
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=14)
ax.legend(title="", loc=Constants.LEGEND_LOCATIONS["lower_right"])

for container, comparison in zip(ax.containers, delta_kappa_plot.columns):
    for bar, raw_model_id in zip(container, ordered_model_keys):
        star = sig_lookup.get((raw_model_id, comparison), "")
        if star in ["ns", "NA", ""]:
            continue
        width = bar.get_width()
        x_pos = width + (0.008 if width >= 0 else -0.008)
        ha = "left" if width >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, star, va="center", ha=ha, fontweight="bold")

ax.set_xlim(delta_kappa_plot.min().min() - 0.05, delta_kappa_plot.max().max() + 0.07)
plt.tight_layout()
utils.save_figure(fig, "q5_delta_kappa_by_reasoning_condition_row_level")

print(f"\nExecution success Summary Output Complete!\nCSV records written to: {utils.csv_dir}\nCharts saved to: {utils.figure_dir}")