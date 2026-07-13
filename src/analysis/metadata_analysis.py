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

utils = Utils("question4/")

# ============================================================
# 1. LOAD DATASETS & PARSE PIPELINES
# ============================================================
human_df = utils.load_human_data()
human_row = human_df[["article_id", "index", "human_label", "source", "politics", "age", "gender"]].copy()

all_rows = []
for prompt_name, filename in Constants.PROMPT_FILES.items():
    df = utils.load_model_outputs(Constants.REASONING_CONDITION_PATHS["Direct"], filename)
    if not df.empty:
        all_rows.append(df.assign(Prompt=prompt_name))

all_llm = pd.concat(all_rows, ignore_index=True).merge(human_row, on=["article_id", "index"], how="left")
all_llm.dropna(subset=["human_label"], inplace=True)
all_llm["human_label"] = all_llm["human_label"].astype(int)

utils.save_csv(all_llm, "q4_metadata_all_outputs_long_row_level.csv", index=False)

baseline = all_llm[all_llm["Prompt"] == "article_only"][[
    "Model", "article_id", "index", "llm_label", "llm_confidence", 
    "human_label", "source", "politics", "age", "gender"
]].copy().rename(columns={"llm_label": "article_only_output", "llm_confidence": "article_only_confidence"})

baseline["article_only_aligned"] = (baseline["article_only_output"] == baseline["human_label"]).astype(int)
utils.save_csv(baseline, "q4_metadata_article_only_baseline_row_level.csv", index=False)

# ============================================================
# 2. RUN EXPERIMENTAL CONDITION COMPARISONS Matrix
# ============================================================
compare_frames, flip_frames, alignment_frames = [], [], []
prediction_mcnemar_frames, alignment_mcnemar_frames, delta_rows = [], [], []

for metadata_condition in Constants.METADATA_ORDER:
    meta_label = Constants.METADATA_LABELS.get(metadata_condition, metadata_condition)
    prompt_df = all_llm[all_llm["Prompt"] == metadata_condition][["Model", "article_id", "index", "llm_label", "llm_confidence"]].copy()
    prompt_df.rename(columns={"llm_label": "metadata_output", "llm_confidence": "metadata_confidence"}, inplace=True)

    compare = baseline.merge(prompt_df, on=["Model", "article_id", "index"], how="inner")
    compare["Metadata_Condition"] = metadata_condition
    compare["Metadata_Label"] = meta_label
    compare["metadata_aligned"] = (compare["metadata_output"] == compare["human_label"]).astype(int)

    compare_frames.append(compare)
    flip_frames.append(utils.summarize_prediction_flips(compare, "article_only_output", "metadata_output", context_fields={"Metadata_Condition": metadata_condition, "Metadata_Label": meta_label}))
    alignment_frames.append(utils.summarize_alignment_change(compare, "article_only_aligned", "metadata_aligned", context_fields={"Metadata_Condition": metadata_condition, "Metadata_Label": meta_label}))
    
    prediction_mcnemar_frames.append(utils.run_mcnemar_by_model(compare, before_col="article_only_output", after_col="metadata_output", context_fields={"Metadata_Condition": metadata_condition, "Metadata_Label": meta_label}, test_type="Prediction change"))
    alignment_mcnemar_frames.append(utils.run_mcnemar_by_model(compare, before_col="article_only_aligned", after_col="metadata_aligned", context_fields={"Metadata_Condition": metadata_condition, "Metadata_Label": meta_label}, test_type="Alignment change"))

    for model_name, sub in compare.groupby("Model"):
        unique_article_sub = sub.drop_duplicates(subset=["article_id", "index"])
        
        base_acc = sub["article_only_aligned"].mean()
        meta_acc = sub["metadata_aligned"].mean()
        base_kappa = utils.safe_kappa_pair(unique_article_sub["human_label"], unique_article_sub["article_only_output"])
        meta_kappa = utils.safe_kappa_pair(unique_article_sub["human_label"], unique_article_sub["metadata_output"])
        base_bias = unique_article_sub["article_only_output"].mean()
        meta_bias = unique_article_sub["metadata_output"].mean()

        delta_rows.append({
            "Model": model_name, "Model_Display": utils.pretty_model_name(model_name),
            "Metadata_Condition": metadata_condition, "Metadata_Label": meta_label, "N": len(sub),
            "Baseline_Accuracy": base_acc, "Metadata_Accuracy": meta_acc, "Delta_Accuracy": meta_acc - base_acc,
            "Baseline_Kappa": base_kappa, "Metadata_Kappa": meta_kappa, "Delta_Kappa": meta_kappa - base_kappa,
            "Baseline_Bias_Rate": base_bias, "Metadata_Bias_Rate": meta_bias, "Delta_Bias_Rate": meta_bias - base_bias
        })

compare_df = pd.concat(compare_frames, ignore_index=True)
flip_df = pd.concat(flip_frames, ignore_index=True)
alignment_change_df = pd.concat(alignment_frames, ignore_index=True)
prediction_mcnemar_df = pd.concat(prediction_mcnemar_frames, ignore_index=True)
alignment_mcnemar_df = pd.concat(alignment_mcnemar_frames, ignore_index=True)
delta_df = pd.DataFrame(delta_rows)

utils.save_csv(compare_df, "q4_metadata_compare_to_article_only_row_level.csv", index=False)
utils.save_csv(flip_df, "q4_metadata_prediction_flip_summary_row_level.csv", index=False)
utils.save_csv(alignment_change_df, "q4_metadata_alignment_change_summary_row_level.csv", index=False)
utils.save_csv(prediction_mcnemar_df, "q4_metadata_mcnemar_prediction_change_row_level.csv", index=False)
utils.save_csv(alignment_mcnemar_df, "q4_metadata_mcnemar_alignment_change_row_level.csv", index=False)
utils.save_csv(pd.concat([prediction_mcnemar_df, alignment_mcnemar_df], ignore_index=True), "q4_metadata_all_mcnemar_tests_row_level.csv", index=False)
utils.save_csv(delta_df, "q4_metadata_delta_metrics_row_level.csv", index=False)

# ============================================================
# 3. GENERATE OVERALL EXPERIMENTAL METRIC REDUCTIONS
# ============================================================
overall_flip = flip_df.groupby(["Metadata_Condition", "Metadata_Label"]).agg(
    Mean_Flip_Rate=("Flip_Rate", "mean"), Mean_To_Biased_Rate=("To_Biased_Rate", "mean"), Mean_To_Not_Biased_Rate=("To_Not_Biased_Rate", "mean")
).reset_index()

overall_delta = delta_df.groupby(["Metadata_Condition", "Metadata_Label"]).agg(
    Mean_Delta_Accuracy=("Delta_Accuracy", "mean"), Mean_Delta_Kappa=("Delta_Kappa", "mean"), Mean_Delta_Bias_Rate=("Delta_Bias_Rate", "mean")
).reset_index()

for df_tgt in [overall_flip, overall_delta]:
    df_tgt["Metadata_Condition"] = pd.Categorical(df_tgt["Metadata_Condition"], categories=Constants.METADATA_ORDER, ordered=True)
    df_tgt.sort_values("Metadata_Condition", inplace=True)

utils.save_csv(overall_flip, "q4_metadata_overall_flip_summary_row_level.csv", index=False)
utils.save_csv(overall_delta, "q4_metadata_overall_delta_summary_row_level.csv", index=False)

# ============================================================
# 4. SCIENTIFIC PAPER CHARTS GENERATION PIPELINE
# ============================================================

# Figure 1: Average metadata flip rate chart
fig, ax = plt.subplots(figsize=(7.2, 4.5))
plot_flip = overall_flip.set_index("Metadata_Condition").reindex(Constants.METADATA_ORDER).reset_index()
bars = ax.bar(plot_flip["Metadata_Label"], plot_flip["Mean_Flip_Rate"], color=[Constants.COLORS[c] for c in plot_flip["Metadata_Condition"]], edgecolor="black", linewidth=0.7, width=0.65)

for bar, val in zip(bars, plot_flip["Mean_Flip_Rate"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=12)

ax.set_ylabel("Mean prediction flip rate")
ax.set_ylim(0,max(0.4, plot_flip["Mean_Flip_Rate"].max() + 0.08))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.tick_params(axis="y", labelsize=11)
ax.set_xticks(np.arange(len(plot_flip)))
ax.set_xticklabels(plot_flip["Metadata_Label"],rotation=25,ha="right",fontsize=12)
utils.save_figure(fig, "q4_metadata_average_flip_rate")

# Figure 2: Directional metadata flips stacked layout
fig, ax = plt.subplots(figsize=(7.2, 4.8))
x_axis = np.arange(len(plot_flip))
ax.bar(x_axis, plot_flip["Mean_To_Biased_Rate"], label="Not biased → Biased", color=Constants.COLORS["Not biased → Biased"], edgecolor="black", linewidth=0.7)
ax.bar(x_axis, plot_flip["Mean_To_Not_Biased_Rate"], bottom=plot_flip["Mean_To_Biased_Rate"], label="Biased → Not biased", color=Constants.COLORS["Biased → Not biased"], edgecolor="black", linewidth=0.7)

for idx, row in plot_flip.iterrows():
    tot = row["Mean_To_Biased_Rate"] + row["Mean_To_Not_Biased_Rate"]
    ax.text(idx, tot + 0.01, f"{tot * 100:.1f}%", ha="center", va="bottom", fontsize=12)

ax.set_xticks(x_axis)
ax.set_xticklabels(plot_flip["Metadata_Label"], rotation=25, ha="right")
ax.set_ylabel("Directional flip rates")
ax.legend(frameon=True)
ax.set_ylim(0, max(0.3, plot_flip["Mean_Flip_Rate"].max() + 0.08))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()
utils.save_figure(fig, "q4_metadata_directional_flips")

# Figure 3: Delta kappa barh layout matrix (Fixed tracking layout mapping keys)
delta_kappa_plot = delta_df.pivot(index="Model", columns="Metadata_Label", values="Delta_Kappa")
ordered_labels = [Constants.METADATA_LABELS[c] for c in Constants.METADATA_ORDER]
delta_kappa_plot = delta_kappa_plot.reindex(columns=ordered_labels)
delta_kappa_plot = delta_kappa_plot.loc[delta_kappa_plot.mean(axis=1).sort_values(ascending=True).index]

sig_lookup = alignment_mcnemar_df.set_index(["Model", "Metadata_Label"])["significance"].to_dict()
ordered_model_keys = list(delta_kappa_plot.index)

fig, ax = plt.subplots(figsize=(9.5, 8.5))
delta_kappa_plot.index = [utils.pretty_model_name(m) for m in ordered_model_keys]
delta_kappa_plot.plot(kind="barh", ax=ax, color=[Constants.COLORS[c] for c in Constants.METADATA_ORDER], edgecolor="black", linewidth=0.9, width=0.9)
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel(r"$\Delta$ Cohen's kappa")
ax.set_ylabel("")
ax.legend(title="", loc="lower right", fontsize=12)

for container, metadata_label in zip(ax.containers, ordered_labels):
    for bar, raw_model_id in zip(container, ordered_model_keys):
        star = sig_lookup.get((raw_model_id, metadata_label), "")
        if star in ["ns", "NA", ""]:
            continue
        width = bar.get_width()
        x_pos = width + (0.001 if width >= 0 else -0.001)
        ha = "left" if width >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2 - 0.06, star, va="center", ha=ha, fontsize=12, fontweight="bold")

ax.set_xlim(delta_kappa_plot.min().min() - 0.02, delta_kappa_plot.max().max() + 0.02)
plt.tight_layout()
utils.save_figure(fig, "q4_metadata_delta_kappa_by_model")

# Figure 4: Delta bias rate plot
fig, ax = plt.subplots(figsize=(7.2, 4.5))
plot_delta = overall_delta.set_index("Metadata_Condition").reindex(Constants.METADATA_ORDER).reset_index()
bars = ax.bar(plot_delta["Metadata_Label"], plot_delta["Mean_Delta_Bias_Rate"], color=[Constants.COLORS[c] for c in plot_delta["Metadata_Condition"]], edgecolor="black", linewidth=0.7, width=0.65)
ax.axhline(0, color="black", linewidth=1, linestyle="--")

for bar, val in zip(bars, plot_delta["Mean_Delta_Bias_Rate"]):
    y_pos = val + 0.01 if val >= 0 else val - 0.03
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val * 100:+.1f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=12)

ax.set_ylabel(r"$\Delta$ bias prediction rate")
ax.set_title("Change in Bias Prediction Rate Across Metadata Conditions")
ax.set_xticks(range(len(plot_delta["Metadata_Label"])))
ax.set_xticklabels(plot_delta["Metadata_Label"], rotation=25, ha="right")
ax.set_ylim(min(-0.15, plot_delta["Mean_Delta_Bias_Rate"].min() - 0.05), max(0.15, plot_delta["Mean_Delta_Bias_Rate"].max() + 0.05))
plt.tight_layout()
utils.save_figure(fig, "q4_metadata_delta_bias_prediction_rate")

# Figure 5: Flip Rate Confusion Matrix Heatmap layout
heatmap_flip = flip_df.pivot(index="Model_Display", columns="Metadata_Label", values="Flip_Rate").reindex(columns=ordered_labels)
heatmap_flip = heatmap_flip.loc[heatmap_flip.mean(axis=1).sort_values(ascending=True).index]

fig, ax = plt.subplots(figsize=(7.5, 5.6))
im = ax.imshow(heatmap_flip.values, aspect="auto", cmap="coolwarm", vmin=0, vmax=max(0.35, np.nanmax(heatmap_flip.values)))

ax.set_xticks(np.arange(len(heatmap_flip.columns)))
ax.set_xticklabels(heatmap_flip.columns, rotation=30, ha="right")
ax.set_yticks(np.arange(len(heatmap_flip.index)))
ax.set_yticklabels(heatmap_flip.index)

for i in range(heatmap_flip.shape[0]):
    for j in range(heatmap_flip.shape[1]):
        val = heatmap_flip.iloc[i, j]
        ax.text(j, i, "NA" if pd.isna(val) else f"{val * 100:.0f}%", ha="center", va="center", fontsize=9, color="white" if not pd.isna(val) and val > 0.25 else "black")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0)
plt.tight_layout()
utils.save_figure(fig, "q4_metadata_model_by_condition_flip_heatmap")

print(f"\nExecution Success! Summary files written to:\n{utils.csv_dir.parent}")