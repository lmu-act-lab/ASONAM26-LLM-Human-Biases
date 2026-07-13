from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.constants import Constants
from core.utils import Utils

matplotlib.use("Agg")
plt.rcParams.update(Constants.PLOT_STYLE)

utils = Utils("question3")
PROMPT_FILE = "prompt_article_info.csv"

# ============================================================
# 1. Load Datasets and Convert to Wide Formats
# ============================================================
long_df = utils.load_model_outputs(Constants.REASONING_CONDITION_PATHS["Direct"], PROMPT_FILE)
utils.save_csv(long_df, "inter_model_predictions_long.csv", index=False)

wide_complete = (
    long_df.pivot_table(index=["index", "article_id"], columns="Model", values="llm_label", aggfunc="first")
    .dropna(axis=0, how="any")
    .astype(int)
)

utils.save_csv(wide_complete, "inter_model_predictions_wide_complete.csv", index=False)
models = wide_complete.columns.tolist()

print(f"\nNumber of shared rows across all models: {len(wide_complete)}")
print(f"Number of models: {len(models)}")

# ============================================================
# 2. Vectorized Pairwise Inter-Model Agreement Computation
# ============================================================
kappa_matrix = pd.DataFrame(1.0, index=models, columns=models, dtype=float)
raw_agreement_matrix = pd.DataFrame(1.0, index=models, columns=models, dtype=float)
disagreement_matrix = pd.DataFrame(0.0, index=models, columns=models, dtype=float)

pairwise_rows = []

for model_a, model_b in combinations(models, 2):
    preds_a, preds_b = wide_complete[model_a], wide_complete[model_b]

    raw_agree = (preds_a == preds_b).mean()
    disagree = 1.0 - raw_agree
    kappa = utils.safe_kappa_pair(preds_a, preds_b)

    for m1, m2 in [(model_a, model_b), (model_b, model_a)]:
        raw_agreement_matrix.loc[m1, m2] = raw_agree
        disagreement_matrix.loc[m1, m2] = disagree
        kappa_matrix.loc[m1, m2] = kappa

    pairwise_rows.append({
        "Model_A": model_a, "Model_B": model_b,
        "Model_A_Display": utils.pretty_model_name(model_a), "Model_B_Display": utils.pretty_model_name(model_b),
        "Raw_Agreement": raw_agree, "Disagreement_Rate": disagree, "Cohen_Kappa": kappa, "N_Shared": len(wide_complete)
    })

pairwise_df = pd.DataFrame(pairwise_rows)

utils.save_csv(kappa_matrix, "inter_model_kappa_matrix.csv", index=True)
utils.save_csv(raw_agreement_matrix, "inter_model_raw_agreement_matrix.csv", index=True)
utils.save_csv(disagreement_matrix, "inter_model_disagreement_matrix.csv", index=True)
utils.save_csv(pairwise_df, "inter_model_pairwise_results.csv", index=False)

# ============================================================
# 3. Model-Level Descriptive Baseline Summaries
# ============================================================
model_summary_rows = []

for model in models:
    other_models = [m for m in models if m != model]
    
    model_summary_rows.append({
        "Model": model,
        "Model_Display": utils.pretty_model_name(model),
        "Mean_Kappa_With_Other_Models": kappa_matrix.loc[model, other_models].mean(),
        "Mean_Raw_Agreement_With_Other_Models": raw_agreement_matrix.loc[model, other_models].mean(),
        "Mean_Disagreement_With_Other_Models": disagreement_matrix.loc[model, other_models].mean(),
        "Bias_Prediction_Rate": wide_complete[model].mean()
    })

model_summary = pd.DataFrame(model_summary_rows).sort_values("Mean_Kappa_With_Other_Models", ascending=False)
utils.save_csv(model_summary, "inter_model_summary_by_model.csv", index=False)

# ============================================================
# 4. Export Statistical Plots & Heatmaps Manifest Files
# ============================================================
utils.plot_heatmap(fig_size=(8, 6), df=kappa_matrix, model_order=models, cbar_label="", filename="q3_inter_model_kappa_heatmap", vmin=-0.2, vmax=1.0, cmap="Blues", triangle="lower")
utils.plot_heatmap(fig_size=(8, 6), df=disagreement_matrix, model_order=models, cbar_label="Disagreement rate", filename="q3_inter_model_disagreement_heatmap", vmin=0, vmax=1, cmap="Blues", triangle="lower")

utils.plot_bar_chart(
    df=model_summary, x_col="Model_Display", y_col="Mean_Kappa_With_Other_Models", filename="mean_inter_model_kappa_by_model",
    ylabel="Mean Cohen's kappa with other models", color=Constants.COLORS["Default"], sort_by="Mean_Kappa_With_Other_Models",
    ascending=False, horizontal_line=0, value_format=".2f", text_offset=0.01
)

utils.plot_bar_chart(
    df=model_summary, x_col="Model_Display", y_col="Bias_Prediction_Rate", filename="bias_prediction_rate_by_model",
    ylabel="Proportion predicted biased", color=Constants.COLORS["Default"], sort_by="Bias_Prediction_Rate",
    ascending=False, ylim=(0, 1), value_format=".2f", text_offset=0.015
)

print(f"\nExecution Complete Summary Success!\nSaved CSV logs to: {utils.csv_dir}\nSaved figures to: {utils.figure_dir}")