import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import cohen_kappa_score

from .constants import Constants


class Utils:
    
    def __init__(self, foldername):
        self.setup(foldername)
        
    def setup(self, foldername):
        self.figure_dir = Constants.ANALYSIS_FOLDER_DIR / foldername / Constants.FIGURE_FOLDER
        self.csv_dir = Constants.ANALYSIS_FOLDER_DIR / foldername / Constants.CSV_FOLDER
        
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
    def save_figure(self, fig, filename):
        fig.savefig(
            self.figure_dir / f"{filename}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        
    def save_csv(self, df, filename, **kwargs):
        df.to_csv(self.csv_dir / filename, **kwargs)
        
    def load_human_data(self):
        df = pd.read_csv(Constants.DATA_FILE)
        df["bias-question"] = df["bias-question"].astype(str).str.strip().str.lower()
        df["human_label"] = df["bias-question"].map(Constants.LABEL_MAP)
        
        df = df.dropna(subset=["human_label"]).copy()
        df["human_label"] = df["human_label"].astype(int)
        return df
    
    def load_model_outputs(self, base_path, prompt_file):
        all_llm = []
        
        for model_dir in sorted(base_path.iterdir()):
            if not model_dir.is_dir():
                continue
                
            file_path = model_dir / "llm_outputs" / prompt_file
            if not file_path.exists():
                print(f"Warning: {file_path} does not exist. Skipping.")
                continue
        
            df = pd.read_csv(
                file_path,
                usecols=["index", "article_id", "llm_assessment", "llm_confidence"]
            ).copy()

            df["llm_assessment"] = df["llm_assessment"].astype(str).str.strip().str.lower()
            df["llm_label"] = df["llm_assessment"].map(Constants.LABEL_MAP)
            df["Model"] = model_dir.name

            all_llm.append(df)

        return pd.concat(all_llm, ignore_index=True) if all_llm else pd.DataFrame()
    
    def pretty_model_name(self, model):
        return Constants.MODEL_NAME_MAP.get(model, model)
    
    def significance_star(self, p):
        if pd.isna(p):
            return "NA"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"
    
    def normalize_label(self, x):
        """Standardize label values"""
        return np.nan if pd.isna(x) else str(x).strip().lower()

    def normalize_source_name(self, x):
        """Normalize common source names (e.g., fox, cnn, bbc)."""
        if pd.isna(x):
            return np.nan
        low = str(x).strip().lower()

        for network in ["fox", "cnn", "bbc"]:
            if network in low:
                return network.upper()
        return str(x).strip()
    
    def safe_kappa(self, x):
        if x["human_label"].nunique() < 2 or x["llm_label"].nunique() < 2:
            return np.nan
        return cohen_kappa_score(x["human_label"], x["llm_label"])

    def safe_kappa_pair(self, a, b):
        """Compute Cohen's kappa for two 1-D sequences with safety checks."""
        a = pd.Series(a).dropna()
        b = pd.Series(b).dropna()

        common_idx = a.index.intersection(b.index)
        a, b = a.loc[common_idx], b.loc[common_idx]

        if len(a) == 0 or a.nunique() < 2 or b.nunique() < 2:
            return np.nan

        return cohen_kappa_score(a, b)

    def run_mcnemar_by_model(self, df, before_col, after_col, *, context_fields=None, test_type=None):
        rows = []

        for model_name, sub in df.groupby("Model"):
            table = pd.crosstab(sub[before_col], sub[after_col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            n01, n10 = int(table.loc[0, 1]), int(table.loc[1, 0])

            if n01 + n10 == 0:
                stat, p_value = np.nan, np.nan
            else:
                from statsmodels.stats.contingency_tables import mcnemar
                result = mcnemar(table.values, exact=False, correction=True)
                stat, p_value = float(result.statistic), float(result.pvalue)

            row = {
                "Model": model_name,
                "Model_Display": self.pretty_model_name(model_name),
                "N": len(sub),
                "n00": int(table.loc[0, 0]),
                "n01": n01,
                "n10": n10,
                "n11": int(table.loc[1, 1]),
                "McNemar_statistic": stat,
                "p_value": p_value,
                "significance": self.significance_star(p_value),
            }

            if test_type is not None:
                row["Test_Type"] = test_type
            if context_fields:
                row.update(context_fields)

            rows.append(row)

        return pd.DataFrame(rows)

    def summarize_prediction_flips(self, df, before_col, after_col, *, context_fields=None):
        rows = []

        for model_name, sub in df.groupby("Model"):
            b_vals, a_vals = sub[before_col], sub[after_col]
            flipped = b_vals != a_vals
            to_biased = (b_vals == 0) & (a_vals == 1)
            to_not_biased = (b_vals == 1) & (a_vals == 0)

            n_total = len(sub)
            n_flipped = int(flipped.sum())

            row = {
                "Model": model_name,
                "Model_Display": self.pretty_model_name(model_name),
                "N": n_total,
                "N_Flipped": n_flipped,
                "N_To_Biased": int(to_biased.sum()),
                "N_To_Not_Biased": int(to_not_biased.sum()),
                "Flip_Rate": n_flipped / n_total if n_total else np.nan,
                "To_Biased_Rate": to_biased.sum() / n_total if n_total else np.nan,
                "To_Not_Biased_Rate": to_not_biased.sum() / n_total if n_total else np.nan,
                "To_Biased_Pct_Of_Flips": to_biased.sum() / n_flipped if n_flipped > 0 else np.nan,
                "To_Not_Biased_Pct_Of_Flips": to_not_biased.sum() / n_flipped if n_flipped > 0 else np.nan,
            }

            if context_fields:
                row.update(context_fields)

            rows.append(row)

        return pd.DataFrame(rows)

    def summarize_alignment_change(self, df, before_col, after_col, *, context_fields=None):
        rows = []

        for model_name, sub in df.groupby("Model"):
            before, after = sub[before_col], sub[after_col]
            improved = (before == 0) & (after == 1)
            worsened = (before == 1) & (after == 0)
            unchanged = before == after

            row = {
                "Model": model_name,
                "Model_Display": self.pretty_model_name(model_name),
                "N": len(sub),
                "Before_Accuracy": before.mean(),
                "After_Accuracy": after.mean(),
                "Delta_Accuracy": after.mean() - before.mean(),
                "N_Improved": int(improved.sum()),
                "N_Worsened": int(worsened.sum()),
                "N_Unchanged": int(unchanged.sum()),
                "Improved_Rate": improved.mean(),
                "Worsened_Rate": worsened.mean(),
                "Net_Improvement_Rate": improved.mean() - worsened.mean(),
            }

            if context_fields:
                row.update(context_fields)

            rows.append(row)

        return pd.DataFrame(rows)

    def finalize_plot(self, ax, *, xlabel="", ylabel="", rotate_x=0, xlim=None, ylim=None, y_percent=False):
        ax.set_title("")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=3, width=0.6)

        if rotate_x:
            plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right")
            
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if y_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        plt.tight_layout(pad=0.3)

    def add_bar_percent_labels(self, ax, bars, counts):
        total = counts.sum()
        for bar, count in zip(bars, counts):
            percentage = 100 * count / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{percentage:.1f}%",
                ha="center", va="bottom", fontsize=12,
            )
            
    def plot_heatmap(self, fig_size, df, model_order, cbar_label, filename, fmt=".2f", vmin=None, vmax=None, cmap="coolwarm_r", column_order=None, triangle=None):
        plot_df = df.copy().reindex(index=model_order)
        if column_order is not None:
            plot_df = plot_df.reindex(columns=column_order)

        pretty_labels = [self.pretty_model_name(x) for x in plot_df.index]
        plot_df.index = pretty_labels
        if column_order is None:
            plot_df.columns = pretty_labels

        n_rows, n_cols = plot_df.shape
        if triangle is not None and n_rows != n_cols:
            raise ValueError("Triangle heatmaps require square matrices.")

        fig, ax = plt.subplots(figsize=fig_size)
        mask = None

        if triangle == "lower":
            mask = np.triu(np.ones_like(plot_df, dtype=bool), k=1)
        elif triangle == "upper":
            mask = np.tril(np.ones_like(plot_df, dtype=bool), k=-1)

        values = np.ma.masked_array(plot_df.values, mask=mask)
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal" if n_rows == n_cols else "auto")

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(plot_df.columns, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(plot_df.index)

        for i in range(n_rows):
            for j in range(n_cols):
                if mask is not None and mask[i, j]:
                    continue

                value = plot_df.iloc[i, j]
                text = f"{value * 100:.1f}" if fmt == "percent_no_symbol" else format(value, fmt)

                norm_val = (value - vmin) / (vmax - vmin) if (vmin is not None and vmax is not None and vmax != vmin) else 0.5
                text_color = "white" if norm_val > 0.6 else "black"

                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=text_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

        if fmt == "percent_no_symbol":
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

        ax.tick_params(axis="both", length=0, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.subplots_adjust(left=0.22, bottom=0.22, right=0.92, top=0.95)
        self.save_figure(fig, filename)
        
    def plot_pie_chart(self, fig, ax, counts, labels, labels_ncols, colors, filename):
        wedges, _, autotexts = ax.pie(
            counts.values, labels=None, colors=colors, autopct="%1.1f%%",
            startangle=90, counterclock=False, wedgeprops={"edgecolor": "white", "linewidth": 0.8}
        )

        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_weight("semibold")
            autotext.set_color("black")
            
        ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=labels_ncols)
        ax.set_aspect("auto")
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_bar_chart(self, df, x_col, y_col, filename, ylabel="", color=None, sort_by=None, ascending=False, ylim=None, horizontal_line=None, value_format=".2f", figsize=(7.2, 4.0), rotation=35, text_offset=0.01):
        plot_df = df.copy()
        if sort_by is not None:
            plot_df = plot_df.sort_values(sort_by, ascending=ascending)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            plot_df[x_col], plot_df[y_col],
            color=color or Constants.COLORS["Default"],
            edgecolor=Constants.COLORS["Edge"],
            linewidth=0.7, width=0.65
        )

        for bar, value in zip(bars, plot_df[y_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + text_offset,
                format(value, value_format), ha="center", va="bottom", fontsize=10
            )

        if horizontal_line is not None:
            ax.axhline(horizontal_line, linestyle="--", linewidth=1, color="black")

        ax.set_xticks(np.arange(len(plot_df)))
        ax.set_xticklabels(plot_df[x_col])
        
        self.finalize_plot(ax, ylabel=ylabel, rotate_x=rotation, ylim=ylim)
        self.save_figure(fig, filename)