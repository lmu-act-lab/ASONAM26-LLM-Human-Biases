from pathlib import Path
import numpy as np
import pandas as pd

from constants import Constants


def _allocate_by_fractions(base: pd.Series, strata_counts: pd.Series, fractions: pd.Series, remaining: int) -> int:
    """Allocate remaining quota using largest fractional parts."""
    for idx in fractions.index:
        if remaining == 0:
            break
        if base.loc[idx] < strata_counts.loc[idx]:
            base.loc[idx] += 1
            remaining -= 1
    return remaining


def _allocate_remaining_capacity(base: pd.Series, strata_counts: pd.Series, remaining: int) -> int:
    """Fill any remaining slots where capacity exists."""
    for idx in strata_counts.index:
        if remaining == 0:
            break
        capacity_left = int(strata_counts.loc[idx] - base.loc[idx])
        if capacity_left > 0:
            add = min(capacity_left, remaining)
            base.loc[idx] += add
            remaining -= add
    return remaining


def allocate_strata_quota(strata_counts: pd.Series, total_needed: int) -> pd.Series:
    """Allocate total_needed across strata proportionally using the largest-remainder method."""
    total_available = int(strata_counts.sum())
    if total_needed > total_available:
        raise ValueError(f"Requested {total_needed} but only {total_available} rows available.")

    proportions = strata_counts / total_available
    raw = proportions * total_needed
    base = np.floor(raw).astype(int)

    remaining = total_needed - int(base.sum())
    fractions = (raw - base).sort_values(ascending=False)

    remaining = _allocate_by_fractions(base, strata_counts, fractions, remaining)
    remaining = _allocate_remaining_capacity(base, strata_counts, remaining)

    if int(base.sum()) != total_needed:
        raise RuntimeError("Could not allocate quotas to match required total.")

    return base


def main():
    Path(Constants.DATA_DIR).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)

    required_cols = ["article_id", "index", "source", "gender", "politics"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["row_number"] = df.index

    target_n = round(len(df) * Constants.SAMPLE_FRACTION)
    print(f"Total rows: {len(df)}")
    print(f"Target sample size ({Constants.SAMPLE_FRACTION * 100}%): {target_n}")

    strata_cols = ["source", "gender", "politics"]
    strata_counts = df.groupby(strata_cols, dropna=False).size()
    print(f"\nTotal strata (source x gender x politics): {len(strata_counts)}")

    target_per_stratum = allocate_strata_quota(strata_counts, target_n)
    rng = np.random.default_rng(Constants.RANDOM_SEED)
    
    grouped = df.groupby(strata_cols, dropna=False)
    picked = []

    for strata_key, take_n in target_per_stratum.items():
        if take_n <= 0:
            continue
            
        group_df = grouped.get_group(strata_key)
        sampled = group_df.sample(
            n=int(take_n),
            replace=False,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        picked.append(sampled)

    sampled_df = pd.concat(picked, axis=0).sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    keep_cols = ["row_number", "article_id", "index", "source", "gender", "politics"]
    rows_df = sampled_df[keep_cols].copy().sort_values("row_number").reset_index(drop=True)
    rows_df.to_csv(Constants.SAMPLE_ROWS_DATA_FILE, index=False)

    print(f"\nSaved:\n  {Constants.SAMPLE_ROWS_DATA_FILE} ({len(rows_df)} rows)")
    print(f"\nRow numbers ({len(rows_df)} total):\n{rows_df['row_number'].tolist()}")

    # DRY reporting interface loops
    reporting_groups = [
        ("source", ["source"]),
        ("source x gender", ["source", "gender"]),
        ("source x politics", ["source", "politics"]),
        ("source x gender x politics", ["source", "gender", "politics"])
    ]

    for label, group_cols in reporting_groups:
        print(f"\nDistribution by {label}:")
        counts = rows_df.groupby(group_cols, dropna=False).size() if len(group_cols) > 1 else rows_df[group_cols[0]].value_counts(dropna=False)
        print(counts.to_string())


if __name__ == "__main__":
    main()