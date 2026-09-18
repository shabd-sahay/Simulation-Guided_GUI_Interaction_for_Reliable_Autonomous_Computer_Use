"""
Data loading and panel construction.

The raw file is one row per (item, month). To forecast, we need each row to
carry what was knowable *before* the month it is predicting - otherwise the
model gets to peek at the answer. That is what build_panel does.
"""

import numpy as np
import pandas as pd

RAW_COLUMNS = ["YEAR", "MONTH", "ITEM CODE", "ITEM TYPE", "RETAIL SALES",
               "RETAIL TRANSFERS", "WAREHOUSE SALES"]


def load_raw(path):
    df = pd.read_csv(path, usecols=RAW_COLUMNS)
    df = df.dropna(subset=["RETAIL SALES", "ITEM TYPE", "ITEM CODE"])
    # A handful of rows carry negative sales (returns/corrections). They are
    # real but they are not demand, so they get clipped rather than dropped.
    df["RETAIL SALES"] = df["RETAIL SALES"].clip(lower=0)
    df["RETAIL TRANSFERS"] = df["RETAIL TRANSFERS"].clip(lower=0)
    df["WAREHOUSE SALES"] = df["WAREHOUSE SALES"].clip(lower=0)
    return df


def build_panel(df, min_history=2):
    """
    Turn the raw rows into a supervised forecasting table.

    The calendar in this dataset is not continuous - some months are simply
    missing. So "previous period" means the previous *observed* period for
    that item, not literally last month. We keep a gap column so the model
    knows how stale its own history is.
    """
    periods = (df[["YEAR", "MONTH"]].drop_duplicates()
               .sort_values(["YEAR", "MONTH"]).reset_index(drop=True))
    periods["period_idx"] = np.arange(len(periods))
    df = df.merge(periods, on=["YEAR", "MONTH"], how="left")

    df = df.sort_values(["ITEM CODE", "period_idx"])
    g = df.groupby("ITEM CODE", sort=False)

    df["lag1_sales"] = g["RETAIL SALES"].shift(1)
    df["lag2_sales"] = g["RETAIL SALES"].shift(2)
    df["lag1_warehouse"] = g["WAREHOUSE SALES"].shift(1)
    df["lag1_transfers"] = g["RETAIL TRANSFERS"].shift(1)
    df["prev_period_idx"] = g["period_idx"].shift(1)
    df["period_gap"] = df["period_idx"] - df["prev_period_idx"]

    # Rolling mean of what came before, shifted so the current month is excluded.
    df["roll3_mean"] = (g["RETAIL SALES"]
                        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    df["roll3_std"] = (g["RETAIL SALES"]
                       .transform(lambda s: s.shift(1).rolling(3, min_periods=2).std()))
    df["item_obs_count"] = g.cumcount()

    df = df[df["item_obs_count"] >= min_history].copy()
    df["roll3_std"] = df["roll3_std"].fillna(0.0)
    df = df.dropna(subset=["lag1_sales", "lag2_sales", "roll3_mean"])

    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12)
    df["target"] = df["RETAIL SALES"]
    return df.reset_index(drop=True), periods


FEATURES = ["lag1_sales", "lag2_sales", "lag1_warehouse", "lag1_transfers",
            "roll3_mean", "roll3_std", "period_gap", "item_obs_count",
            "month_sin", "month_cos", "item_type_code"]


def encode_item_type(train, *others):
    """Fit the item-type encoding on train only, then apply it everywhere."""
    types = sorted(train["ITEM TYPE"].unique())
    mapping = {t: i for i, t in enumerate(types)}
    out = []
    for frame in (train,) + others:
        f = frame.copy()
        f["item_type_code"] = f["ITEM TYPE"].map(mapping).fillna(-1).astype(int)
        out.append(f)
    return out if len(out) > 1 else out[0]


def time_split(panel, n_test_periods=5):
    """
    Split by time, never at random. The test set is the most recent periods,
    so the model is always predicting forward - the way it would be used.
    """
    cutoff = panel["period_idx"].max() - n_test_periods + 1
    train = panel[panel["period_idx"] < cutoff].copy()
    test = panel[panel["period_idx"] >= cutoff].copy()
    return train, test, cutoff
