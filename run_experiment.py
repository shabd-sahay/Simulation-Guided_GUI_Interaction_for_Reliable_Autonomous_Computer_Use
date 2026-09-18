"""
The experiment.

One question, asked fairly: does training on real data plus synthetic demand
scenarios forecast real future months better than training on real data alone?

Rules that make the answer mean something:
  - the test set is real data only, and it is the most recent months
  - both models are the same algorithm with the same settings
  - both are evaluated on the identical test rows
  - several seeds, because one lucky run is not a result
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from src.data import (load_raw, build_panel, time_split, encode_item_type,
                      FEATURES)
from src.scenarios import get_multipliers, augment_training_frame


def fit_predict(train, test, seed):
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_depth=None,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )
    model.fit(train[FEATURES], np.log1p(train["target"]))
    pred = np.expm1(model.predict(test[FEATURES]))
    return np.clip(pred, 0, None)


def score(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_AE": float(np.median(np.abs(y_true - y_pred))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scenario-mode", default="offline",
                    choices=["offline", "llm", "learned"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--aug-frac", type=float, default=1.0)
    ap.add_argument("--test-periods", type=int, default=5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    raw = load_raw(args.data)
    panel, periods = build_panel(raw)
    train_raw, test_raw, cutoff = time_split(panel, args.test_periods)
    train_raw, test_raw = encode_item_type(train_raw, test_raw)

    print(f"periods total      : {len(periods)}")
    print(f"train rows         : {len(train_raw):,}  (periods < {cutoff})")
    print(f"test rows (real)   : {len(test_raw):,}  (periods >= {cutoff})")
    print(f"scenario mode      : {args.scenario_mode}")
    print()

    y_true = test_raw["target"].to_numpy()

    # Naive benchmark: "this month looks like last month." Not a model, just
    # the honest floor a real model has to beat to be worth anything.
    naive_score = score(y_true, test_raw["lag1_sales"].to_numpy())
    print(f"naive (repeat last month): MAE {naive_score['MAE']:.3f} | "
          f"RMSE {naive_score['RMSE']:.3f}")
    print()

    rows = []
    for seed in args.seeds:
        base_pred = fit_predict(train_raw, test_raw, seed)
        base = score(y_true, base_pred)

        mult = get_multipliers(200, mode=args.scenario_mode, seed=seed,
                               train=train_raw)
        train_aug = augment_training_frame(
            train_raw, FEATURES, mult, frac=args.aug_frac, seed=seed)
        aug_pred = fit_predict(train_aug, test_raw, seed)
        aug = score(y_true, aug_pred)

        rows.append({"seed": seed,
                     **{f"baseline_{k}": v for k, v in base.items()},
                     **{f"augmented_{k}": v for k, v in aug.items()},
                     "mult_mean": float(np.mean(mult)),
                     "mult_min": float(np.min(mult)),
                     "mult_max": float(np.max(mult)),
                     "train_rows_aug": int(len(train_aug))})
        print(f"seed {seed}: baseline MAE {base['MAE']:.3f} | "
              f"augmented MAE {aug['MAE']:.3f}")

    df = pd.DataFrame(rows)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "per_seed_results.csv", index=False)

    # Paired comparison across seeds - same test rows, same seeds, so the
    # difference per seed is the meaningful quantity, not the raw averages.
    diff_mae = df["augmented_MAE"] - df["baseline_MAE"]
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(df["baseline_MAE"], df["augmented_MAE"])
        p = float(p)
    except Exception:
        p = None

    summary = {
        "n_seeds": len(df),
        "test_rows": int(len(test_raw)),
        "train_rows_real": int(len(train_raw)),
        "scenario_mode": args.scenario_mode,
        "aug_frac": args.aug_frac,
        "naive_MAE": naive_score["MAE"],
        "naive_RMSE": naive_score["RMSE"],
        "baseline_MAE_mean": float(df["baseline_MAE"].mean()),
        "baseline_MAE_std": float(df["baseline_MAE"].std()),
        "augmented_MAE_mean": float(df["augmented_MAE"].mean()),
        "augmented_MAE_std": float(df["augmented_MAE"].std()),
        "baseline_RMSE_mean": float(df["baseline_RMSE"].mean()),
        "augmented_RMSE_mean": float(df["augmented_RMSE"].mean()),
        "mean_MAE_delta": float(diff_mae.mean()),
        "augmented_better_in_seeds": int((diff_mae < 0).sum()),
        "wilcoxon_p_MAE": p,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
