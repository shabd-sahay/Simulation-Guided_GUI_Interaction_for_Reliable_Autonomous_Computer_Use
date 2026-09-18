"""
The three-way comparison.

Same question as run_experiment.py, but asked of every scenario generator at
once: given the choice between a hand-written recipe, a general-purpose
language model, and a model trained on this dataset's own history, does any
of them actually produce synthetic data worth training on?

Every arm is evaluated identically:
  - the same real training rows
  - the same real, held-out, most-recent test months
  - the same model type and hyperparameters
  - the same seeds
The only thing that varies is where the synthetic scenarios came from.

Usage:
  python run_comparison.py Warehouse_and_Retail_Sales.csv
  python run_comparison.py Warehouse_and_Retail_Sales.csv --with-llm
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_raw, build_panel, time_split, encode_item_type, FEATURES
from src.scenarios import (get_multipliers, augment_training_frame,
                           MODE_LABELS, DEFAULT_LLM)
from run_experiment import fit_predict, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--aug-frac", type=float, default=1.0)
    ap.add_argument("--test-periods", type=int, default=5)
    ap.add_argument("--with-llm", action="store_true",
                    help="include the language-model arm (needs transformers "
                         "and downloads model weights on first use)")
    ap.add_argument("--llm-model", default=DEFAULT_LLM)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    raw = load_raw(args.data)
    panel, periods = build_panel(raw)
    train_raw, test_raw, cutoff = time_split(panel, args.test_periods)
    train_raw, test_raw = encode_item_type(train_raw, test_raw)
    y_true = test_raw["target"].to_numpy()

    modes = ["offline", "learned"] + (["llm"] if args.with_llm else [])

    print(f"train rows        : {len(train_raw):,}")
    print(f"test rows (real)  : {len(test_raw):,}")
    print(f"arms              : no augmentation, " +
          ", ".join(MODE_LABELS[m] for m in modes))
    print()

    naive = score(y_true, test_raw["lag1_sales"].to_numpy())
    print(f"naive 'repeat last month' : MAE {naive['MAE']:.4f}")

    # Arm 0: no augmentation at all. This is the thing every generator has to
    # beat to have been worth adding.
    baseline = {}
    for seed in args.seeds:
        baseline[seed] = score(y_true, fit_predict(train_raw, test_raw, seed))
    base_mae = np.mean([baseline[s]["MAE"] for s in args.seeds])
    print(f"no augmentation           : MAE {base_mae:.4f}")
    print()

    rows = []
    for mode in modes:
        for seed in args.seeds:
            mult = get_multipliers(200, mode=mode, seed=seed,
                                   train=train_raw, model_name=args.llm_model)
            tr = augment_training_frame(train_raw, FEATURES, mult,
                                        frac=args.aug_frac, seed=seed)
            s = score(y_true, fit_predict(tr, test_raw, seed))
            rows.append({
                "mode": mode,
                "label": MODE_LABELS[mode],
                "seed": seed,
                "baseline_MAE": baseline[seed]["MAE"],
                "augmented_MAE": s["MAE"],
                "delta_MAE": s["MAE"] - baseline[seed]["MAE"],
                "augmented_RMSE": s["RMSE"],
            })
        d = pd.DataFrame(rows)
        d = d[d["mode"] == mode]
        print(f"{MODE_LABELS[mode]:<34}: MAE {d.augmented_MAE.mean():.4f}  "
              f"(delta {d.delta_MAE.mean():+.4f}, "
              f"beat baseline in {int((d.delta_MAE < 0).sum())}/{len(d)} seeds)")

    df = pd.DataFrame(rows)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "comparison_results.csv", index=False)

    summary = {"naive_MAE": naive["MAE"], "no_augmentation_MAE": float(base_mae),
               "n_seeds": len(args.seeds), "aug_frac": args.aug_frac,
               "arms": []}
    for mode in modes:
        d = df[df["mode"] == mode]
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d.baseline_MAE, d.augmented_MAE).pvalue)
        except Exception:
            p = None
        summary["arms"].append({
            "mode": mode,
            "label": MODE_LABELS[mode],
            "MAE_mean": float(d.augmented_MAE.mean()),
            "MAE_std": float(d.augmented_MAE.std()),
            "delta_vs_no_augmentation": float(d.delta_MAE.mean()),
            "beat_baseline_in_seeds": int((d.delta_MAE < 0).sum()),
            "wilcoxon_p": p,
        })

    with open(outdir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
