"""
Augmentation-volume sweep.

A single comparison can be an accident of how much synthetic data happened to
be added. This runs the same paired comparison at several volumes so the
conclusion does not rest on one arbitrary choice.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_raw, build_panel, time_split, encode_item_type, FEATURES
from src.scenarios import get_multipliers, augment_training_frame
from run_experiment import fit_predict, score

DATA = sys.argv[1] if len(sys.argv) > 1 else "../Logistics_genai_LLM-main/Warehouse_and_Retail_Sales.csv"
SEEDS = [0, 1, 2, 3, 4]
FRACS = [0.25, 0.5, 1.0, 2.0]

raw = load_raw(DATA)
panel, periods = build_panel(raw)
train_raw, test_raw, cutoff = time_split(panel, 5)
train_raw, test_raw = encode_item_type(train_raw, test_raw)
y_true = test_raw["target"].to_numpy()

print(f"train {len(train_raw):,} rows | test {len(test_raw):,} real rows "
      f"| cutoff period {cutoff}\n")

baseline = {}
for seed in SEEDS:
    baseline[seed] = score(y_true, fit_predict(train_raw, test_raw, seed))
    print(f"baseline seed {seed}: MAE {baseline[seed]['MAE']:.4f}")
print()

rows = []
for frac in FRACS:
    for seed in SEEDS:
        mult = get_multipliers(200, mode="offline", seed=seed)
        tr = augment_training_frame(train_raw, FEATURES, mult, frac=frac, seed=seed)
        s = score(y_true, fit_predict(tr, test_raw, seed))
        rows.append({"aug_frac": frac, "seed": seed,
                     "baseline_MAE": baseline[seed]["MAE"],
                     "augmented_MAE": s["MAE"],
                     "baseline_RMSE": baseline[seed]["RMSE"],
                     "augmented_RMSE": s["RMSE"],
                     "delta_MAE": s["MAE"] - baseline[seed]["MAE"]})
    d = pd.DataFrame(rows)
    d = d[d.aug_frac == frac]
    print(f"frac {frac}: mean delta MAE {d.delta_MAE.mean():+.4f} "
          f"| augmented better in {int((d.delta_MAE < 0).sum())}/{len(d)} seeds")

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/sweep_results.csv", index=False)

summary = []
for frac in FRACS:
    d = df[df.aug_frac == frac]
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(d.baseline_MAE, d.augmented_MAE).pvalue)
    except Exception:
        p = None
    summary.append({
        "aug_frac": frac,
        "baseline_MAE_mean": float(d.baseline_MAE.mean()),
        "augmented_MAE_mean": float(d.augmented_MAE.mean()),
        "mean_delta_MAE": float(d.delta_MAE.mean()),
        "augmented_better_seeds": int((d.delta_MAE < 0).sum()),
        "n_seeds": len(d),
        "wilcoxon_p": p,
    })

with open("results/sweep_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print()
print(json.dumps(summary, indent=2))
