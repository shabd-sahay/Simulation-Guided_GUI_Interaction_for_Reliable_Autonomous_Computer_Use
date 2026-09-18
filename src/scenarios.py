"""
Scenario generation.

The idea under test: if we show the model demand situations it has never
actually seen - a sudden spike, a collapse, a slow drift - does it forecast
real months any better?

Two ways to produce those scenarios:

  "llm"     - ask a language model for demand multipliers (needs transformers
              + model weights; this is the path the original notebook used)
  "offline" - draw multipliers from an explicit mixture of regimes, seeded

Whichever path produces them, every multiplier goes through the same sanity
filter before it is allowed near the training set. That filter matters more
than the generator: an LLM asked for numbers in plain English will happily
return a year, a page number, or a 47. Accepting those unchecked is how you
end up "augmenting" your data with garbage.
"""

import re
import numpy as np

# A multiplier outside this band is not a demand scenario, it is a typo.
MIN_MULT, MAX_MULT = 0.2, 3.0

# Which language model the "llm" mode uses. Instruction-tuned models follow
# the "answer with numbers only" instruction far better than a raw base model
# like gpt2, which tends to ramble. Any HuggingFace text-generation model id
# works here; these are ordered small-to-large.
LLM_CHOICES = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "gpt2-medium",
]
DEFAULT_LLM = LLM_CHOICES[0]

LLM_PROMPT = (
    "A retail warehouse is planning inventory. List 10 realistic demand "
    "multipliers relative to a normal month, covering quiet months, normal "
    "months, and busy months. Answer with decimal numbers only, separated by "
    "commas.\nAnswer: 0.6,"
)


def sanity_filter(values):
    """Keep only numbers that could plausibly be a demand multiplier."""
    return [float(v) for v in values if MIN_MULT <= float(v) <= MAX_MULT]


def _parse_numbers(text):
    return [float(m) for m in re.findall(r"\d+\.\d+", text)]


def generate_llm(n, model_name=DEFAULT_LLM, seed=0, return_trace=False):
    """
    Ask a language model for multipliers. Requires `transformers` and network
    access to fetch weights. Raises if unavailable - deliberately, so a failed
    LLM run can never silently masquerade as a successful one.

    Instruction-tuned chat models are prompted through their chat template
    when they have one; base models (gpt2) get the raw prompt. Either way the
    output is free text, so it gets parsed and sanity-filtered identically.
    """
    from transformers import pipeline, set_seed

    set_seed(seed)
    gen = pipeline("text-generation", model=model_name)
    tok = getattr(gen, "tokenizer", None)
    use_chat = bool(getattr(tok, "chat_template", None))

    collected = []
    trace = {"samples": [], "n_raw_parsed": 0, "n_kept": 0, "attempts": 0,
             "model": model_name, "chat_template": use_chat}
    attempts = 0
    while len(collected) < n and attempts < 40:
        if use_chat:
            out = gen([{"role": "user", "content": LLM_PROMPT}],
                      max_new_tokens=80, temperature=0.9, do_sample=True)
            generated = out[0]["generated_text"]
            text = (generated[-1]["content"] if isinstance(generated, list)
                    else str(generated))
        else:
            out = gen(LLM_PROMPT, max_new_tokens=60, temperature=0.9,
                      do_sample=True, pad_token_id=50256)
            text = out[0]["generated_text"][len(LLM_PROMPT):]

        raw_nums = _parse_numbers(text)
        kept = sanity_filter(raw_nums)
        collected.extend(kept)
        trace["n_raw_parsed"] += len(raw_nums)
        trace["n_kept"] += len(kept)
        if len(trace["samples"]) < 3:
            trace["samples"].append({"generated_text": text.strip()[:300],
                                     "parsed": raw_nums, "kept": kept})
        attempts += 1
    trace["attempts"] = attempts
    if not collected:
        raise RuntimeError(
            f"{model_name} returned no usable multipliers after 40 tries. "
            f"Rather than fall back silently, this stops - try a different "
            f"model from LLM_CHOICES."
        )
    result = np.array(collected[:n])
    return (result, trace) if return_trace else result


def generate_offline(n, seed=0, return_trace=False):
    """
    Explicit regime mixture, no model required. Roughly: mostly normal months,
    with a minority of genuinely quiet and genuinely busy ones, plus a thin
    tail of shocks. These proportions are a modelling assumption, not a
    measurement - they are stated here so they can be argued with.
    """
    rng = np.random.default_rng(seed)
    regimes = [
        ("normal", 0.55, 1.00, 0.10),
        ("quiet",  0.20, 0.72, 0.10),
        ("busy",   0.20, 1.35, 0.18),
        ("shock",  0.05, 1.90, 0.35),
    ]
    weights = np.array([r[1] for r in regimes])
    weights = weights / weights.sum()
    picks = rng.choice(len(regimes), size=n * 3, p=weights)
    raw = np.array([rng.normal(regimes[p][2], regimes[p][3]) for p in picks])
    kept_mask = (raw >= MIN_MULT) & (raw <= MAX_MULT)
    kept = raw[kept_mask].tolist()
    kept_regimes = [regimes[p][0] for p, m in zip(picks, kept_mask) if m]
    if len(kept) < n:  # extremely unlikely, but do not return a short array
        kept = kept + [1.0] * (n - len(kept))
        kept_regimes = kept_regimes + ["normal"] * (n - len(kept_regimes))
    result = np.array(kept[:n])
    if not return_trace:
        return result
    trace = {
        "regimes": [{"name": r[0], "probability": r[1], "mean": r[2], "std": r[3]}
                    for r in regimes],
        "realized_counts": {name: kept_regimes[:n].count(name)
                            for name, *_ in regimes},
    }
    return result, trace


def generate_learned(n, train, seed=0, return_trace=False, max_components=6):
    """
    Learn the scenario distribution from the data itself, instead of assuming
    it (offline) or borrowing it from a model that has never seen this domain
    (llm).

    The idea: every consecutive pair of months for a real item already *is* a
    demand scenario. If an item sold 100 last month and 137 this month, the
    world generated a 1.37x scenario. Collect every one of those real ratios,
    fit a probability distribution to them, and sample new ones from it.

    A Gaussian Mixture Model is used because demand changes are not one smooth
    bell curve - there are genuinely different modes (flat months, growth
    months, collapse months). The number of modes is chosen by BIC rather than
    asserted, so the data decides how many regimes exist rather than us.

    Fitting happens on TRAINING ROWS ONLY. Fitting on the full dataset would
    leak information about the test months into the generator, which would
    quietly invalidate the whole comparison.
    """
    from sklearn.mixture import GaussianMixture

    prev = train["lag1_sales"].to_numpy(dtype=float)
    curr = train["target"].to_numpy(dtype=float)

    # Only pairs where both months have real movement give a meaningful ratio.
    usable = (prev > 0) & (curr > 0)
    ratios = curr[usable] / prev[usable]
    ratios = ratios[(ratios >= MIN_MULT) & (ratios <= MAX_MULT)]
    if len(ratios) < 50:
        raise RuntimeError(
            f"Only {len(ratios)} usable month-to-month ratios found in the "
            f"training data - too few to learn a distribution from. Use "
            f"'offline' mode for this dataset."
        )

    # Work in log space: demand changes are multiplicative, so log-ratios are
    # roughly symmetric and far better suited to a mixture of Gaussians than
    # raw ratios (which are hard-floored at 0 and skewed right).
    log_ratios = np.log(ratios).reshape(-1, 1)

    # Cap the fitting sample - a few tens of thousands of points is far more
    # than enough to pin down a 1-D density, and it keeps the GUI responsive.
    rng = np.random.default_rng(seed)
    if len(log_ratios) > 40000:
        idx = rng.choice(len(log_ratios), size=40000, replace=False)
        fit_data = log_ratios[idx]
    else:
        fit_data = log_ratios

    best, best_bic, bic_table = None, np.inf, []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, random_state=seed,
                             covariance_type="full", max_iter=200)
        gm.fit(fit_data)
        bic = gm.bic(fit_data)
        bic_table.append({"components": k, "BIC": float(bic)})
        if bic < best_bic:
            best, best_bic = gm, bic

    # Oversample then filter, so the returned array is always full length.
    # NOTE: GaussianMixture.sample() returns points grouped by component, so
    # the array must be shuffled before truncating to n - otherwise the first
    # n points over-represent whichever components happen to come first, and
    # the sampled distribution silently stops matching the fitted one.
    drawn, tries = [], 0
    while len(drawn) < n and tries < 20:
        sample_log, _ = best.sample(max(n * 3, 1000))
        sample = np.exp(sample_log.ravel())
        rng.shuffle(sample)
        drawn.extend(sanity_filter(sample))
        tries += 1
    if len(drawn) < n:
        drawn = drawn + [1.0] * (n - len(drawn))
    drawn = np.array(drawn)
    rng.shuffle(drawn)
    result = drawn[:n]

    if not return_trace:
        return result

    means = np.exp(best.means_.ravel())
    weights = best.weights_.ravel()
    order = np.argsort(means)
    trace = {
        "n_real_ratios": int(len(ratios)),
        "n_fit_sample": int(len(fit_data)),
        "chosen_components": int(best.n_components),
        "bic_table": bic_table,
        "components": [
            {"mean_multiplier": float(means[i]),
             "share": float(weights[i])}
            for i in order
        ],
        "real_ratio_sample": ratios[rng.choice(len(ratios),
                                               size=min(2000, len(ratios)),
                                               replace=False)].tolist(),
    }
    return result, trace


def get_multipliers(n, mode="offline", seed=0, model_name=DEFAULT_LLM,
                    return_trace=False, train=None):
    """
    One entry point for all three generators.

    mode="offline" - hand-specified regime mixture (an assumption)
    mode="llm"     - a general-purpose language model (borrowed knowledge)
    mode="learned" - a mixture model fit on this dataset's own history
                     (requires `train`)
    """
    if mode == "llm":
        return generate_llm(n, model_name=model_name, seed=seed,
                            return_trace=return_trace)
    if mode == "offline":
        return generate_offline(n, seed=seed, return_trace=return_trace)
    if mode == "learned":
        if train is None:
            raise ValueError(
                "mode='learned' needs the training rows to learn from - "
                "pass train=<training dataframe>."
            )
        return generate_learned(n, train, seed=seed, return_trace=return_trace)
    raise ValueError(f"unknown scenario mode: {mode}")


MODES = ["offline", "llm", "learned"]

MODE_LABELS = {
    "offline": "Hand-written rules",
    "llm": "General-purpose language model",
    "learned": "Model trained on this data",
}

MODE_BLURBS = {
    "offline": "A fixed recipe someone wrote by hand: mostly normal months, "
               "some quiet ones, some busy ones, a few shocks. It never looks "
               "at your data at all.",
    "llm": "A general-purpose AI language model is asked, in plain English, "
           "to suggest realistic demand multipliers. It knows a lot about "
           "language and very little about your warehouse.",
    "learned": "A statistical model trained on your own sales history. It "
               "measures how demand actually moved month to month in your "
               "data, then generates new scenarios from that measured "
               "pattern.",
}


def augment_training_frame(train, features, multipliers, frac=1.0, seed=0):
    """
    Build synthetic training rows by re-scaling the demand-valued columns of
    real rows - history *and* target together, so each synthetic row stays
    internally consistent. A row where last month tripled but the target did
    not would teach the model something false.

    Only training rows are ever augmented. The test set stays untouched real
    data, always.
    """
    rng = np.random.default_rng(seed)
    n_syn = int(len(train) * frac)
    idx = rng.choice(len(train), size=n_syn, replace=True)
    syn = train.iloc[idx].copy().reset_index(drop=True)
    mult = rng.choice(multipliers, size=n_syn, replace=True)

    demand_cols = ["lag1_sales", "lag2_sales", "lag1_warehouse",
                   "lag1_transfers", "roll3_mean", "roll3_std", "target"]
    for c in demand_cols:
        syn[c] = syn[c].to_numpy() * mult

    syn["is_synthetic"] = 1
    real = train.copy()
    real["is_synthetic"] = 0
    import pandas as pd
    return pd.concat([real, syn], ignore_index=True)
