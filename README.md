# Can AI invent useful training data? Three attempts, honestly measured.

A warehouse wants to know how much of each product it will sell next month.
There is a popular idea in applied machine learning that says: your sales
history only contains the months that actually happened, so use AI to *invent*
the ones that didn't — sudden rushes, collapses, strange quiet spells — add
those to your training data, and your forecasting model will handle surprises
better.

It is a genuinely appealing idea. This project tests whether it is true, using
three different ways of inventing those situations, on 307,645 rows of real
sales data.

**Short answer: none of the three improved the forecasts.** The most interesting
part is *how they failed differently* — and that the one trained on the data
itself came closest to doing no harm, while the hand-written one measurably hurt.

---

## Table of contents

- [What problem this actually solves](#what-problem-this-actually-solves)
- [Every number in this project, explained](#every-number-in-this-project-explained)
- [The three scenario writers](#the-three-scenario-writers)
- [How the comparison is kept fair](#how-the-comparison-is-kept-fair)
- [Results](#results)
- [What I would not claim from this](#what-i-would-not-claim-from-this)
- [Running it yourself](#running-it-yourself)
- [How the code is organised](#how-the-code-is-organised)
- [Using your own data](#using-your-own-data)
- [Honest limitations](#honest-limitations)
- [What this project was rebuilt from](#what-this-project-was-rebuilt-from)

---

## What problem this actually solves

Retail and warehouse businesses order stock in advance. Order too much and cash
sits on a shelf; order too little and you lose the sale. Everything rests on a
forecast: *how many units of this product will move next month?*

Sales history is the obvious thing to learn from. The catch is that history is
a record of what happened, not of what could happen. A model trained on three
calm years has never seen a panic-buying month. So when one arrives, the model
has no idea what to do.

The proposed fix — **synthetic data augmentation** — is to manufacture the
missing situations and train on them too. This is genuinely standard practice in
some fields (rotating and cropping photos to train image classifiers works very
well). Whether it works for demand forecasting is the question here.

### The specific question being tested

> If you take real sales history, add AI-invented demand scenarios to it, and
> train a forecasting model on the combination — does that model predict real
> future months more accurately than one trained on the real history alone?

Everything in this repository exists to answer that as fairly as possible, with
the answer allowed to come back "no."

---

## Every number in this project, explained

This section is for anyone who doesn't work in machine learning daily. None of
these are complicated ideas; they just have jargon attached.

### What the model predicts
For each product, in each month: **how many units will sell.** The dataset calls
this "retail sales." That's it — one number per product per month.

### "Average miss" — the main score (MAE)
If the model predicts 12 units and 8 actually sold, it missed by 4. Do that for
every product in every tested month, then average all those misses.

**Lower is better.** An average miss of 3.1 means the model is typically about
3 units off for a given product in a given month.

The technical name is **Mean Absolute Error (MAE)**. It's called "absolute"
because over-predicting by 4 and under-predicting by 4 both count as a miss of
4 — the direction doesn't cancel out.

### RMSE — the other score you'll see
Same idea, but big misses are punished much more harshly than small ones
(errors are squared before averaging). A model can have a *better* MAE and a
*worse* RMSE than another — that means it's usually closer, but occasionally
catastrophically wrong. Both are reported here because showing only the
flattering one would be cherry-picking.

### The naive benchmark
The simplest forecast imaginable: **"this product will sell exactly what it sold
last month."** No model, no training, no AI.

This matters enormously. Any trained model that can't beat this is not earning
its complexity, and a surprising number of published models quietly don't. In
this project the naive guess is always computed and shown first, as a floor.

### Training data vs. test data
The model is shown older months to learn from (**training data**), and then
judged on the most recent months, which it has never seen (**test data**).

Critically, the split is **chronological, never random**. A random split would
let the model train on March and be tested on January — learning from the future
to predict the past, which no real deployment can do. This detail invalidates a
lot of amateur forecasting projects.

### "Seeds" / "runs"
Training involves randomness, so running the identical setup twice gives
slightly different results. Everything here is repeated with several different
random starting points ("seeds"). If a method only wins on some seeds, that's
luck, not an improvement.

### The p-value
It answers one question: *"if these two methods were really equally good, how
often would I see a difference this big purely by chance?"*

- **p below 0.05** — the difference is probably real.
- **p above 0.05** — can't distinguish them; the difference may well be noise.

A large p-value is **not a failed experiment.** "These are indistinguishable" is
a legitimate, useful finding, and it's most of what this project found.

### The multiplier (what a "scenario" actually is)
Every invented scenario here is a single number: a **demand multiplier.**

- `1.4` = "a month 40% busier than usual"
- `0.7` = "a month 30% quieter than usual"
- `1.0` = "a completely normal month"

To build a synthetic training row, a real row is picked and everything about it
— its history *and* its answer — is scaled by that multiplier together. Scaling
the history without the answer would teach the model something false.

---

## The three scenario writers

This is the heart of the project. Three fundamentally different philosophies for
inventing demand scenarios, all measured the same way.

### 1. Hand-written rules (`offline`)
A fixed recipe a human wrote: 55% normal months, 20% quiet, 20% busy, 5% rare
shocks, each with an assumed average and spread.

**It never looks at your data at all.** That's precisely why it's included — it
represents "an expert's assumption," and it's the thing the other two should be
able to beat if they're worth anything.

### 2. A general-purpose language model (`llm`)
An actual language model is asked, in plain English, to suggest realistic demand
multipliers. Whatever numbers appear in its reply are parsed out and checked.

Defaults to `Qwen/Qwen2.5-0.5B-Instruct`; `SmolLM2-360M-Instruct`,
`Qwen2.5-1.5B-Instruct` and `gpt2-medium` are also selectable. Instruction-tuned
models follow "answer with numbers only" far better than raw base models like
GPT-2, which tend to ramble.

This arm has an obvious conceptual weakness, and it's the point: **the model
knows a great deal about language and nothing whatsoever about your warehouse.**
It has never seen your sales history. It is guessing from general world
knowledge.

Nothing it says is trusted. Every number passes the same plausibility filter
(between 0.2× and 3×) before use, because a language model asked for numbers in
prose will cheerfully return a year, a page number, or a stray `47`. The GUI
shows you exactly what it wrote and which numbers survived.

### 3. A model trained on this data (`learned`)
The interesting one. Rather than assuming what demand scenarios look like, or
borrowing an opinion from a model that's never seen the domain, **this measures
them from the data itself.**

The insight: *every consecutive pair of months for every product already is a
real demand scenario.* If a product sold 100 units then 137, the world just
generated a 1.37× month. Your history contains hundreds of thousands of these.

So the method is:

1. Compute the month-to-month ratio for every product pair in the **training
   data only** (111,682 usable ratios in the sample dataset).
2. Convert to log space — demand changes are multiplicative, so log-ratios are
   roughly symmetric and far better suited to what comes next.
3. Fit a **Gaussian Mixture Model**: a statistical model that represents the
   data as a blend of several overlapping bell curves, capturing the fact that
   demand changes aren't one smooth pattern but several distinct regimes.
4. **Let the data choose how many regimes exist**, using BIC (a standard
   criterion that rewards fit while penalising unnecessary complexity). On the
   sample data it selects 6 — more structure than the hand-written recipe's
   assumed 4.
5. Sample new multipliers from the fitted distribution.

It is fit on **training rows only**. Fitting on the full dataset would leak
information about the test months into the generator and quietly invalidate the
entire comparison.

#### Does it actually work?

Comparing the generated multipliers against the real month-to-month changes:

| Percentile | Real history | Learned generator | Hand-written rules |
|---|---|---|---|
| 5th | 0.33 | 0.32 | 0.65 |
| 25th | 0.67 | 0.66 | 0.88 |
| 50th (middle) | 1.00 | 0.98 | 1.01 |
| 75th | 1.37 | 1.36 | 1.16 |
| 95th | 2.40 | 2.31 | 1.63 |

A Kolmogorov–Smirnov test (which measures how far apart two distributions are;
smaller is closer) gives **0.030 for the learned generator versus 0.201 for the
hand-written rules** — roughly a **7× better match to reality**.

The hand-written recipe is visibly too timid: it almost never invents the very
quiet or very busy months that genuinely occur. The learned generator does,
because it measured them.

---

## How the comparison is kept fair

The result is only worth anything if the arms are genuinely comparable. The
guarantees:

- **Identical test data.** Every arm is judged on the same real, held-out,
  most-recent months. Synthetic data never enters the test set under any
  circumstances.
- **Identical model.** Same algorithm (gradient-boosted trees), same
  hyperparameters, same random seed per run. The *only* thing that varies is
  where the synthetic training rows came from.
- **Chronological split.** Always predicting forward, never backward.
- **Repeated runs.** Multiple seeds, with per-seed pairing, so each arm is
  compared against the baseline that shared its exact random conditions.
- **Paired statistics.** Wilcoxon signed-rank test, which suits paired
  comparisons and doesn't assume the differences are normally distributed.
- **An always-present floor.** The naive "same as last month" benchmark is
  computed every run.
- **Internally consistent synthetic rows.** History and target are scaled
  together by the same multiplier.
- **No leakage in the generator.** The learned generator sees training rows
  only.

---

## Results

Reported here is a real run on the sample dataset — 5 seeds, +200% synthetic
data added, 5 months held out. (Exact figures are version-sensitive — see the
reproducibility note below — but the shape of the result is what to trust.)

| Method | Average miss (MAE) | vs. no invented data | Beat baseline in | p-value |
|---|---|---|---|---|
| Naive — "same as last month" | 3.611 | — | — | — |
| **No invented data (real only)** | **3.100** | — | — | — |
| Model trained on this data | 3.095 | −0.005 (slightly better) | 3 / 5 seeds | 0.44 |
| Hand-written rules | 3.123 | +0.022 (worse) | 1 / 5 seeds | 0.13 |

### Reading this honestly

**The forecasting model itself works.** 3.100 against the naive benchmark's
3.611 is a real, substantial improvement — the model is genuinely learning
something from history.

**Neither scenario writer significantly beat simply not inventing anything** —
that's the headline answer, and it's a negative one. But the two didn't fail
identically, and that's the finding worth discussing:

- **Hand-written rules hurt** (+0.022, losing on 4 of 5 seeds). Feeding a model
  confidently-wrong assumptions about your domain is worse than feeding it
  nothing.
- **The learned generator came out very slightly ahead of both alternatives**
  (−0.005 vs. no augmentation, winning on 3 of 5 seeds) — but the p-value
  (0.44) says this margin is well within what randomness alone could produce.
  It's fair to call it a draw with "no augmentation," not a win. What's
  genuinely true, though: it did no measurable harm, unlike the hand-written
  version — consistent with it generating data 7× closer to the real
  distribution (see the table above). It's adding rows that look so much like
  real data that the model learns almost exactly what it would have anyway.

That last point is the genuinely interesting conclusion, and it's a bit of a
paradox worth sitting with:

> **The better your synthetic data imitates reality, the less new information it
> adds.** Rescaling a real month by 1.37 produces something the model could
> already interpolate to. Augmentation of this shape adds *volume*, not
> *knowledge* — and volume is not what this model was short of.

An augmentation strategy that helped would need to invent situations that are
plausible but genuinely *structurally* different from anything in the history —
not louder or quieter versions of what's already there. That's a substantially
harder problem, and it's the honest direction for future work here.

### A note on reproducibility

Exact figures depend on installed versions of scikit-learn, numpy and pandas.
Gradient boosting's histogram binning and floating-point summation order aren't
guaranteed identical across releases, so the same code and same seed can produce
numbers differing in the second decimal place on a different machine. The
direction and shape of the result is what to trust, not the last digit.

Run `pip freeze > requirements.txt` after your first successful run and commit
it — that's what actually documents which versions produced which numbers.

---

## What I would not claim from this

- **Not** that synthetic data augmentation doesn't work. It demonstrably does in
  other domains.
- **Not** that language models can't generate useful training data. This tested
  small models, ungrounded, on one narrow task.
- **Only** that *these three* ways of generating *this kind* of scenario did not
  improve *this* forecaster on *this* dataset — measured as carefully as I knew
  how to measure it.

---

## Running it yourself

```bash
pip install pandas numpy scikit-learn scipy streamlit
```

### The interactive app (recommended)

```bash
streamlit run app.py
```

Opens in your browser. Pick which scenario writers to compare, press **Run the
comparison**, and watch every step execute live — data loading, the naive
benchmark, how each generator works internally, the head-to-head contest, and
the resulting order sheet. Nothing is pre-computed.

### The command line

```bash
# three-way comparison (the headline result)
python run_comparison.py Warehouse_and_Retail_Sales.csv

# include the language-model arm (downloads model weights on first use)
python run_comparison.py Warehouse_and_Retail_Sales.csv --with-llm

# a single arm in detail
python run_experiment.py --data Warehouse_and_Retail_Sales.csv --scenario-mode learned

# how much synthetic data is optimal?
python run_sweep.py Warehouse_and_Retail_Sales.csv
```

### Deploying it publicly

The app needs a running Python backend, so it can't be a static page.
[Streamlit Community Cloud](https://share.streamlit.io) hosts it free: push this
folder to a public GitHub repo, sign in with GitHub, select the repo, set the
main file to `app.py`, and deploy.

---

## How the code is organised

```
app.py               the interactive comparison (Streamlit)
run_comparison.py    three-way head-to-head, the headline result
run_experiment.py    one arm in depth, with full metrics
run_sweep.py         how much synthetic data is best?
src/
  data.py            CSV -> forecasting table; lag features; chronological split
  scenarios.py       the three generators + the augmentation itself
  decide.py          forecast -> order quantity (a business rule, not a model)
results/             CSV and JSON output from runs
```

The separation of `decide.py` is deliberate. The forecast is *learned from data*
and can be evaluated. The order quantities are *policy choices someone picked*.
Blurring those two is how projects end up claiming a model "decides" when really
an if-statement does.

---

## Using your own data

The app accepts any CSV with these columns:

`YEAR`, `MONTH`, `ITEM CODE`, `ITEM TYPE`, `RETAIL SALES`, `RETAIL TRANSFERS`,
`WAREHOUSE SALES`

You need enough months of history that hiding several for testing still leaves
the model something to learn from — the app checks this and tells you if your
file is too short. The sample dataset (Maryland warehouse and retail sales,
included) exists so the whole pipeline can be demonstrated without needing your
own file.

---

## Honest limitations

- **Calendar gaps.** The sample data is missing several months entirely, so
  "previous period" means the previous *observed* period. The gap size is given
  to the model as a feature, but it remains a flaw in the data rather than
  something the code fixes.
- **Five seeds** is enough to see that a large effect is absent. It is not
  enough to rule out a very small one.
- **One dataset, one domain, one model family.** The findings belong to this
  setting and shouldn't be over-generalised.
- **Small language models.** The `llm` arm uses models in the 0.4–1.5B parameter
  range for practicality. A large frontier model might generate better
  scenarios; that's untested here.
- **One shape of augmentation.** Everything here multiplicatively rescales
  existing rows. Generators that synthesise structurally novel situations are a
  different and more promising experiment.
- **Illustrative order policy.** The thresholds in `decide.py` were never tuned
  against a real inventory cost function, because no cost data exists here.

---

## What this project was rebuilt from

An earlier version of this had three flaws worth naming publicly, because they
are extremely common ones:

1. **No model existed.** Scenarios were generated, averaged, and passed to a
   hardcoded `if` statement. Nothing was ever trained, so nothing could be
   evaluated.
2. **The evaluation was circular.** "Predicted" demand was derived from a
   group's historical mean, then compared against that same historical mean. A
   train/test split was created and then never used.
3. **Generated numbers went unchecked.** Whatever decimals appeared in the
   language model's prose reply were regex-parsed and used directly.

Each is fixed here: a real trained model, an enforced chronological split, and a
plausibility filter every generated number must pass. The rebuild is why this
repository reports a negative result rather than a flattering one — the original
couldn't have detected either.
