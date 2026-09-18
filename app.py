"""
GUI for the scenario-generator comparison.

A viewer over the same functions the CLI scripts use (src/ + run_experiment),
never a second implementation - so what you see here is what those scripts
produce.

Run:  streamlit run app.py
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.data import (load_raw, build_panel, time_split, encode_item_type,
                      FEATURES, RAW_COLUMNS)
from src.scenarios import (get_multipliers, augment_training_frame,
                           MODE_LABELS, MODE_BLURBS, LLM_CHOICES, DEFAULT_LLM)
from src.decide import decide
from run_experiment import fit_predict, score

st.set_page_config(page_title="Which AI writes better what-if scenarios?",
                    page_icon="📦", layout="wide")

# Only hand-drawn elements get explicit colours, each with a dark-mode pair.
# Streamlit's own widgets already follow the viewer's theme correctly.
st.markdown("""
<style>
  :root { --card-bg:#fffdf8; --card-border:#e2ddd2; --card-text:#1c1a17; }
  @media (prefers-color-scheme: dark) {
    :root { --card-bg:#262019; --card-border:#3a3226; --card-text:#f0ece2; }
  }
  .block-container { padding-top:2rem; max-width:1060px; }
  h1,h2,h3 { font-family:Georgia,serif; letter-spacing:-0.01em; }
  .card, .verdict-box {
    background:var(--card-bg); border:1px solid var(--card-border);
    border-left:4px solid #7a5c2e; border-radius:4px;
    padding:16px 20px; margin:10px 0; color:var(--card-text);
  }
  .card.plain { border-left-color:#6b8a7a; font-size:0.93em; }
  .stButton>button { background:#7a5c2e; color:white; border:none; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

st.title("📦 Which AI writes better what-if scenarios?")

st.markdown("""
**The problem, in one paragraph.** A warehouse wants to predict how much of each
product it will sell next month, so it can order the right amount. You can train
a computer model on sales history to do that. But history only contains the
months that actually happened — it has never seen a sudden rush, a collapse, or a
strange quiet spell that didn't occur in your records. A popular idea says: use
AI to *invent* those missing situations, add them to the training data, and the
model will handle surprises better.

**This app tests whether that's actually true** — and compares three different
ways of inventing those situations, including one trained on your own data.
""")

st.markdown("""
<div class="card">
<b>📌 Documented reference result</b> (5 repeat runs, +200% synthetic data, verified and
written up in the README):

Model trained on this data: <b>3.095</b> average miss vs. <b>3.100</b> with no invented
data (won 3 of 5 runs, p=0.44 — a small, statistically inconclusive edge).
Hand-written rules: <b>3.123</b> (worse, p=0.13).

<br><br>The live run below repeats this from scratch, in front of you, with real
randomness — so it can land slightly differently on any given click,
<i>especially with fewer repeat runs</i>. That's expected behavior for a genuine
near-tie, not a bug. The number above is the one to trust as the finding.
</div>
""", unsafe_allow_html=True)

with st.expander("📖 First time here? Read this — it explains every number you'll see"):
    st.markdown("""
**What the model is predicting.** For each product, in each month: how many units
will sell. The dataset records this as "retail sales."

**How we check if it's any good.** We hide the most recent few months from the
model completely. It trains only on older data. Then we ask it to predict those
hidden months and compare its answers against what really sold. The model never
sees the answers while learning — otherwise it would just be memorising.

**"Average miss" (technically called MAE).** If the model predicts 12 units and
8 actually sold, it missed by 4. Do that for every product and every hidden
month, then take the average. **Lower is better.** An average miss of 3.1 means
the model is typically off by about 3 units on a given product in a given month.

**Why we compare against a "naive guess".** The simplest possible forecast is
"this product will sell exactly what it sold last month." If a trained model
can't beat that, the model isn't earning its complexity. So we always show that
number first as a floor.

**What "seeds" / "runs" mean.** Training involves randomness, so the same setup
run twice gives slightly different results. We repeat everything several times
with different random starting points. If one method only wins sometimes, that's
luck, not a real improvement.

**What the p-value means.** It answers: "if these two methods were actually
equally good, how likely is a difference this big just by chance?" A small
p-value (below 0.05) means the difference is probably real. A large one means
we genuinely can't tell them apart yet — which is a legitimate finding, not a
failure.
    """)

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("Settings")

    st.subheader("1. Data")
    source = st.radio("Data source", ["Use the sample data", "Upload my own CSV"])
    data_path, upload_error = None, None
    if source == "Use the sample data":
        data_path = "Warehouse_and_Retail_Sales.csv"
        st.caption("Maryland warehouse & retail sales, ~307k rows across "
                  "34,000 products and 24 months.")
        with st.expander("Change file path"):
            data_path = st.text_input("CSV path", value=data_path)
    else:
        st.caption("Your CSV needs these columns: " +
                  ", ".join(f"`{c}`" for c in RAW_COLUMNS))
        up = st.file_uploader("Your CSV", type=["csv"])
        if up is not None:
            try:
                head = pd.read_csv(up, nrows=5)
                missing = [c for c in RAW_COLUMNS if c not in head.columns]
                if missing:
                    upload_error = ("Missing column(s): " +
                                    ", ".join(f"`{c}`" for c in missing))
                else:
                    up.seek(0)
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                    tmp.write(up.getvalue()); tmp.close()
                    data_path = tmp.name
                    st.success(f"`{up.name}` looks right.")
            except Exception as e:
                upload_error = f"Couldn't read that CSV: {e}"
        if upload_error:
            st.error(upload_error)

    st.subheader("2. Which scenario-writers to compare")
    use_offline = st.checkbox("Hand-written rules", value=True,
                              help=MODE_BLURBS["offline"])
    use_learned = st.checkbox("Model trained on this data", value=True,
                              help=MODE_BLURBS["learned"])
    use_llm = st.checkbox("General-purpose language model", value=False,
                          help=MODE_BLURBS["llm"] +
                               " Downloads model weights on first use.")
    llm_model = DEFAULT_LLM
    if use_llm:
        llm_model = st.selectbox("Which language model", LLM_CHOICES, index=0)

    st.subheader("3. How carefully to test")
    n_seeds = st.slider("Repeat runs", 1, 5, 5,
                        help="Each run retrains everything from scratch with a "
                             "different random starting point. More runs makes "
                             "the comparison more trustworthy but slower.")
    test_periods = st.slider("Months hidden for testing", 2, 8, 5)
    aug_frac = st.select_slider("How much invented data to add",
                                options=[0.25, 0.5, 1.0, 2.0], value=1.0,
                                format_func=lambda x: f"+{int(x*100)}%")

    run_clicked = st.button("▶ Run the comparison", use_container_width=True)

modes = ([m for m, on in [("offline", use_offline), ("learned", use_learned),
                          ("llm", use_llm)] if on])
csv_ready = bool(data_path) and Path(data_path).exists()

if not csv_ready and not upload_error:
    st.info("Pick a data source in the sidebar to begin.")
if run_clicked and not modes:
    st.warning("Tick at least one scenario-writer in the sidebar to compare.")

st.divider()

# --------------------------------------------------------------------- run --
if run_clicked and csv_ready and modes:

    # -- Step 1 ------------------------------------------------------------
    s1 = st.status("**Step 1 — Reading the sales history**", expanded=True)
    t0 = time.time()
    raw = load_raw(data_path)
    s1.write(f"Read {len(raw):,} rows of sales history.")
    panel, periods = build_panel(raw)
    s1.write(f"Reshaped into {len(panel):,} learnable examples. Each one pairs "
             f"*one product in one month* with only the facts known **before** "
             f"that month — last month's sales, the recent trend, the season, "
             f"and so on. Never the answer itself.")

    if periods["period_idx"].max() < test_periods + 3:
        st.error(f"This file only covers {len(periods)} months — too few to "
                f"hide {test_periods} and still have enough left to learn "
                f"from. Lower 'months hidden', or use a longer history.")
        st.stop()

    train_raw, test_raw, cutoff = time_split(panel, test_periods)
    train_raw, test_raw = encode_item_type(train_raw, test_raw)
    s1.write(f"Hid the **{test_periods} most recent months** completely. The "
             f"model learns from everything before them, and is judged only "
             f"on them.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Examples to learn from", f"{len(train_raw):,}")
    c2.metric("Examples hidden for testing", f"{len(test_raw):,}")
    c3.metric("Products tracked", f"{raw['ITEM CODE'].nunique():,}")
    s1.update(label=f"**Step 1 — Data ready** ({time.time()-t0:.1f}s)",
             state="complete")

    y_true = test_raw["target"].to_numpy()

    # -- Step 2 ------------------------------------------------------------
    st.subheader("Step 2 — Two reference points before any AI is involved")
    naive = score(y_true, test_raw["lag1_sales"].to_numpy())
    base_scores, base_preds = {}, {}
    with st.spinner("Training a model on the real data only..."):
        for seed in range(n_seeds):
            p = fit_predict(train_raw, test_raw, seed)
            base_preds[seed] = p
            base_scores[seed] = score(y_true, p)
    base_mae = float(np.mean([base_scores[s]["MAE"] for s in range(n_seeds)]))

    c1, c2 = st.columns(2)
    c1.metric("Naive guess — 'same as last month'", f"{naive['MAE']:.3f}",
             help="Average miss, in units. No model at all.")
    c2.metric("Trained model, real data only", f"{base_mae:.3f}",
             delta=f"{base_mae - naive['MAE']:+.3f} vs naive",
             delta_color="inverse",
             help="Average miss, in units. This is the number every "
                  "scenario-writer has to improve on.")
    st.markdown(
        f'<div class="card plain">Reading these: the naive guess is off by '
        f'about <b>{naive["MAE"]:.1f} units</b> on a typical product-month. '
        f'A real model trained on history gets that down to '
        f'<b>{base_mae:.1f} units</b> — so the model is genuinely learning '
        f'something. <b>The question for the rest of this page is whether '
        f'adding AI-invented scenarios pushes it lower still.</b></div>',
        unsafe_allow_html=True)

    # -- Step 3 ------------------------------------------------------------
    st.subheader("Step 3 — Meet the scenario-writers")
    st.caption("Each one invents 'what-if' demand situations. A scenario here "
              "is a multiplier: 1.4 means 'a month 40% busier than usual', "
              "0.7 means 'a month 30% quieter'.")

    traces = {}
    tabs = st.tabs([MODE_LABELS[m] for m in modes])
    for tab, mode in zip(tabs, modes):
        with tab:
            st.write(MODE_BLURBS[mode])
            try:
                with st.spinner(f"Running {MODE_LABELS[mode]}..."):
                    mult, tr = get_multipliers(200, mode=mode, seed=0,
                                               return_trace=True,
                                               train=train_raw,
                                               model_name=llm_model)
                traces[mode] = (mult, tr)
            except Exception as e:
                st.error(f"This one failed: {e}")
                traces[mode] = None
                continue

            if mode == "offline":
                rdf = pd.DataFrame(tr["regimes"])
                rdf["actually produced"] = rdf["name"].map(tr["realized_counts"])
                rdf.columns = ["regime", "intended share", "typical multiplier",
                              "spread", "actually produced (of 200)"]
                st.dataframe(rdf, use_container_width=True, hide_index=True)
                st.caption("These four regimes and their proportions were "
                          "chosen by a human, not measured from data. That's "
                          "the point of including this arm — it's the "
                          "assumption the other two are tested against.")

            elif mode == "learned":
                st.markdown(
                    f'<div class="card plain">This looked at <b>'
                    f'{tr["n_real_ratios"]:,} real month-to-month changes</b> '
                    f'in your own sales history. Every consecutive pair of '
                    f'months for every product already <i>is</i> a real '
                    f'scenario — if something sold 100 then 137, the world '
                    f'produced a 1.37× month. It fit a statistical model to '
                    f'all of those, and now generates new ones in the same '
                    f'shape.<br><br>It tested 1–6 underlying patterns and let '
                    f'the data pick: <b>{tr["chosen_components"]} distinct '
                    f'demand patterns</b> fit best.</div>',
                    unsafe_allow_html=True)
                comp = pd.DataFrame(tr["components"])
                comp["share"] = (comp["share"] * 100).round(1)
                comp.columns = ["typical multiplier", "share of scenarios (%)"]
                comp["typical multiplier"] = comp["typical multiplier"].round(3)
                st.dataframe(comp, use_container_width=True, hide_index=True)

                real = np.array(tr["real_ratio_sample"])
                cmp_df = pd.DataFrame({
                    "percentile": ["5th", "25th", "50th (middle)", "75th", "95th"],
                    "real history": np.round(np.quantile(real, [.05,.25,.5,.75,.95]), 2),
                    "this generator": np.round(np.quantile(mult, [.05,.25,.5,.75,.95]), 2),
                })
                st.write("**Does what it invents actually look like reality?**")
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)
                st.caption("If these two columns track each other closely, the "
                          "generator has genuinely learned the shape of real "
                          "demand changes rather than guessing.")

            else:  # llm
                st.caption(f"Model: `{tr['model']}`" +
                          (" (chat-tuned)" if tr.get("chat_template") else
                           " (raw base model)"))
                for i, s in enumerate(tr["samples"]):
                    with st.expander(f"What it actually wrote — example {i+1}",
                                    expanded=(i == 0)):
                        st.code(s["generated_text"] or "(empty)", language=None)
                        st.write(f"Numbers found: {s['parsed']}")
                        st.write(f"Numbers that passed the plausibility "
                                f"check: {s['kept']}")
                st.markdown(
                    f'<div class="card plain"><b>{tr["n_kept"]} of '
                    f'{tr["n_raw_parsed"]}</b> numbers it produced were '
                    f'plausible enough to use (between 0.2× and 3×). The rest '
                    f'were discarded — a language model asked for numbers in '
                    f'prose will happily return a year or a page number, so '
                    f'nothing it says is trusted without checking.</div>',
                    unsafe_allow_html=True)

    modes = [m for m in modes if traces.get(m) is not None]
    if not modes:
        st.error("Every scenario-writer failed — nothing left to compare.")
        st.stop()

    # -- Step 4 ------------------------------------------------------------
    st.subheader("Step 4 — The actual contest")
    st.caption("Each scenario-writer's inventions get added to the same real "
              "training data, the same model is retrained, and it's judged on "
              "the same hidden months. Identical in every way except where the "
              "invented scenarios came from.")

    prog = st.progress(0.0)
    log = st.container()
    rows = []
    aug_preds = {}
    total = len(modes) * n_seeds
    done = 0
    for mode in modes:
        for seed in range(n_seeds):
            lg = log.status(f"{MODE_LABELS[mode]} — run {seed+1}", expanded=False)
            mult = get_multipliers(200, mode=mode, seed=seed, train=train_raw,
                                   model_name=llm_model)
            tr_aug = augment_training_frame(train_raw, FEATURES, mult,
                                            frac=aug_frac, seed=seed)
            lg.write(f"Training on {len(tr_aug):,} examples "
                    f"({len(train_raw):,} real + "
                    f"{len(tr_aug)-len(train_raw):,} invented)...")
            pred = fit_predict(tr_aug, test_raw, seed)
            s = score(y_true, pred)
            aug_preds[(mode, seed)] = pred
            delta = s["MAE"] - base_scores[seed]["MAE"]
            lg.write(f"Average miss: **{s['MAE']:.3f}** vs "
                    f"**{base_scores[seed]['MAE']:.3f}** without any invented "
                    f"data — {'better' if delta < 0 else 'worse'} by "
                    f"{abs(delta):.3f}.")
            lg.update(label=f"{MODE_LABELS[mode]} — run {seed+1}: "
                           f"{s['MAE']:.3f} ({'better' if delta<0 else 'worse'})",
                     state="complete")
            rows.append({"mode": mode, "label": MODE_LABELS[mode], "seed": seed,
                        "MAE": s["MAE"], "baseline_MAE": base_scores[seed]["MAE"],
                        "delta": delta})
            done += 1
            prog.progress(done / total)

    df = pd.DataFrame(rows)

    # -- Step 5 ------------------------------------------------------------
    st.subheader("Step 5 — Results")
    chart = df.pivot_table(index="seed", columns="label", values="MAE")
    chart["No invented data"] = [base_scores[s]["MAE"] for s in range(n_seeds)]
    st.line_chart(chart, height=300)
    st.caption("Lower is better. Each point is one complete retraining. "
              "If the lines cross and tangle, the methods are not meaningfully "
              "different from each other.")

    summary_rows = []
    for mode in modes:
        d = df[df["mode"] == mode]
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d.baseline_MAE, d.MAE).pvalue)
        except Exception:
            p = None
        summary_rows.append({
            "Scenario writer": MODE_LABELS[mode],
            "Average miss": round(d.MAE.mean(), 3),
            "vs. no invented data": round(d.delta.mean(), 3),
            "Beat baseline in": f"{int((d.delta<0).sum())} / {len(d)} runs",
            "p-value": round(p, 3) if p is not None else "—",
        })
    summary = pd.DataFrame(summary_rows).sort_values("Average miss")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    best = summary.iloc[0]
    winner_beats_baseline = best["Average miss"] < round(base_mae, 3)
    any_significant = any(isinstance(r["p-value"], float) and r["p-value"] < 0.05
                          for r in summary_rows)

    if winner_beats_baseline and any_significant:
        verdict = (f"**{best['Scenario writer']}** produced the best forecasts "
                  f"({best['Average miss']} average miss vs {base_mae:.3f} with "
                  f"no invented data), and the difference is statistically "
                  f"solid.")
    elif winner_beats_baseline:
        verdict = (f"**{best['Scenario writer']}** came out best "
                  f"({best['Average miss']} vs {base_mae:.3f} with no invented "
                  f"data), but the gap is small enough that it could still be "
                  f"chance. More repeat runs would settle it.")
    else:
        verdict = (f"**No scenario-writer beat simply not inventing anything.** "
                  f"The best of them ({best['Scenario writer']}, "
                  f"{best['Average miss']}) still didn't improve on the "
                  f"{base_mae:.3f} average miss from real data alone. That's a "
                  f"real answer to the question this page asks — the popular "
                  f"idea did not hold up here.")
    st.markdown(f'<div class="verdict-box">{verdict}</div>',
               unsafe_allow_html=True)

    # -- Step 6 ------------------------------------------------------------
    st.subheader("Step 6 — What the forecasts actually look like")
    best_mode = summary.iloc[0]["Scenario writer"]
    best_key = [m for m in modes if MODE_LABELS[m] == best_mode][0]
    chosen = (aug_preds[(best_key, 0)] if winner_beats_baseline
             else base_preds[0])
    trend = test_raw.copy()
    trend["prediction"] = chosen
    trend["month"] = (trend["YEAR"].astype(str) + "-" +
                      trend["MONTH"].astype(str).str.zfill(2))
    monthly = (trend.groupby("month")
              .agg(**{"what actually sold": ("target", "sum"),
                      "what the model predicted": ("prediction", "sum")})
              .sort_index())
    st.line_chart(monthly, height=300)

    err = ((monthly["what the model predicted"] - monthly["what actually sold"]).abs()
           / monthly["what actually sold"].clip(lower=1))
    st.markdown(
        f'<div class="card plain">Totalling every product into one '
        f'demand-per-month number, the forecast was off by about '
        f'<b>{err.mean()*100:.1f}%</b> per month. This looks much better than '
        f'the per-product "average miss" above, and both are true: individual '
        f'products are hard to call exactly, but their errors partly cancel '
        f'out in aggregate — which is the number that actually matters for '
        f'planning total warehouse volume.</div>',
        unsafe_allow_html=True)

    # -- Step 7 ------------------------------------------------------------
    st.subheader("Step 7 — Turning forecasts into an order sheet")
    st.caption("A fixed business rule applied on top of the forecast — not "
              "something the model learned. Quantities are whole units.")
    view = st.radio("Show", ["By category", "Top 15 individual products"],
                    horizontal=True)
    n_p = test_raw["period_idx"].nunique()

    if view == "By category":
        agg = (trend.groupby("ITEM TYPE")
              .agg(total=("prediction", "sum"), spread=("prediction", "std"))
              .reset_index())
        agg["per_month"] = agg["total"] / n_p
        out = decide(agg["per_month"], agg["spread"].fillna(0))
        out.insert(0, "category", agg["ITEM TYPE"])
    else:
        agg = (trend.groupby(["ITEM CODE", "ITEM TYPE"])
              .agg(total=("prediction", "sum"), spread=("prediction", "std"))
              .reset_index())
        agg["per_month"] = agg["total"] / n_p
        agg = agg.sort_values("per_month", ascending=False).head(15)
        out = decide(agg["per_month"], agg["spread"].fillna(0))
        out.insert(0, "category", agg["ITEM TYPE"].values)
        out.insert(0, "product", agg["ITEM CODE"].values)

    out = out.rename(columns={"forecast": "forecast / month",
                              "order_quantity": "order this many",
                              "safety_buffer": "keep spare",
                              "tier": "moves how fast", "policy": "suggested policy"})
    st.dataframe(out.sort_values("forecast / month", ascending=False),
                use_container_width=True, hide_index=True)

elif not run_clicked:
    st.info("Choose your settings in the sidebar, then press "
           "**▶ Run the comparison**. Everything runs live — nothing here is "
           "pre-computed.")