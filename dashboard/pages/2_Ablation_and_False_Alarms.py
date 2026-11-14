"""Ablation, false alarm fixes, the kinds of episode that cause errors and training curves."""

import charts
import streamlit as st
from common import BLUE, STRATEGIES, finding, header, name, section, setup, show

PALETTE = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7"]

data = setup("Ablation and false alarms")
header("Ablation and false alarms",
       "What each part of the proposed model adds, and how false alarms were reduced")

section("Ablation", "Each variant is the proposed model with one part changed or switched off, "
        "trained on the same folds.")
kinds = data.test.set_index("model")["kind"]
sig = data.significance
# The CONTINUUM-style model is another architecture, not an ablation of this one.
variants = sig[sig["model"].map(kinds).eq("gnn") & ~sig["model"].str.startswith(data.reference)
               & sig["model"].ne("continuum_style")]
left, right = st.columns([3, 2])
with left:
    show(charts.ablation(variants, float(sig["ours_f1"].iloc[0])))
with right:
    for row in variants.sort_values("t_test_p_f1").itertuples():
        change = row.ours_f1 - row.theirs_f1
        word = "drops" if change > 0 else "rises"
        sure = "a significant change" if row.t_test_p_f1 < 0.05 else "could be chance"
        finding(f"<b>{name(row.model)}</b>: mean CV F1 {word} by {abs(change):.3f} "
                f"({sure}).")

section("False alarm fixes", "Ways to cut false alarms, compared on the folds. The chosen fix has "
        "the lowest false positive rate without losing more than a little recall.")
fp = data.folds[data.folds["study"] == "fp"]
means = fp.groupby("strategy")[["false_positive_rate", "recall", "f1"]].mean().reset_index()
chosen = data.final.partition(" + ")[2] or "default"
left, right = st.columns([3, 2])
with left:
    show(charts.strategies(means, chosen, STRATEGIES))
with right:
    before = data.test.set_index("model").loc[f"{data.reference} + default"]
    after = data.test.set_index("model").loc[data.final]
    st.markdown("**Test part, before and after the fix**")
    st.markdown(f"""
| | Before | After |
| --- | --- | --- |
| False alarms | {int(before['fp'])} | {int(after['fp'])} |
| False positive rate | {before['false_positive_rate']:.1%} | {after['false_positive_rate']:.1%} |
| Recall | {before['recall']:.3f} | {after['recall']:.3f} |
| F1 | {before['f1']:.3f} | {after['f1']:.3f} |
""")
    st.caption(f"Chosen fix: {STRATEGIES.get(chosen, chosen).lower()}.")

section("Which episodes cause the errors", "Coincidental patterns: burst (short window), "
        "cluster (one network), lookalike (one network, short window), scattered. Coordinated: "
        "fast and stealth (slow) campaigns.")
scored = [m for m in data.test["model"] if f"pred_{m}" in data.predictions
          and not m.endswith(" + default")]
default = [m for m in [data.final, "random_forest", "continuum_style", "rule_based"]
           if m in scored]
picked = st.multiselect("Models", scored, default=default, format_func=name, key="patterns")
if picked:
    others = iter(PALETTE)
    colours = {m: BLUE if m == data.final or m == data.reference else next(others)
               for m in picked}
    show(charts.patterns(data.predictions, picked, colours))

section("Training curves of the proposed model", "Loss and accuracy per epoch on one "
        "cross-validation fold. The dashed line is the epoch kept by early stopping.")
if data.histories:
    fold = st.radio("Fold", list(range(1, len(data.histories) + 1)), horizontal=True)
    show(charts.training_curves(data.histories[fold - 1]))
else:
    st.info("histories.json not found in the output folder.")
