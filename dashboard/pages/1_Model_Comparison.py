"""Model comparison: the results table, scores per model, curves and significance tests."""

import charts
import pandas as pd
import streamlit as st
from common import BLUE, group, header, name, p_text, section, setup, show

PALETTE = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7a5cc7", "#52514e", "#3fb5d1", "#a0522d"]

data = setup("Model comparison")
header("Model comparison",
       "Every model on the same 70/20/10 split and the same 5 folds")

table = data.results.copy()
table["group"] = [group(m, k, data.reference) for m, k in zip(table["model"], table["kind"],
                                                              strict=True)]

section("Results table", "CV columns are mean over the 5 folds; the rest is the test part, "
        "scored once. The proposed model is highlighted.")
view = pd.DataFrame({
    "Model": [name(m) for m in table["model"]],
    "Balancing": table["balancing"],
    "CV accuracy": [f"{m:.3f} ± {s:.3f}" for m, s in zip(table["cv_accuracy_mean"],
                                                        table["cv_accuracy_std"], strict=True)],
    "Test accuracy": table["test_accuracy"].round(3),
    "Precision": table["precision"].round(3),
    "Recall": table["recall"].round(3),
    "F1": table["f1"].round(3),
    "ROC-AUC": table["roc_auc"].round(3),
    "False alarms": table["fp"].astype(int),
    "p-value vs proposed": [p_text(p) for p in table["p_value_vs_ours"]],
})
proposed = table["group"].eq("proposed").to_numpy()
st.dataframe(
    view.style.apply(lambda r: ["background-color: #EAF1FB" if proposed[r.name] else ""] * len(r),
                     axis=1).format(precision=3),
    hide_index=True, width="stretch",
)

section("Scores per model", "Blue is the proposed GNN, green the other GNN designs, orange the "
        "baselines.")
left, right = st.columns(2)
with left:
    show(charts.model_bars(table, "f1", "Test F1 (higher is better)"))
with right:
    show(charts.model_bars(table, "cv_accuracy_mean",
                                "CV accuracy, mean ± std over folds", error="cv_accuracy_std"))
left, right = st.columns(2)
with left:
    show(charts.model_bars(table, "fp", "False alarms on the test part (lower is better)",
                                lower_is_better=True))
with right:
    show(charts.model_bars(table, "roc_auc", "Test ROC-AUC (higher is better)"))

section("Precision-recall and ROC curves", "Pick the models to draw. Curves use the scores on "
        "the test part.")
scored = [m for m in data.test["model"] if f"score_{m}" in data.predictions
          and not m.endswith(" + default")]
default = [m for m in [data.final, "random_forest", "logistic_regression", "continuum_style",
                       "rule_based"] if m in scored]
picked = st.multiselect("Models", scored, default=default, format_func=name)
if picked:
    others = iter(PALETTE)
    colours = {m: BLUE if m == data.final or m == data.reference else next(others)
               for m in picked}
    show(charts.pr_and_roc(data.predictions, picked, colours))

section("Confusion matrices", "Test part. A false alarm is a coincidental episode flagged as "
        "an attack.")
models = list(data.test["model"])


def test_row(model):
    return data.test[data.test["model"] == model].iloc[0]


left, right = st.columns(2)
with left:
    show(charts.confusion(test_row(data.final)))
with right:
    choices = [m for m in models if m != data.final]
    start = choices.index("random_forest") if "random_forest" in choices else 0
    other = st.selectbox("Compare with", choices, index=start, format_func=name)
    show(charts.confusion(test_row(other)))

section("Significance tests", "Is the difference to the proposed GNN real or could it be "
        "chance? Below 0.05 means real.")
sig = data.significance
st.dataframe(pd.DataFrame({
    "Model": [name(m) for m in sig["model"]],
    "Proposed CV F1": sig["ours_f1"].round(3),
    "Its CV F1": sig["theirs_f1"].round(3),
    "t-test p": [p_text(p) for p in sig["t_test_p_f1"]],
    "Wilcoxon p": [p_text(p) for p in sig["wilcoxon_p_f1"]],
    "McNemar p (test)": [p_text(p) for p in sig["mcnemar_p"]],
    "Real difference?": ["yes" if p < 0.05 else "no, could be chance"
                         for p in sig["t_test_p_f1"]],
}), hide_index=True, width="stretch")
st.caption("With 5 folds the smallest possible two-sided Wilcoxon p-value is 0.0625, so the "
           "t-test is the main test here.")
