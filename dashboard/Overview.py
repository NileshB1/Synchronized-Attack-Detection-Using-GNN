"""Overview page: the data, the method and the main findings.

    streamlit run dashboard/Overview.py
"""

import charts
import streamlit as st
from common import (
    AQUA,
    BLUE,
    NAVY,
    ORANGE,
    finding,
    header,
    kpi,
    name,
    p_text,
    section,
    setup,
    show,
)

data = setup("Overview")
final = data.results.set_index("model").loc[data.final]
counts = data.class_counts.set_index("part")

header("Syncronized Attack Detection Using GNN",
       "Dual-stream graph neural network on VM backup telemetry &nbsp;|&nbsp; "
       "5-fold cross-validation &nbsp;|&nbsp; synthetic episodes")

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Episodes", f"{int(counts.loc['all', 'total']):,}",
    f"{int(counts.loc['all', 'coincidental']):,} coincidental, "
    f"{int(counts.loc['all', 'coordinated']):,} synchronized", NAVY)


kpi(k2, "Proposed GNN test F1", f"{final['f1']:.3f}",
    f"Precision {final['precision']:.3f}, recall {final['recall']:.3f}", BLUE)
kpi(k3, "False alarm rate", f"{final['fp_rate']:.1%}",
    f"{int(final['fp'])} false alarms on the test part", ORANGE)
kpi(k4, "Test ROC-AUC", f"{final['roc_auc']:.3f}", "Ranking quality, 1.0 is perfect", AQUA)

section("The data and the method ",
        "Each episode is a group of flagged backup events. The task: is it one synchronized "
        "attack or unrelated incidents at the same time?")
left, right = st.columns([1, 1])
with left:
    show(charts.class_split(data.class_counts))
with right:
    st.markdown("""
1. **Split.** Episodes are split 70 / 20 / 10 into training, validation and test, keeping
   the same class mix in every part. The test part is scored once, at the end.
2. **Cross-validation.** The training part is cut into 5 folds. Every model trains on four
   folds and is scored on the fifth, five times over.

3. **Proposed model.** A temporal stream (GRU over the events in time order) and a spatial
   stream (graph attention over VMs, networks and storage) are joined by a learned gate.
   Each step also knows whether it moved along a shared network or storage pool, and a short
   episode summary is added before the decision.

""")

section("Key findings", "Computed from the result files, p-values from a paired t-test on the "
        "fold F1 scores. Below 0.05 means the difference is unlikely to be chance.")

sig = data.significance

sig = sig[~sig["model"].str.startswith(data.reference + " + ")]
kinds = data.test.set_index("model")["kind"]
order = sorted(sig.itertuples(), key=lambda r: (kinds.get(r.model) == "gnn", r.model))
cols = st.columns(2)
for i, row in enumerate(order):

    p = row.t_test_p_f1
    if p < 0.05:
        verdict = ("the proposed GNN is significantly better" if row.ours_f1 > row.theirs_f1
                   else "this model is significantly better")
    else:
        verdict = "the difference is not statistically significant"
    with cols[i % 2]:
        finding(f"<b>{name(data.reference)} vs {name(row.model)}</b><br>"
                f"Mean CV F1 {row.ours_f1:.3f} vs {row.theirs_f1:.3f}, p = {p_text(p)}: "
                f"{verdict}.")

st.caption("All episodes come from the synthetic generator, so the numbers compare methods on "
           "this data and do not measure performance on real attack campaigns.")
