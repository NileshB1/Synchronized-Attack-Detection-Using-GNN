"""Page setup, styling and data loading shared by every dashboard page."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output"

NAVY = "#1B3A6B"
BLUE = "#2a78d6"
AQUA = "#1baf7a"

ORANGE = "#eb6834"
GREY = "#9a9893"

NAMES = {
    "gated": "Proposed GNN (gated)",
    "concat": "GNN, concat fusion",
    "temporal_only": "GNN, temporal stream only",
    "spatial_only": "GNN, spatial stream only",

    "no_step_links": "GNN without step links",
    "no_summary": "GNN without episode summary",

    "continuum_style": "CONTINUUM-style GAT-GRU",
    "rule_based": "Rule-based (SIEM style)",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}

STRATEGIES = {
    "default": "Threshold from validation F1",
    "threshold_0.5": "Fixed 0.5 threshold",
    "threshold_precision": "Precision-floor threshold",

    "platt": "Platt calibration",


    "isotonic": "Isotonic calibration",
    "cost_sensitive": "Cost-sensitive loss",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #EAF1FB 0%, #CFE0F5 100%);
    padding: 20px 30px; border-radius: 12px; margin-bottom: 22px;
    border-left: 5px solid #1B3A6B;
}
.main-header h1 { color: #0b0b0b; font-size: 30px; font-weight: 700; margin: 0; }
.main-header p { color: #52514e; font-size: 13px; margin: 4px 0 0 0; }

.kpi-box {
    background: white; border-radius: 10px; padding: 16px 20px;

    border-left: 4px solid #1B3A6B; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.kpi-label {
    font-size: 11px; font-weight: 600; color: #7F8C8D;


    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;
}
.kpi-value {
    font-size: 30px; font-weight: 700; color: #1B3A6B;

    font-family: 'IBM Plex Mono', monospace; line-height: 1;
}
.kpi-sub { font-size: 11px; color: #95A5A6; margin-top: 4px; }

.section-header {
    background: #F8F9FA; border-radius: 8px; padding: 12px 16px;

    margin: 22px 0 12px 0; border-left: 4px solid #1B3A6B;
}
.section-header h3 { margin: 0; font-size: 16px; font-weight: 600; color: #1B3A6B; }
.section-header p { margin: 4px 0 0 0; font-size: 12px; color: #7F8C8D; }

.finding-card {
    background: #EBF5FB; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    border-left: 3px solid #2E86C1; font-size: 13px; color: #1B2631; line-height: 1.5;
}

[data-testid="stSidebar"] { background: #EAF1FB; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""


@dataclass
class Results:
    folder: Path
    test: pd.DataFrame
    folds: pd.DataFrame
    significance: pd.DataFrame

    predictions: pd.DataFrame
    class_counts: pd.DataFrame
    histories: list
    results: pd.DataFrame

    @property
    def reference(self) -> str:
        return self.results["model"].iloc[0]

    @property
    def final(self) -> str:
        """The proposed model with the chosen false positive fix."""
        fixed = [m for m in self.results["model"] if m.startswith(self.reference + " + ")]
        return fixed[0] if fixed else self.reference


def name(model: str) -> str:
    base, _, fix = model.partition(" + ")
    label = NAMES.get(base, base.replace("_", " "))
    return f"{label} + {STRATEGIES.get(fix, fix).lower()}" if fix else label


def group(model: str, kind: str, reference: str) -> str:
    if model == reference or model.startswith(reference + " + "):
        return "proposed"
    return "gnn" if kind == "gnn" else "baseline"


COLOURS = {"proposed": BLUE, "gnn": AQUA, "baseline": ORANGE}
GROUP_NAMES = {"proposed": "Proposed GNN", "gnn": "Other GNN designs", "baseline": "Baselines"}


@st.cache_data(show_spinner=False)
def load(folder: str) -> Results | None:
    out = Path(folder)
    cv = out / "crossval"
    needed = ["test.csv", "folds.csv", "significance.csv", "predictions.csv", "class_counts.csv"]
    table = out / "reports" / "results_table.csv"
    if not all((cv / f).is_file() for f in needed) or not table.is_file():
        return None
    histories = json.loads((cv / "histories.json").read_text(encoding="utf-8")) if (
        cv / "histories.json").is_file() else []
    results = pd.read_csv(out / "reports" / "results_table.csv")
    test = pd.read_csv(cv / "test.csv")
    kinds = test.set_index("model")["kind"]

    results["kind"] = results["model"].map(kinds).fillna("baseline")

    return Results(
        out, test, pd.read_csv(cv / "folds.csv"), pd.read_csv(cv / "significance.csv"),
        pd.read_csv(cv / "predictions.csv"), pd.read_csv(cv / "class_counts.csv"),
        histories, results,
    )


def setup(title: str) -> Results:
    """Page config, style and the sidebar. Stops the page if there are no results."""
    st.set_page_config(page_title=f"{title} | Syncronized Attack Detection", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Syncronized Attack\nDetection Using GNN")
        st.markdown("MSc Thesis, National College of Ireland")

        st.markdown("---")

        folder = st.text_input("Output folder", str(DEFAULT_OUTPUT))

        st.caption("Written by run_crossval.py")
        st.markdown("---")
        st.markdown("**Pages**")
        st.markdown("- Overview: data and key findings\n- Model comparison\n"
                    "- Ablation and false alarms")
        if st.button("Reload results"):
            st.cache_data.clear()
            st.rerun()
    data = load(folder)
    if data is None:
        st.error(f"No cross-validation results in {folder}. Run python run_crossval.py first.")
        st.stop()
    return data


def header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="main-header"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def section(title: str, text: str = "") -> None:
    st.markdown(f'<div class="section-header"><h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True)


def kpi(column, label: str, value: str, sub: str, colour: str = NAVY) -> None:
    column.markdown(
        f'<div class="kpi-box" style="border-left-color:{colour};">'

        f'<div class="kpi-label">{label}</div>'


        f'<div class="kpi-value" style="color:{colour};">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def show(fig) -> None:
    """Draw a matplotlib figure and free it, so reruns do not pile up memory."""
    st.pyplot(fig)
    plt.close(fig)


def finding(text: str) -> None:
    st.markdown(f'<div class="finding-card">{text}</div>', unsafe_allow_html=True)


def p_text(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
