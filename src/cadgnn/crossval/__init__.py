"""Cross-validation study: class balance, folds, false positives, ablation and tests."""

from cadgnn.crossval.runner import CrossvalResult, run_crossval, stratified_folds
from cadgnn.crossval.spec import CrossvalSpec, load_crossval_spec, parse_crossval_spec
from cadgnn.crossval.stats import mcnemar, paired_t_p, wilcoxon_p
from cadgnn.crossval.tables import results_frame, write_reports

__all__ = [
    "CrossvalResult", "run_crossval", "stratified_folds",
    "CrossvalSpec", "load_crossval_spec", "parse_crossval_spec",
    "mcnemar", "paired_t_p", "wilcoxon_p",
    "results_frame", "write_reports",
]  # fmt: skip
