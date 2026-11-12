"""Fit the rule-based and classic ML baselines on the same split as the GNN.

python run_baselines.py
python run_baselines.py --output-dir runs/seed42

Needs episode_graphs.pt, so run the graph builder first.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cadgnn.cli import main


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--config" not in args and "-c" not in args:
        args = ["--config", str(ROOT / "configs" / "default.yaml"), *args]
    sys.exit(main([*args, "baselines"]))
