"""Train the coordination model on the episode graphs and pick the threshold.

python run_training.py
python run_training.py --output-dir runs/seed42
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
    sys.exit(main([*args, "train"]))
