"""Cross-validation study: class balance, folds, false positives, ablation and baselines.

    python run_crossval.py
    python run_crossval.py --spec configs/crossval.yaml --out runs/crossval

Needs episode_graphs.pt, so run the generator and graph builder first. Writes
splits/, crossval/, reports/ and figures/ into the output directory.
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
    command = ["crossval", "--spec", str(ROOT / "configs" / "crossval.yaml")]
    for flag in ("--spec", "--out"):
        if flag in args:
            i = args.index(flag)
            if flag == "--spec":
                command[2] = args[i + 1]
            else:
                command += [flag, args[i + 1]]
            del args[i : i + 2]
    sys.exit(main([*args, *command]))
