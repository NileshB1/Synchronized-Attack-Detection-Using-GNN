"""Compare model variants across generator scenarios and seeds.

    python run_experiments.py
    python run_experiments.py --spec configs/experiments.yaml --out runs/comparison

Uses configs/default.yaml as the base config unless --config is given.
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
    
    command = ["experiment", "--spec", str(ROOT / "configs" / "experiments.yaml")]
    for flag in ("--spec", "--out"):
        if flag in args:
            i = args.index(flag)
            if flag == "--spec":
                command[2] = args[i + 1]
            else:
                command += [flag, args[i + 1]]
            del args[i : i + 2]
    sys.exit(main([*args, *command]))
