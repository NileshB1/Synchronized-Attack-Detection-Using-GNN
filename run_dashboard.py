"""Results dashboard for the cross-validation study.

    python run_dashboard.py

Opens in the browser. Needs the files written by run_crossval.py.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    app = ROOT / "dashboard" / "Overview.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))
