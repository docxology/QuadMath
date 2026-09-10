import os
import sys

# Force headless backend for matplotlib in tests
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Script modules under test (validate_markdown, sympy_formalisms, ...);
# no name collisions with src/ today (src has symbolic.py, not sympy_formalisms.py)
SCRIPTS = os.path.join(ROOT, "quadmath", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
