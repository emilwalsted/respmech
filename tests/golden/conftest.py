# Ensure the golden-test helpers (make_golden.py) are importable regardless of
# the directory pytest is invoked from.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
