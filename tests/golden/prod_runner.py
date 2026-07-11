#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subprocess entry point: run the UNMODIFIED respmech.py on one prepared settings
dict (paths already rewritten to local, single-file mask) and exit non-zero on
failure.

Run in isolation so a crash / hard failure in the original code (e.g. the latent
EMG-plot bug) is captured as a non-zero exit instead of taking down the
orchestrator. Reads a JSON settings file path from argv[1].

    python prod_runner.py <settings.json>
"""
import sys
import json
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESPMECH_PY = os.path.join(REPO_ROOT, "respmech.py")


def main():
    settings_path = sys.argv[1]
    with open(settings_path) as f:
        settings = json.load(f)

    spec = importlib.util.spec_from_file_location("respmech", RESPMECH_PY)
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)

    # respmech.analyse installs a global excepthook that swallows tracebacks and
    # returns; restore the default so any failure is a real non-zero exit here.
    rm.analyse(settings)
    sys.excepthook = sys.__excepthook__
    print("PROD_RUNNER_OK")


if __name__ == "__main__":
    main()
