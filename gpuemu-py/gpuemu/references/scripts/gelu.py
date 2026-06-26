#!/usr/bin/env python3
"""Registered gpuemu reference for `gelu` (fp64 oracle).

Set `reference` in gpuemu.toml to this file's path. The script bootstraps
its own import path, so it works whether or not gpuemu is pip-installed in
the daemon's interpreter.
"""
import os
import sys

# Add the package root (…/gpuemu/references/scripts/X.py -> root) to sys.path
_root = os.path.abspath(__file__)
for _ in range(4):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from gpuemu.references.ops import run, gelu  # noqa: E402

if __name__ == "__main__":
    run(gelu)
