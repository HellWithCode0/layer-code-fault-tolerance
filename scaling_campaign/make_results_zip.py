#!/usr/bin/env python3
"""Pack results/ plus environment information into results_bundle.zip (send this back)."""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    res = ROOT / "results"
    env = dict(python=sys.version, platform=platform.platform(), cpus=os.cpu_count(),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    for m in ("numpy", "scipy", "stim", "sinter", "ldpc", "ortools", "matplotlib"):
        try:
            env[m] = importlib.import_module(m).__version__
        except Exception as e:
            env[m] = f"unavailable ({e.__class__.__name__})"
    out = ROOT / "results_bundle.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("environment.json", json.dumps(env, indent=2))
        for p in sorted(res.rglob("*")) if res.exists() else []:
            if p.is_file():
                z.write(p, Path("results") / p.relative_to(res))
                n += 1
        for extra in ("run.log",):
            if (ROOT / extra).exists():
                z.write(ROOT / extra, extra)
                n += 1
    print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
