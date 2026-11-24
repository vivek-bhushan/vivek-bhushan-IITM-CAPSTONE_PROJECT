"""
run_all.py
==========
One-command end-to-end pipeline for the Intelligent Retail Decision Assistant.

    python src/run_all.py

Runs, in order:
    1. data_prep        (Phase 1)
    2. classical_ml     (Phase 2)
    3. enrich_genai     (Phase 3)
    4. rag_recommendation (Phase 4a)
    5. policy_qa        (Phase 4b)
    6. agent_controller (Phase 5)
    7. evaluation       (Final Phase)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# make `src` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402


def main() -> int:
    config.ensure_dirs()
    steps = [
        ("Phase 1  data_prep", "data_prep"),
        ("Phase 2  classical_ml", "classical_ml"),
        ("Phase 3  enrich_genai", "enrich_genai"),
        ("Phase 4a rag_recommendation", "rag_recommendation"),
        ("Phase 4b policy_qa", "policy_qa"),
        ("Phase 5  agent_controller", "agent_controller"),
        ("Final    evaluation", "evaluation"),
    ]

    results = {}
    for label, module_name in steps:
        t0 = time.time()
        print(f"\n{'='*70}\n>>> {label}\n{'='*70}")
        module = __import__(module_name)
        out = module.run()
        dt = time.time() - t0
        results[module_name] = {"seconds": round(dt, 2)}
        print(f"<<< {label} done in {dt:.1f}s")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
