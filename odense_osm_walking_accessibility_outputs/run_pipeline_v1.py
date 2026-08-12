#!/usr/bin/env python3
"""Run the v1 re-analysis pipeline in order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Prefer package-root .venv, then analysis-folder .venv, then current interpreter
_candidates = [
    SCRIPT_DIR.parent / ".venv" / "bin" / "python",
    SCRIPT_DIR / ".venv" / "bin" / "python",
    Path(sys.executable),
]
PY = next((c for c in _candidates if c.exists()), Path(sys.executable))

STEPS = [
    # Demography v2: StatBank parish × BBR bolig dasymetric + DST class soft-constraint
    "00_dasymetric_demography_v2.py",
    "01_build_official_green_and_env_v1.py",
    "demographic_indicator_index_odense.py",
    "aee_pca_umap_feature_space_odense.py",
    "geoai_functional_urc_typology_odense.py",
    "urc_benchmarking_uncertainty_odense.py",
    "urc_conventional_vs_aee_mismatch_map_odense.py",
    "aee_summary_statistics_table_odense.py",
    "generate_aee_method_workflow_chart.py",
    "regenerate_all_maps_styled.py",
]


def main() -> None:
    for step in STEPS:
        path = SCRIPT_DIR / step
        print("\n" + "=" * 70)
        print(f"RUNNING {step}")
        print("=" * 70)
        r = subprocess.run([str(PY), str(path)], cwd=str(SCRIPT_DIR))
        if r.returncode != 0:
            print(f"FAILED: {step} (exit {r.returncode})")
            sys.exit(r.returncode)
    print("\nPipeline v1 completed successfully.")


if __name__ == "__main__":
    main()
