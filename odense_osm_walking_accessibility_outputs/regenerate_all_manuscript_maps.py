#!/usr/bin/env python3
"""Regenerate every manuscript map/graph from existing analysis outputs.

Prefer --map-only / GPKG-based plotting over full OSM recompute.
Writes a JSON report of each figure's status.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
VENV_PY = PROJECT_DIR / ".venv" / "bin" / "python"
REPORT_PATH = SCRIPT_DIR / "manuscript_map_regeneration_report.json"

results: list[dict] = []


def record(name: str, script: str, outputs: list[Path], status: str, detail: str = "") -> None:
    entry = {
        "name": name,
        "script": script,
        "status": status,
        "detail": detail,
        "outputs": [
            {
                "path": str(p),
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
                if p.exists()
                else None,
            }
            for p in outputs
        ],
    }
    results.append(entry)
    print(f"[{status}] {name}: {detail}")
    for o in entry["outputs"]:
        print(f"    -> {o['path']} exists={o['exists']} size={o['size_bytes']}")


def run_script(script_path: Path, args: list[str] | None = None, env_extra: dict | None = None) -> tuple[int, str]:
    cmd = [str(VENV_PY), str(script_path), *(args or [])]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))
    if env_extra:
        env.update(env_extra)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out.strip()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def regen_map_only(label: str, script_name: str, outputs: list[Path]) -> None:
    script = SCRIPT_DIR / script_name
    before = {p: (p.stat().st_mtime if p.exists() else None) for p in outputs}
    code, log = run_script(script, ["--map-only"])
    if code != 0:
        record(label, script_name, outputs, "FAILED", log[-2000:])
        return
    refreshed = any(
        p.exists() and (before[p] is None or p.stat().st_mtime > before[p]) for p in outputs
    )
    status = "OK" if refreshed or all(p.exists() for p in outputs) else "PARTIAL"
    record(label, script_name, outputs, status, "map-only regenerate\n" + log[-1500:])


def regen_multisource_from_gpkg() -> None:
    label = "Multi-source environmental quality / burden maps"
    script_name = "multisource_environmental_index_odense.py"
    outputs = [
        SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_environmental_quality_index_500m_map.png",
        SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_environmental_burden_index_500m_map.png",
        SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_environmental_quality_index_500m_map_osm.png",
    ]
    try:
        mod = load_module(SCRIPT_DIR / script_name, "multisource_env_regen")
        gpkg = mod.output_gpkg
        hex_output = gpd.read_file(gpkg, layer="environmental_index_500m_hexagons").to_crs(mod.target_crs)
        municipality = gpd.read_file(gpkg, layer="odense_municipality_districts").to_crs(mod.target_crs)
        boundary = gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=mod.target_crs)
        mod.plot_index_map(
            hex_output,
            municipality,
            boundary,
            "env_quality_class",
            "Odense Environmental Quality Index by 500 m Cell",
            "Environmental quality",
            mod.quality_colors(),
            mod.output_quality_map,
        )
        mod.plot_index_map(
            hex_output,
            municipality,
            boundary,
            "env_burden_class",
            "Odense Environmental Burden Index by 500 m Cell",
            "Environmental burden",
            mod.burden_colors(),
            mod.output_burden_map,
        )
        mod.plot_osm_basemap_quality_map(hex_output, municipality, boundary)
        record(label, script_name, outputs, "OK", "plotted from existing GPKG")
    except Exception:
        record(label, script_name, outputs, "FAILED", traceback.format_exc()[-2000:])


def regen_demographic_from_gpkg() -> None:
    label = "Demographic vulnerability / density maps"
    script_name = "demographic_indicator_index_odense.py"
    outputs = [
        SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_population_density_500m_map.png",
        SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_age_vulnerability_500m_map.png",
        SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_socioeconomic_vulnerability_500m_map.png",
        SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_vulnerability_index_500m_map.png",
        SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_vulnerability_index_500m_map_osm.png",
    ]
    try:
        mod = load_module(SCRIPT_DIR / script_name, "demographic_regen")
        gpkg = mod.output_gpkg
        hex_output = gpd.read_file(gpkg, layer="demographic_indicators_500m_hexagons").to_crs(mod.target_crs)
        municipality = gpd.read_file(gpkg, layer="odense_municipality_districts").to_crs(mod.target_crs)
        boundary = gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=mod.target_crs)
        mod.plot_class_map(
            hex_output,
            municipality,
            boundary,
            "pop_density_class",
            "Odense Population Density by 500 m Cell",
            "Population density",
            mod.sequential_colors(),
            mod.output_population_map,
        )
        mod.plot_class_map(
            hex_output,
            municipality,
            boundary,
            "age_vulnerability_class",
            "Odense Age Vulnerability by 500 m Cell",
            "Age vulnerability",
            mod.vulnerability_colors(),
            mod.output_age_map,
        )
        mod.plot_class_map(
            hex_output,
            municipality,
            boundary,
            "socioeconomic_vulnerability_class",
            "Odense Socio-economic Vulnerability by 500 m Cell",
            "Socio-economic vulnerability",
            mod.vulnerability_colors(),
            mod.output_socio_map,
        )
        mod.plot_class_map(
            hex_output,
            municipality,
            boundary,
            "demo_vulnerability_class",
            "Demographic vulnerability index",
            "Demographic vulnerability",
            mod.vulnerability_colors(),
            mod.output_vulnerability_map,
        )
        mod.plot_osm_basemap_vulnerability_map(
            hex_output, municipality, boundary, mod.output_vulnerability_map_osm
        )
        record(label, script_name, outputs, "OK", "plotted from existing GPKG")
    except Exception:
        record(label, script_name, outputs, "FAILED", traceback.format_exc()[-2000:])


def regen_full(label: str, script_name: str, outputs: list[Path]) -> None:
    script = SCRIPT_DIR / script_name
    before = {p: (p.stat().st_mtime if p.exists() else None) for p in outputs}
    code, log = run_script(script)
    if code != 0:
        record(label, script_name, outputs, "FAILED", log[-2500:])
        return
    refreshed = any(
        p.exists() and (before[p] is None or p.stat().st_mtime > before[p]) for p in outputs
    )
    status = "OK" if refreshed or all(p.exists() for p in outputs) else "PARTIAL"
    record(label, script_name, outputs, status, "full script run\n" + log[-1500:])


def regen_mismatch_fixed_path() -> None:
    label = "Conventional vs AEE mismatch maps"
    script_name = "urc_conventional_vs_aee_mismatch_map_odense.py"
    out_dir = SCRIPT_DIR / "odense_urc_mismatch_outputs"
    outputs = [
        out_dir / "odense_urc_conventional_vs_aee_mismatch_map.png",
        out_dir / "odense_urc_mismatch_intensity_map.png",
        out_dir / "odense_urc_agreement_map.png",
        out_dir / "odense_conventional_urban_rural_classification_map.png",
        out_dir / "odense_aee_functional_urc_class_map.png",
    ]
    try:
        mod = load_module(SCRIPT_DIR / script_name, "mismatch_regen")
        # Fix broken PROJECT_DIR-relative path (typology GPKG lives under SCRIPT_DIR)
        correct = (
            SCRIPT_DIR
            / "odense_geoai_functional_urc_typology_outputs"
            / "odense_geoai_functional_urc_typology_500m.gpkg"
        )
        mod.INPUT_GPKG = correct
        mod.main()
        record(
            label,
            script_name,
            outputs,
            "OK" if all(p.exists() for p in outputs) else "PARTIAL",
            f"INPUT_GPKG patched to {correct}",
        )
    except Exception:
        record(label, script_name, outputs, "FAILED", traceback.format_exc()[-2500:])


def regen_legacy_pipeline_maps() -> None:
    """Replot legacy outputs_urc_aee maps from saved GPKG (not manuscript primary)."""
    label = "Legacy URC-AEE pipeline maps (outputs_urc_aee)"
    script_name = "urc_aee_mapping_pipeline.py"
    out_dir = PROJECT_DIR / "outputs_urc_aee"
    map_cols = [
        "functional_vulnerability",
        "green_access_mismatch",
        "aee_cluster",
        "cluster_prob",
        "pca1",
    ]
    outputs = [out_dir / f"map_{col}.png" for col in map_cols]
    gpkg = out_dir / "urc_aee_outputs.gpkg"
    if not gpkg.exists():
        record(label, script_name, outputs, "SKIPPED", "legacy GPKG missing")
        return
    try:
        import matplotlib.pyplot as plt

        gdf = gpd.read_file(gpkg)
        plotted = []
        for col in map_cols:
            path = out_dir / f"map_{col}.png"
            if col not in gdf.columns:
                plotted.append(False)
                continue
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf.plot(ax=ax, column=col, legend=True, edgecolor="none", linewidth=0)
            ax.set_title(f"Legacy: {col}")
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plotted.append(True)
        status = "OK" if all(plotted) else ("PARTIAL" if any(plotted) else "FAILED")
        record(label, script_name, outputs, status, f"columns={list(gdf.columns)}")
    except Exception:
        record(label, script_name, outputs, "FAILED", traceback.format_exc()[-2000:])


def main() -> None:
    print("=== Regenerating manuscript maps/graphs ===")
    print(f"Python: {VENV_PY}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    # Stage 1 — map-only from GPKGs
    regen_map_only(
        "Walking accessibility (OSM basemap)",
        "walkability_index_revised.py",
        [SCRIPT_DIR / "odense_walking_accessibility_services_500m_map_osm.png"],
    )
    regen_map_only(
        "Cycling accessibility (5/15 min + OSM)",
        "bikeability_index_odense.py",
        [
            SCRIPT_DIR
            / "odense_osm_cycling_accessibility_outputs"
            / "odense_cycling_accessibility_services_500m_map_5min.png",
            SCRIPT_DIR
            / "odense_osm_cycling_accessibility_outputs"
            / "odense_cycling_accessibility_services_500m_map_15min.png",
            SCRIPT_DIR
            / "odense_osm_cycling_accessibility_outputs"
            / "odense_cycling_accessibility_services_500m_map_15min_osm.png",
        ],
    )
    regen_map_only(
        "PT stop accessibility (10/20 min + OSM)",
        "public_transport_stop_accessibility_index_odense.py",
        [
            SCRIPT_DIR
            / "odense_osm_public_transport_accessibility_outputs"
            / "odense_public_transport_stop_accessibility_500m_map_10min.png",
            SCRIPT_DIR
            / "odense_osm_public_transport_accessibility_outputs"
            / "odense_public_transport_stop_accessibility_500m_map_20min.png",
            SCRIPT_DIR
            / "odense_osm_public_transport_accessibility_outputs"
            / "odense_public_transport_stop_accessibility_500m_map_20min_osm.png",
        ],
    )
    regen_map_only(
        "Green area share (+ OSM)",
        "green_area_share_index_odense (1).py",
        [
            SCRIPT_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m_map.png",
            SCRIPT_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m_map_osm.png",
        ],
    )
    regen_map_only(
        "OSM-only environmental quality / burden",
        "environmental_index_odense.py",
        [
            SCRIPT_DIR
            / "odense_osm_environmental_index_outputs"
            / "odense_environmental_quality_index_500m_map.png",
            SCRIPT_DIR
            / "odense_osm_environmental_index_outputs"
            / "odense_environmental_burden_index_500m_map.png",
            SCRIPT_DIR
            / "odense_osm_environmental_index_outputs"
            / "odense_environmental_quality_index_500m_map_osm.png",
        ],
    )

    regen_multisource_from_gpkg()
    regen_demographic_from_gpkg()

    # Stage 2–4 — recompute + plots from local GPKGs (no OSM)
    regen_full(
        "PCA / UMAP feature-space maps & scatters",
        "aee_pca_umap_feature_space_odense.py",
        [
            SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_pca_scatter.png",
            SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_umap_scatter.png",
            SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_pca_cluster_map.png",
            SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_umap_cluster_map.png",
            SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_feature_cluster_map.png",
        ],
    )
    regen_full(
        "GeoAI functional URC typology + score maps",
        "geoai_functional_urc_typology_odense.py",
        [
            SCRIPT_DIR
            / "odense_geoai_functional_urc_typology_outputs"
            / "odense_geoai_functional_urc_typology_map.png",
            SCRIPT_DIR
            / "odense_geoai_functional_urc_typology_outputs"
            / "odense_functional_urc_score_map.png",
        ],
    )
    regen_full(
        "URC benchmarking & uncertainty maps",
        "urc_benchmarking_uncertainty_odense.py",
        [
            SCRIPT_DIR
            / "odense_urc_benchmarking_uncertainty_outputs"
            / "odense_urc_gmm_membership_uncertainty_map.png",
            SCRIPT_DIR
            / "odense_urc_benchmarking_uncertainty_outputs"
            / "odense_urc_boundary_heterogeneity_map.png",
            SCRIPT_DIR
            / "odense_urc_benchmarking_uncertainty_outputs"
            / "odense_urc_scenario_stability_map.png",
            SCRIPT_DIR
            / "odense_urc_benchmarking_uncertainty_outputs"
            / "odense_urc_gmm_kmeans_agreement_map.png",
            SCRIPT_DIR
            / "odense_urc_benchmarking_uncertainty_outputs"
            / "odense_urc_density_benchmark_map.png",
        ],
    )
    regen_full(
        "AEE method workflow chart",
        "generate_aee_method_workflow_chart.py",
        [
            SCRIPT_DIR / "odense_aee_method_workflow_chart.png",
            SCRIPT_DIR / "odense_aee_method_workflow_chart.pdf",
            SCRIPT_DIR / "odense_aee_method_workflow_chart.svg",
        ],
    )

    regen_mismatch_fixed_path()
    regen_legacy_pipeline_maps()

    # Summary statistics (tables only — note in report)
    code, log = run_script(SCRIPT_DIR / "aee_summary_statistics_table_odense.py")
    table_outs = [
        SCRIPT_DIR / "odense_aee_summary_statistics_outputs" / "odense_aee_summary_statistics.xlsx",
        SCRIPT_DIR / "odense_aee_summary_statistics_outputs" / "odense_aee_indicator_summary.csv",
    ]
    record(
        "AEE summary statistics tables (no maps)",
        "aee_summary_statistics_table_odense.py",
        table_outs,
        "OK" if code == 0 else "FAILED",
        log[-1500:],
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_jobs": len(results),
        "n_ok": sum(1 for r in results if r["status"] == "OK"),
        "n_failed": sum(1 for r in results if r["status"] == "FAILED"),
        "n_partial": sum(1 for r in results if r["status"] == "PARTIAL"),
        "n_skipped": sum(1 for r in results if r["status"] == "SKIPPED"),
        "jobs": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")
    print(
        f"Summary: OK={report['n_ok']} PARTIAL={report['n_partial']} "
        f"FAILED={report['n_failed']} SKIPPED={report['n_skipped']}"
    )


if __name__ == "__main__":
    main()
