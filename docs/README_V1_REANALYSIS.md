# Odense AEE / URC re-analysis (v1)

**Folder:** `odense_osm_walking_accessibility_outputs_1`  
**Date:** 2026-07-29  

## What changed vs the previous folder

| Component | v1 approach |
|-----------|-------------|
| Walking / cycling / PT accessibility | Same OSM method; **carried forward** from prior GPKGs (method unchanged by today’s data). Maps regenerated here. |
| Green-area share | Rebuilt from **GeoDanmark** `official_landcover_natural.gpkg` (not OSM). |
| Environmental quality / burden | Rebuilt from **official** vectors + heat/noise/air rasters (OSM optional fallback omitted in v1 builder). |
| Demographics | 1 km density-class grid + non-Western counts, scaled to **KOMM_TOT 2021 = 205,987**. Not full DST Kvadratnet age/sex. |

## Honest limits

- Hex population is a **density-class–derived estimate**, not Statistics Denmark person counts.
- Age/sex exist at **municipality** level (`Map Layers/Demographic/odense_basisdata_o/`); not yet on hex.
- Equity clustering remains cautious: density used as morphology; full Track A equity still pending DST Kvadratnet.

## Run order

```bash
cd odense_osm_walking_accessibility_outputs_1
../.venv/bin/python 00_build_demographic_source_grid_v1.py
../.venv/bin/python 01_build_official_green_and_env_v1.py
../.venv/bin/python demographic_indicator_index_odense.py
../.venv/bin/python aee_pca_umap_feature_space_odense.py
../.venv/bin/python geoai_functional_urc_typology_odense.py
../.venv/bin/python urc_benchmarking_uncertainty_odense.py
../.venv/bin/python urc_conventional_vs_aee_mismatch_map_odense.py
../.venv/bin/python aee_summary_statistics_table_odense.py
../.venv/bin/python generate_aee_method_workflow_chart.py
../.venv/bin/python regenerate_all_manuscript_maps.py
```

Or: `../.venv/bin/python run_pipeline_v1.py`
