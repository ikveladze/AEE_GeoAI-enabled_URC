# Code index — what to run

All scripts live in `odense_osm_walking_accessibility_outputs_1/`.

## Recommended reproduction path

```bash
cd odense_osm_walking_accessibility_outputs_1
python run_pipeline_v1.py
```

This executes, in order:

1. `00_build_demographic_source_grid_v1.py` — density source grid  
2. `01_build_official_green_and_env_v1.py` — official green + env composites  
3. `demographic_indicator_index_odense.py` — hex demography  
4. `aee_pca_umap_feature_space_odense.py` — AEE feature space  
5. `geoai_functional_urc_typology_odense.py` — GMM typology  
6. `urc_benchmarking_uncertainty_odense.py` — benchmarking / uncertainty  
7. `urc_conventional_vs_aee_mismatch_map_odense.py` — mismatch maps  
8. `aee_summary_statistics_table_odense.py` — summary tables  
9. `generate_aee_method_workflow_chart.py` — workflow figure  

Then optionally:

```bash
python regenerate_all_manuscript_maps.py
```

## Optional OSM re-download (not required to inspect deposited results)

- `walkability_index_revised.py`
- `bikeability_index_odense.py`
- `public_transport_stop_accessibility_index_odense.py`

## Legacy (do not use for v1 official-env claims)

- `green_area_share_index_odense.py`
- `environmental_index_odense.py`
- `multisource_environmental_index_odense.py`

See Excel sheet `Code_map` in `Appendix_Table_A1_Data_Sources.xlsx` for full input/output paths.
