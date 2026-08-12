# V1 re-analysis status — revised 2026-07-30

## Demography policy
- **Total population / density only**
- Excluded from all analysis: `nonwestern_origin`, `nonwestern_share`, `turkey_origin_pop`, `ikkevestlig_pop`
- Source: 1 km density-class estimate scaled to KOMM_TOT_21 = 205,987

## Maps (standard cartography)
- OSM basemap under semi-transparent hexagons (α ≈ 0.55)
- North arrow (upper right)
- Graphical scale bar + RF (lower right)
- `© OpenStreetMap contributors` (lower right)
- No “OSM” secondary titles

## Re-run
```bash
cd odense_osm_walking_accessibility_outputs_1
../.venv/bin/python run_pipeline_v1.py
../.venv/bin/python regenerate_standard_maps_v2.py
```
