# Odense AEE / URC re-analysis (v1)

**Folder:** `odense_osm_walking_accessibility_outputs_1`  
**Date:** 2026-07-29 (demography v2 + map style revised 2026-07-31)

## What changed vs the previous folder

| Component | v1 approach |
|-----------|-------------|
| Walking / cycling / PT accessibility | Same OSM method; **carried forward** from prior GPKGs (method unchanged). Maps regenerated with standard basemap style. |
| Green-area share | Rebuilt from **GeoDanmark** `official_landcover_natural.gpkg` (not OSM). |
| Environmental quality / burden | Rebuilt from **official** vectors + heat/noise/air rasters. |
| Demographics | **v2 dasymetric:** StatBank parish (SOGN) × BBR `BYG_BOLIG_` weights, soft-constrained to DST 1 km density classes, scaled to **FOLK1A ≈ 213,431**. Age 0–14 / 65+ from POSTNR1×BBR. Migration-origin fields excluded. |

## Honest limits

- Hex population is a **dasymetric estimate** (parish/postcode counts + building ancillary), not paid DST 100 m person counts.
- DST extranet ground truth is density *classes*, not exact headcounts.
- Equity clustering remains cautious: density used as morphology only in GMM.

## Map style (all geographic figures)

- Titles **left-aligned**
- OSM basemap **only** when the filename contains `osm` (e.g. `*_map_osm.png`)
- Non-`osm` filenames: plain white background, no OSM attribution
- Hex α ≈ 0.58 with OSM / ≈ 0.92 without; north arrow (upper right)  
- Titles without OSM secondary lines  

```bash
python regenerate_all_maps_styled.py
```

## Run order

```bash
cd odense_osm_walking_accessibility_outputs_1
../.venv/bin/python run_pipeline_v1.py
../.venv/bin/python regenerate_all_maps_styled.py
```
