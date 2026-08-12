# V1 re-analysis status — updated 2026-07-31 (demography v2)

## Demography (v2 — preferred)

**Primary method:** Parish (StatBank SOGN) × BBR residential floor area (`BYG_BOLIG_`) dasymetric mapping, soft-constrained to DST Kvadratnet 1 km density *classes*, scaled to StatBank FOLK1A.

**Control total:** FOLK1A 2026K2 ≈ **213,431** (hex sum after clip ≈ **213,380**)  
**Age structure:** POSTNR1 age bands × BBR weights, scaled to Basisdata KOMM_TOT_21 shares  
**Ground truth check:** DST extranet 1 km density classes (305 cells) — preferred surface class agreement ≈ **69%**, Spearman ≈ **0.94**

**Source GPKG:** `Map Layers/Demographic/dasymetric_population_hex500_v2.gpkg`  
**Hex output:** `odense_demographic_indicator_outputs/odense_demographic_indicators_500m.gpkg`  
**Methods report:** `odense_demographic_dasymetric_v2_outputs/`

### Used
- StatBank SOGN1 / SOGN10 parish population + POSTNR1 age
- Basisdata KOMM_TOT_21 age–sex structure
- BBR 2020 residential `BYG_BOLIG_` as ancillary weights
- DST Kvadratnet extranet density classes (validation + soft constraint)
- DAWA/DAGI parish & postcode geometries

### Not used for hex population
- `demographic_grid.gpkg` — invalid inflated counts
- OSM place population points
- `nonwestern` / `ikkevestlig` / `turkey` — migration subset (study design)
- OMRP age–sex micro-areas — no spatial crosswalk in folder

### Legacy reference (not preferred)
- `00_build_demographic_source_grid_v1.py` / `dst_kvadratnet_population.gpkg` — class-midpoint estimate scaled to KOMM_TOT_21 = 205,987

## Downstream (re-run completed 2026-07-31 after demography v2)
- AEE feature space / PCA–UMAP — **re-run OK**
- GeoAI functional URC typology — **re-run OK**; BIC-selected **k = 5** (was 7 under previous density surface)
- URC benchmarking / uncertainty — **re-run OK**
- Conventional vs AEE mismatch — **re-run OK**
- AEE summary statistics — **re-run OK**
- All manuscript maps via `regenerate_all_maps_styled.py` — **re-run OK**

Accessibility, green-area, and official environmental layers were **not rebuilt** (no demographic inputs); their maps were restyled only.
