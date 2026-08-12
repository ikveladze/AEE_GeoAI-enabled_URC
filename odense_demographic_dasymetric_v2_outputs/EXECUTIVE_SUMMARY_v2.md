# Odense demography v2 — executive summary

## Preferred surface (used for maps)
**M3c_folk**: StatBank parish (SOGN) population × BBR residential housing floor area (`BYG_BOLIG_`) dasymetric redistribution, soft-constrained to DST Kvadratnet 1 km density *classes*, then scaled to StatBank FOLK1A total (**213,431**).

Age 0–14 / 65+: POSTNR1 age bands × same BBR weights, scaled to Basisdata KOMM_TOT_21 age structure adjusted to FOLK totals.

## DST ground-truth check (305 × 1 km cells)
| Method | Class agreement | Spearman vs class mid |
|--------|-----------------|------------------------|
| M1 areal SOGN | ~38% | 0.65 |
| M2 binary BBR | ~60% | 0.93 |
| M3 floor-area BBR | ~48–50% | 0.91 |
| **M3c_folk (preferred)** | **~69%** | **0.94** |
| M4 DST midpoint areal | ~35% | 0.93 (partly circular) |
| M5 hybrid DST×BBR | ~40% | 0.94 (uses DST as source) |
| M6 postcode×BBR | ~25–31% | 0.84 |

## Control totals
- KOMM_TOT_21 (Basisdata): 205,987
- SOGN10 2022: 205,954
- SOGN1 2026: 213,067
- FOLK1A 2026K2: 213,431
- Preferred hex sum (clipped n=1,539): ≈213,380

## Key outputs
- `Map Layers/Demographic/dasymetric_population_hex500_v2.gpkg`
- `odense_demographic_dasymetric_v2_outputs/` (metrics, maps, publications)
- Updated `odense_demographic_indicator_outputs/` maps & GPKG
