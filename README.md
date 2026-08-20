# AEE GeoAI-enabled Urban–Rural Continuum (URC)

**Repository target:** https://github.com/ikveladze/AEE_GeoAI-enabled_URC  

**Study:** GeoAI-enabled Accessibility–Environment–Equity (AEE) mapping of the functional urban–rural continuum  
**Case:** Odense Municipality, Denmark  
**Analytical unit:** 500 m hexagons (`n = 1,539`), CRS **EPSG:25832**  
**Package purpose:** enable scientific replication, and transfer of the workflow to other mid-sized municipalities

**Please be aware of the data availability:** Some demographic data have been removed from this GitHub repository due to their sensitive nature and cannot be shared publicly. References to the corresponding field names may remain in parts of the code; however, no sensitive values are included. These field names are retained intentionally to document the types of input data required and to help others reproduce or adapt the analytical workflow using appropriately authorized or equivalent datasets. 


This folder is the GitHub-ready deposit of codes, metadata tables, essential spatial inputs, hex-level outputs, and manuscript figures.

---

## 1. What this package contains

| Path | Contents |
|------|----------|
| `docs/` | Appendix Table A1, variable dictionaries, dataset inventory Excel/CSV, feature-set tables, code index, pipeline notes |
| `Map Layers/` | Municipal boundary, 500 m hex grid, demographic source layers, official environmental composites used in the reported run |
| `external_sources/` | Density-class source and municipal population control totals |
| `odense_osm_walking_accessibility_outputs_1/` | Python pipeline scripts + deposited hex outputs (GPKG/CSV/maps) |
| `figures/` | Figure 2 workflow, Figure 5 PCA scatter, modelling flowcharts |
| `requirements.txt` | Python dependencies |
| `DATA_LICENSING.md` | Attribution and redistribution notes for OSM, GeoDanmark, WDPA, etc. |
| `.gitignore` | Excludes virtual environments, caches, logs |

**Not bundled (too large / re-downloadable / licence-restricted):**
- Full GeoFabrik OSM country extracts
- Raw full GeoDanmark national dumps (analysis-ready composites *are* included)
- Paid Statistics Denmark Kvadratnet person counts
- National DCE air-quality products (interim CAMS/Open-Meteo NO₂ is included as supporting layer)

---

## 2. Scientific logic of the study (why the pipeline is ordered this way)

The study treats the urban–rural continuum as a **functional** Accessibility–Environment–demographic configuration, not as a purely morphological gradient.

1. **Common spatial frame** — all indicators are constructed on the same 500 m hexagon grid so multimodal accessibility, environmental proxies, and demographic exposure are co-located.
2. **Indicator construction (AEE domains)**  
   - **Accessibility (A):** network-based walk / bike / PT *stop-access* potential (not GTFS door-to-door travel).  
   - **Environment (E):** relative amenity–pressure proxies (green/blue/protected vs built-up/road; supporting heat/noise/air).  
   - **Equity (E):** density-/age-related demographic exposure for interpretation; **excluded from GMM clustering**.
3. **Feature screening** — join by `hex_id`, median imputation, drop near-collinear features (`|r| > 0.90`, iterative VIF > 10). Environmental burden is dropped because of near-perfect correlation with environmental quality.
4. **Interpretation before classification** — PCA (and exploratory UMAP) summarise dominant gradients; they do **not** define class labels.
5. **Primary typology** — Gaussian Mixture Model (GMM) on the screened, z-standardised feature matrix; *k* selected by BIC with minimum share and silhouette constraints (**k = 5** components → **four** planning map labels after Queen smoothing).
6. **Robustness** — K-means and single-domain/quintile benchmarks; ARI/NMI agreement; membership uncertainty, boundary heterogeneity, scenario stability.
7. **Reporting** — typology, continuous URC score, uncertainty surfaces, summary tables, workflow figures.

“GeoAI-enabled” here means an **interpretable geospatial ML workflow** (PCA + GMM + benchmarks + uncertainty), not a novel deep-learning algorithm.

---

## 3. Quick start — inspect deposited results (no recompute)

```bash
# Clone or download this package, then:
cd "GitHub ready materials"   # or the repository root after upload

# Open metadata first
open docs/AEE_Dataset_Inventory_Odense.xlsx
open docs/Appendix_Table_A1_Data_Sources.xlsx

# Inspect key hex outputs in QGIS / GeoPandas
# - odense_osm_walking_accessibility_outputs_1/odense_geoai_functional_urc_typology_outputs/
# - odense_osm_walking_accessibility_outputs_1/odense_urc_benchmarking_uncertainty_outputs/
# - odense_osm_walking_accessibility_outputs_1/odense_aee_feature_space_outputs/
```

---

## 4. Quick start — reproduce the analysis pipeline

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd odense_osm_walking_accessibility_outputs_1
python run_pipeline_v1.py
```

Optional figure regeneration:

```bash
python generate_figure2_classic_workflow.py
python generate_figure5_pca_scatter_manuscript.py
python regenerate_all_maps_styled.py
```

### Important notes on re-running accessibility
Walking / cycling / PT hex GPKGs are **deposited** from the frozen OSM run. Re-running:
- `walkability_index_revised.py`
- `bikeability_index_odense.py`
- `public_transport_stop_accessibility_index_odense.py`

will call live OSMnx/Overpass and may produce **small differences**. Prefer deposited GPKGs for exact manuscript reproduction.

Relative path expectation: scripts resolve `../Map Layers` from the analysis folder (package layout mirrors the working project).

---

## 5. Pipeline stages and what each script does

| Order | Script | What / why |
|------:|--------|------------|
| 1 | `00_dasymetric_demography_v2.py` | Build dasymetric population/age surfaces (StatBank × BBR; DST class soft-constraint) |
| 2 | `01_build_official_green_and_env_v1.py` | Official green/blue/built-up/road/protected overlays + quality/burden composites |
| 3 | `demographic_indicator_index_odense.py` | Aggregate demography to 500 m hex; descriptive vulnerability (not GMM) |
| 4 | `aee_pca_umap_feature_space_odense.py` | Merge AEE indicators; PCA/UMAP feature space |
| 5 | `geoai_functional_urc_typology_odense.py` | Screen features; GMM typology; map labels; URC score |
| 6 | `urc_benchmarking_uncertainty_odense.py` | K-means / domain benchmarks; uncertainty & agreement diagnostics |
| 7 | `urc_conventional_vs_aee_mismatch_map_odense.py` | Conventional vs AEE mismatch visualisation |
| 8 | `aee_summary_statistics_table_odense.py` | Manuscript tables / correlations |
| 9 | `generate_aee_method_workflow_chart.py` | Method workflow chart |
| 10 | `regenerate_all_maps_styled.py` | Refresh map PNGs from GPKGs |

**Accessibility builders (optional recompute):**  
`walkability_index_revised.py`, `bikeability_index_odense.py`, `public_transport_stop_accessibility_index_odense.py`

**Supporting utilities:** `map_style_utils.py`, figure generators (`generate_figure2_classic_workflow.py`, `generate_figure5_pca_scatter_manuscript.py`)

Full input/output mapping: `docs/Appendix_Table_A1_Data_Sources.xlsx` sheet **Code_map** and `docs/CODE_INDEX.md`.

---

## 6. Key analytical parameters (Odense implementation)

### Spatial frame
- Hex size: **500 m**
- Cells: **1,539**
- CRS: **EPSG:25832**
- Municipal area (validation): ~**304.37 km²**

### Accessibility
| Mode | Speed | Decay β | Cut-offs | Meaning |
|------|-------|---------|----------|---------|
| Walk | 1.34 m/s | 700 m | 15 min | Service access on walk network |
| Bike | 15 km/h | 1800 m | 5 / 15 min | Service access on bike network |
| PT | 1.34 m/s (walk to stop) | 1000 m | 10 / 20 min | **Stop-access potential** (not GTFS) |

Service weights (walk/bike): food_retail 0.25; health 0.20; education_childcare 0.15; food_drink 0.15; civic_daily_services 0.15; recreation 0.10.  
Scores min–max scaled **0–100** within the municipality.

### Environment
- Official GeoDanmark green/blue/built-up/roads; WDPA ∪ LBST §3 protected
- Major-road exposure: **50 m** buffer of major classes
- Supporting: summer LST, state-road noise Lden, interim NO₂
- Quality/burden composites scaled **0–100** (relative, not regulatory exposure)

### GMM typology features (8)
`walk_access_15m`, `bike_access_15m`, `pt_stop_access_20m`, `green_share_pct`, `env_quality`, `builtup_pct`, `road_buf_pct`, `pop_density_km2`

- z-standardised; no extra GMM feature weights  
- `env_burden` dropped (|r| ≈ 0.961 with `env_quality`)  
- Demographic vulnerability **excluded** from GMM  
- Selected model: **k = 5** (BIC + share/silhouette constraints) → **4** smoothed map labels

See also: `docs/AEE_Clustering_Feature_Sets.csv`.

---

## 7. Metadata tables (reviewer / replication documentation)

Open these first when auditing or transferring the study:

| File | Use |
|------|-----|
| `docs/AEE_Dataset_Inventory_Odense.xlsx` | Layer-by-layer inventory: provider, date, processing, derived indicator, analytical role |
| `docs/Appendix_Table_A1_Data_Sources.xlsx` | Multi-sheet Appendix A1 + weights + code map |
| `docs/Appendix_Table_A1_variable_dictionary_v1.csv` | Full variable dictionary (filters, β, aggregation, polarity, GMM use) |
| `docs/Appendix_Table_A1_Complete_Variable_Metadata_Publication.xlsx` | Publication-oriented wide metadata |
| `docs/AEE_Geospatial_Data_and_Indicator_Groups.xlsx` | Grouping of geospatial inputs and indicators |
| `docs/Table_Data_Sources_Updated.md` | Short Table 1 companion for the manuscript |

---

## 8. How to transfer the study to another city

The **workflow architecture** transfers; Odense-specific class shares, weights, and municipality-relative scores do **not**.

### Minimum requirements for a new case
1. A municipal (or study-area) boundary in a projected metric CRS.  
2. A regular analysis grid (500 m hexagons recommended; document MAUP if changed).  
3. Pedestrian/cycle networks + service destinations (OSM/OSMnx is the default path).  
4. PT stop/station locations (or GTFS stops if upgrading beyond stop-access).  
5. Environmental amenity and pressure layers (national land cover / roads / protected areas).  
6. Demographic density (and ideally age/socio-economics) for interpretation.  
7. Local justification of cut-offs, β, service weights, and composite weights.

### Adaptation checklist
1. Replace `Map Layers/Odense_Municipality_1.gpkg` and `Odense-500mHexaCells_1.gpkg` with the new boundary/grid.  
2. Update hard-coded place names / paths in scripts (search for `Odense`, `odense`, EPSG:25832).  
3. Re-run accessibility builders **or** supply equivalent hex indicators with the same field names.  
4. Rebuild environmental composites from local official layers; keep the amenity–pressure logic.  
5. Rebuild demography with local census/register sources; keep density in GMM only if socio-economics are unavailable.  
6. Re-run feature screening (correlation/VIF) — do **not** assume the same features are dropped.  
7. Re-select GMM *k* with BIC + substantive profile checks; re-label classes from local profiles.  
8. Recompute uncertainty and benchmarks before any planning use.  
9. Recalibrate URC score weights if used; treat as interpretive, not a clustering input.  
10. Document every local dataset in a new Appendix A1 / inventory Excel.

### What must be locally recalibrated
- Network speeds, β, and time cut-offs (ideally with local behavioural evidence)  
- Service taxonomy and weights  
- Environmental composite weights and available supporting rasters  
- GMM feature set after collinearity screening  
- Class labels and planning interpretation  
- CRS / grid size (state MAUP limits explicitly)

### Transferability statement (for papers / repositories)
> Workflow transferability concerns analytical structure (common grid → AEE indicators → screening → PCA interpretation → GMM typology → uncertainty/benchmarks). It does not imply portability of Odense class boundaries, municipality-relative scores, or policy prescriptions.

---

## 9. Expected key outputs

After a successful run (or from deposited files):

- `odense_geoai_functional_urc_typology_outputs/` — functional classes, feature matrix, profiles  
- `odense_urc_benchmarking_uncertainty_outputs/` — uncertainty, agreement, benchmark maps  
- `odense_aee_feature_space_outputs/` — merged AEE feature space, PCA/UMAP  
- `odense_aee_summary_statistics_outputs/` — tables for the manuscript  
- `odense_*_accessibility_outputs/` / green / environmental / demographic folders — domain indicators  
- `figures/` — workflow and PCA manuscript figures  

---

## 10. Citation and licence notes

- Cite the *Urban Science* article when published, and this repository commit/DOI when reusing code or deposited hex layers.  
- Follow `DATA_LICENSING.md` for OSM, GeoDanmark/Klimadatastyrelsen, WDPA, LBST, Vejdirektoratet, ESA/Zenodo CLIM4cities, and Statistics Denmark extracts.  
- Do not redistribute paid register products beyond what providers allow.

---

## 11. Contact

Open a GitHub issue for path errors, missing dependencies, or questions about regenerating a specific figure or transferring the workflow to another city.
