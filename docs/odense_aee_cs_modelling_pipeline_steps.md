# Detailed modelling pipeline steps (as implemented)

Parameters verified against analysis scripts (2026-08-01).

## A. Spatial frame and data ingestion
1. Build/clip 500 m hex grid H (n=1,539; EPSG:25832; municipal area ≈304.37 km²).
2a. Ingest OSM via OSMnx/Overpass: walk/bike graphs, POIs, PT stops (no GTFS).
2b. Ingest GeoDanmark (Datafordeler), BBR (`BYG_BOLIG_`), StatBank SOGN/POSTNR/FOLK1A, DST 1 km density classes.

## B. Accessibility features (network algorithms)
3. **Walking 15 min:** multi-source Dijkstra; v=1.34 m/s; T=15 → d_max≈1,206 m; a=exp(-d/700); 6 weighted service categories; min-max → [0,100]; quintiles among accessible cells.
4. **Cycling 5 & 15 min:** Dijkstra; v=15 km/h (=4.167 m/s); d_max≈1,250 / 3,750 m; a=exp(-d/1800); → `bike_access_5m`, `bike_access_15m`.
5. **PT stop 10 & 20 min:** walk to OSM stops; v=1.34 m/s; d_max≈804 / 1,608 m; a=exp(-d/1000); index = 0.5·all-stops + 0.5·category-weighted → `pt_stop_access_10m/_20m`.

## C. Environment and equity
6. Environmental shares/composites: green, built-up, blue, eco; major-road buffer **50 m** (official v1); quality/burden scores 0–100; optional LST/noise/NO₂ supporting layers.
7. Dasymetric demography v2: SOGN×BBR → FOLK1A (≈213,431); DST class soft-constraint; ages POSTNR1×BBR; density ρ; DST validation (class agreement ≈69%, Spearman ≈0.94).
8. Vulnerability indices (descriptive only): age, socio, composite demo vulnerability; **excluded from GMM**; migration/origin excluded.

## D. Feature-space QC
9. Join by `hex_id` → X_raw.
10. Clustering candidates (priority): `walk_access_15m`, `bike_access_15m`, `pt_stop_access_20m`, `green_share_pct`, `env_quality`, `env_burden`, `builtup_pct`, `road_buf_pct`, `pop_density_km2`.
11. Drop |r|>0.90; iterative VIF>10 cull; median impute; z-score StandardScaler → Z∈R^{n×p}.
12. PCA (≤6 comps) + UMAP (n_neighbors=20, min_dist=0.10) exploratory only.

## E. Models
13. GMM: full covariance; n_init=10; seed=42; K∈[2,10]; select K* by BIC (+ min share≥0.03, silhouette preference≥0.10). **This run K*=5.**
14. K-means benchmark at K*; n_init=50; agreement diagnostics.

## F. Decoding / scoring
15. γ_ik posteriors; ŷ=argmax γ; u=1−max γ.
16. Queen majority smoothing → 4 interpreted functional classes (~15.4% / 30.8% / 5.4% / 48.4%).
17. Continuous URC score: 0.35·low access + 0.25·low density + 0.15·low built-up + 0.25·green → [0,100].

## G. Evaluation
18. Uncertainty maps (membership, boundary heterogeneity, scenario stability).
19. Single-domain / quintile benchmarks; ARI/NMI; mismatch mapping.
20. GMM–K-means algorithmic agreement map.

## H. Artefacts
21. GPKGs, maps, profiles, BIC plot, corr/VIF tables, validation reports (`run_pipeline_v1.py`).
