# Astronomical Cross-Modal Anomaly Detection (starter)

This is a runnable baseline for a research prototype. It fetches public catalog records around a sky position, scores anomalies within each catalog/channel, records explainable reasons, cross-matches candidates spatially, and writes CSV outputs.

## What it does
- Uses Astroquery/VizieR to query public catalogs: Gaia DR3 (optical/astrometry), 2MASS and AllWISE (infrared), NVSS (radio), and an optional ROSAT X-ray catalog.
- Computes robust median/MAD deviations within each catalog's available numeric features.
- Runs Isolation Forest as a second detector on numeric features.
- Keeps a continuous score and a ranked candidate flag rather than treating scores as probabilities.
- Cross-matches detections spatially and labels temporal relationships only when usable timestamps exist.
- Records a transparent review note. This is not an automated physics proof: it flags data-quality/interpretation checks for human review.

## Important scientific limitations
1. VizieR catalogs are not equivalent to raw observations. They may summarize measurements from different epochs and may not expose observation timestamps. The pipeline therefore does NOT infer simultaneity if dates are absent.
2. Catalog records from different surveys may share source associations or systematic errors. Spatial coincidence alone is not independent corroboration.
3. A median baseline is used as requested, but the median is computed per catalog and feature, with MAD as a robust scale estimate. If a feature has too few valid measurements, its deviation is not used.
4. Isolation Forest finds unusual feature combinations, not “alien life” or unexplained physics. A high score can indicate artifacts or rare but known objects.
5. The sample coordinates in config.json are an example sky position, not a recommended science target. Change RA/Dec and radius before running.
6. Catalog identifiers/columns can change; inspect the raw CSVs and logs if a remote catalog query fails.

## Install
Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
python astro_anomaly.py --config config.json --out output
```

The script writes:
- `raw_<channel>.csv`: downloaded catalog rows
- `scored_<channel>.csv`: rows with robust deviations, Isolation Forest score, and candidate flag
- `all_candidates.csv`: all candidate rows with channel tags
- `crossmatch_pairs.csv`: spatially associated candidates from different channels
- `run_summary.json`: counts and query settings

## Change the sky region
Edit `config.json`:
- `ra_deg`: right ascension in decimal degrees
- `dec_deg`: declination in decimal degrees
- `radius_deg`: cone-search radius in degrees

## Adding another catalog
Add a catalog entry to `fetch.catalogs` in config.json. Use an Astroquery/VizieR catalog identifier and make sure the catalog has sky-coordinate columns. If the catalog's coordinate columns are not automatically recognized by Astropy, update `find_coord_columns()` in `astro_anomaly.py`.

## Interpretation
- `robust_max_abs_z`: maximum absolute robust deviation among usable features.
- `iforest_anomaly_score`: higher means more unusual after sign inversion of sklearn's decision score.
- `candidate_flag`: either the robust baseline or Isolation Forest puts the record in the selected tail.
- `review_note`: prompts for manual inspection; it is not a physical classification.
- `temporal_relation`: `overlap_or_near` only if both records have valid time fields; otherwise `unknown`.

Before making scientific claims, validate against known sources, inspect artifacts, test thresholds, account for selection effects, and compare with simple baselines.
