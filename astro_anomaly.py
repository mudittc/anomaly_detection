#!/usr/bin/env python3
"""Starter cross-modal astronomical anomaly pipeline.

This program is a candidate-finding aid, not an automated physical-explanation engine.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOG = logging.getLogger("astro-anomaly")

COORD_RA_CANDIDATES = ["RA_ICRS", "RAJ2000", "RAdeg", "RA", "_RAJ2000", "RAJ2000"]
COORD_DEC_CANDIDATES = ["DE_ICRS", "DEJ2000", "DEdeg", "DEC", "DE", "_DEJ2000"]
TIME_CANDIDATES = [
    "obs_date", "ObsDate", "MJD", "mjd", "JD", "jd", "date", "Date",
    "Epoch", "epoch", "ObsTime", "time", "Time"
]
QUALITY_NAME_HINTS = ("qual", "flag", "qflg", "cc_flg", "ph_qual", "err", "error", "e_")

def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    # Fallback to names that look like coordinate columns.
    for c in df.columns:
        cl = str(c).lower().replace("_", "")
        if cl in {"ra", "raj2000", "raicrs", "radeg"} and any(x.lower() in cl for x in ["ra"]):
            return str(c)
    return None

def find_coord_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    ra = find_column(df, COORD_RA_CANDIDATES)
    dec = find_column(df, COORD_DEC_CANDIDATES)
    # VizieR column metadata may include RAJ2000/DEJ2000, but some catalogs use
    # differently named coordinate columns. Do not guess arbitrary numeric fields.
    if ra is None:
        for c in df.columns:
            if str(c).lower() in {"ra", "raj2000", "ra_icrs", "radeg", "_raj2000"}:
                ra = str(c); break
    if dec is None:
        for c in df.columns:
            if str(c).lower() in {"dec", "dej2000", "de_icrs", "dedeg", "_dej2000"}:
                dec = str(c); break
    return ra, dec

def find_time_column(df: pd.DataFrame) -> str | None:
    return find_column(df, TIME_CANDIDATES)

def numeric_feature_columns(df: pd.DataFrame, ra_col: str | None, dec_col: str | None,
                            time_col: str | None) -> list[str]:
    exclude = {x for x in [ra_col, dec_col, time_col] if x}
    cols = []
    for c in df.columns:
        if c in exclude or str(c).startswith("_"):
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        # Avoid ID columns and numeric-coded quality flags in the initial model.
        name = str(c).lower()
        if any(token in name for token in ["source_id", "objid", "designation", "recno", "seq"]):
            continue
        if any(hint in name for hint in QUALITY_NAME_HINTS):
            continue
        if series.notna().sum() >= max(5, int(0.1 * len(df))):
            # Need actual variation; constants don't help anomaly scoring.
            if series.nunique(dropna=True) > 1:
                cols.append(str(c))
    return cols

def fetch_catalog(catalog_id: str, ra_deg: float, dec_deg: float,
                  radius_deg: float, row_limit: int) -> pd.DataFrame:
    """Query a public VizieR catalog within a cone search."""
    center = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    vizier = Vizier(columns=["*"], row_limit=row_limit)
    tables = vizier.query_region(center, radius=radius_deg * u.deg, catalog=catalog_id)
    if not tables:
        LOG.warning("No table returned for VizieR catalog %s", catalog_id)
        return pd.DataFrame()
    # Usually one table per catalog ID; choose the first non-empty table.
    for table in tables:
        df = table.to_pandas()
        if not df.empty:
            return df
    return pd.DataFrame()

def robust_median_scores(df: pd.DataFrame, features: list[str],
                         min_group_size: int, z_threshold: float) -> pd.DataFrame:
    """Median/MAD scores computed within this channel/catalog.

    For each feature: z_robust = (x - median) / (1.4826 * MAD).
    If MAD is zero/undefined, fall back to IQR/1.349. Features with insufficient
    observations or no spread are skipped. This is not a Gaussian probability.
    """
    out = df.copy()
    per_feature = []
    for col in features:
        values = pd.to_numeric(out[col], errors="coerce")
        valid = values.dropna()
        if len(valid) < min_group_size:
            continue
        med = float(valid.median())
        mad = float((valid - med).abs().median())
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            q1, q3 = valid.quantile([0.25, 0.75])
            scale = float((q3 - q1) / 1.349)
        if not np.isfinite(scale) or scale <= 0:
            continue
        score_col = f"rz_{safe_name(col)}"
        out[score_col] = (values - med) / scale
        per_feature.append(score_col)
    if per_feature:
        z = out[per_feature].abs()
        out["robust_max_abs_z"] = z.max(axis=1, skipna=True)
        out["robust_feature_evidence"] = z.apply(
            lambda row: "; ".join(
                f"{c[3:]}={row[c]:.2f}" for c in per_feature
                if pd.notna(row[c]) and row[c] >= z_threshold
            ), axis=1
        )
    else:
        out["robust_max_abs_z"] = np.nan
        out["robust_feature_evidence"] = ""
    return out

def add_iforest_scores(df: pd.DataFrame, features: list[str],
                       random_state: int, n_estimators: int,
                       contamination: float) -> pd.DataFrame:
    out = df.copy()
    if len(features) < 1 or len(out) < 10:
        out["iforest_anomaly_score"] = np.nan
        return out
    X = out[features].apply(pd.to_numeric, errors="coerce")
    # Remove features that are entirely missing.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        out["iforest_anomaly_score"] = np.nan
        return out
    # Median imputation and robust scaling reduce the influence of units/scales.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(),
        IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    model.fit(X)
    # sklearn's lower decision_function means more anomalous; invert for clarity.
    scores = -model.decision_function(X)
    out["iforest_anomaly_score"] = scores
    return out

def make_review_note(row: pd.Series, features: list[str]) -> str:
    """Transparent checklist, not an automated physical verdict."""
    notes = []
    if pd.notna(row.get("robust_max_abs_z")) and row.get("robust_max_abs_z") >= 4:
        notes.append("large deviation from catalog median/MAD baseline")
    if pd.notna(row.get("iforest_anomaly_score")):
        notes.append("unusual multivariate feature combination flagged by Isolation Forest")
    if not notes:
        notes.append("flagged by configured ranking threshold; inspect underlying measurements")
    notes.extend([
        "verify measurement uncertainties and catalog quality flags",
        "check instrument/systematic artifacts and survey selection effects",
        "compare with known astrophysical explanations before calling it unexplained",
    ])
    return "; ".join(notes)

def score_channel(channel: str, raw: pd.DataFrame, cfg: dict,
                  out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if raw.empty:
        return pd.DataFrame(), {"channel": channel, "n_raw": 0, "n_candidates": 0,
                                "status": "no data returned"}
    df = raw.copy()
    df["channel"] = channel
    df["catalog_row_id"] = np.arange(len(df), dtype=int)
    ra_col, dec_col = find_coord_columns(df)
    time_col = find_time_column(df)
    if ra_col:
        df["ra_deg"] = pd.to_numeric(df[ra_col], errors="coerce")
    if dec_col:
        df["dec_deg"] = pd.to_numeric(df[dec_col], errors="coerce")
    if time_col:
        # Numeric MJD/JD values are preserved. String dates are parsed to ISO when possible.
        numeric_time = pd.to_numeric(df[time_col], errors="coerce")
        if numeric_time.notna().sum() > 0:
            df["observation_time_raw"] = numeric_time
            if str(time_col).lower() == "mjd":
                df["observation_time_utc"] = pd.to_datetime(numeric_time, unit="D", origin="1858-11-17", errors="coerce", utc=True)
            elif str(time_col).lower() == "jd":
                df["observation_time_utc"] = pd.to_datetime(numeric_time - 2440587.5, unit="D", errors="coerce", utc=True)
        else:
            df["observation_time_utc"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    features = numeric_feature_columns(df, ra_col, dec_col, time_col)
    if not features:
        LOG.warning("%s: no usable numeric features found", channel)
        df["robust_max_abs_z"] = np.nan
        df["iforest_anomaly_score"] = np.nan
        df["candidate_flag"] = False
        df["review_note"] = "No usable numeric features; inspect catalog schema and configure feature selection."
        return df, {"channel": channel, "n_raw": len(df), "n_candidates": 0,
                    "features": [], "status": "no usable features"}
    mcfg = cfg["model"]
    df = robust_median_scores(df, features, int(mcfg["minimum_group_size"]),
                              float(mcfg["robust_z_threshold"]))
    df = add_iforest_scores(df, features, int(mcfg["random_state"]),
                            int(mcfg["n_estimators"]), float(mcfg["contamination"]))
    fraction = float(mcfg["candidate_top_fraction"])
    if len(df) and df["iforest_anomaly_score"].notna().any():
        cutoff = df["iforest_anomaly_score"].quantile(max(0, min(1, 1 - fraction)))
        iforest_flag = df["iforest_anomaly_score"] >= cutoff
    else:
        iforest_flag = pd.Series(False, index=df.index)
    robust_flag = df["robust_max_abs_z"] >= float(mcfg["robust_z_threshold"])
    df["candidate_flag"] = (iforest_flag | robust_flag).fillna(False)
    df["review_note"] = df.apply(lambda row: make_review_note(row, features) if row["candidate_flag"] else "", axis=1)
    df["features_used"] = ", ".join(features)
    path = out_dir / f"scored_{safe_name(channel)}.csv"
    df.to_csv(path, index=False)
    summary = {
        "channel": channel,
        "n_raw": int(len(df)),
        "n_candidates": int(df["candidate_flag"].sum()),
        "features": features,
        "ra_column_detected": ra_col,
        "dec_column_detected": dec_col,
        "time_column_detected": time_col,
        "status": "scored",
        "output": str(path),
    }
    LOG.info("%s: %d rows, %d candidates, %d features",
             channel, len(df), int(df["candidate_flag"].sum()), len(features))
    return df, summary

def angular_separation_arcsec(ra1, dec1, ra2, dec2) -> float:
    c1 = SkyCoord(ra=float(ra1) * u.deg, dec=float(dec1) * u.deg, frame="icrs")
    c2 = SkyCoord(ra=float(ra2) * u.deg, dec=float(dec2) * u.deg, frame="icrs")
    return float(c1.separation(c2).arcsec)

def crossmatch_candidates(candidates: pd.DataFrame, max_sep_arcsec: float,
                          simultaneity_window_days: float) -> pd.DataFrame:
    """Cross-match flagged rows across channels. Missing timestamps remain unknown."""
    columns = [
        "channel_a", "row_a", "channel_b", "row_b", "separation_arcsec",
        "time_a_utc", "time_b_utc", "time_delta_days", "temporal_relation",
        "evidence_note"
    ]
    if candidates.empty or not {"ra_deg", "dec_deg", "channel"}.issubset(candidates.columns):
        return pd.DataFrame(columns=columns)
    df = candidates.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)
    records = []
    for i in range(len(df)):
        a = df.iloc[i]
        for j in range(i + 1, len(df)):
            b = df.iloc[j]
            if a["channel"] == b["channel"]:
                continue
            try:
                sep = angular_separation_arcsec(a["ra_deg"], a["dec_deg"], b["ra_deg"], b["dec_deg"])
            except Exception:
                continue
            if sep > max_sep_arcsec:
                continue
            ta = pd.to_datetime(a.get("observation_time_utc", pd.NaT), errors="coerce", utc=True)
            tb = pd.to_datetime(b.get("observation_time_utc", pd.NaT), errors="coerce", utc=True)
            if pd.notna(ta) and pd.notna(tb):
                delta = abs((ta - tb).total_seconds()) / 86400.0
                temporal = "overlap_or_near" if delta <= simultaneity_window_days else "different_epochs"
            else:
                delta = np.nan
                temporal = "unknown"
            records.append({
                "channel_a": a["channel"], "row_a": a.get("catalog_row_id", i),
                "channel_b": b["channel"], "row_b": b.get("catalog_row_id", j),
                "separation_arcsec": sep,
                "time_a_utc": ta.isoformat() if pd.notna(ta) else "",
                "time_b_utc": tb.isoformat() if pd.notna(tb) else "",
                "time_delta_days": delta,
                "temporal_relation": temporal,
                "evidence_note": (
                    "spatial association only; inspect positional uncertainties, source density, "
                    "catalog dependence, exposure windows, and known explanations"
                )
            })
    return pd.DataFrame(records, columns=columns)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        LOG.error("Config file not found: %s", cfg_path)
        return 2
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = cfg["query"]
    ra, dec, radius = float(q["ra_deg"]), float(q["dec_deg"]), float(q["radius_deg"])
    fetch_cfg = cfg["fetch"]
    row_limit = int(fetch_cfg.get("row_limit", 5000))
    summaries = []
    all_scored = []

    for channel, entry in fetch_cfg["catalogs"].items():
        if not entry.get("enabled", True):
            LOG.info("Skipping disabled catalog %s", channel)
            continue
        catalog_id = entry["catalog"]
        try:
            LOG.info("Fetching %s (%s) around RA=%.5f Dec=%.5f radius=%.4f deg",
                     channel, catalog_id, ra, dec, radius)
            raw = fetch_catalog(catalog_id, ra, dec, radius, row_limit)
            if not raw.empty:
                raw.to_csv(out_dir / f"raw_{safe_name(channel)}.csv", index=False)
            scored, summary = score_channel(channel, raw, cfg, out_dir)
            summaries.append(summary)
            if not scored.empty and "candidate_flag" in scored:
                all_scored.append(scored[scored["candidate_flag"]].copy())
        except Exception as exc:
            LOG.exception("Failed catalog %s: %s", channel, exc)
            summaries.append({"channel": channel, "catalog": catalog_id,
                              "status": "error", "error": str(exc),
                              "n_raw": 0, "n_candidates": 0})

    if all_scored:
        candidate_df = pd.concat(all_scored, ignore_index=True, sort=False)
    else:
        candidate_df = pd.DataFrame()
    candidate_df.to_csv(out_dir / "all_candidates.csv", index=False)

    mc = cfg.get("matching", {})
    pairs = crossmatch_candidates(
        candidate_df,
        float(mc.get("max_separation_arcsec", 5.0)),
        float(mc.get("simultaneity_window_days", 1.0)),
    )
    pairs.to_csv(out_dir / "crossmatch_pairs.csv", index=False)
    summary_obj = {
        "query": q,
        "catalog_summaries": summaries,
        "candidate_rows_total": int(len(candidate_df)),
        "crossmatch_pairs_total": int(len(pairs)),
        "warning": (
            "Anomaly scores are not probabilities or physical explanations. "
            "Review data quality, known phenomena, source associations, and survey systematics."
        ),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary_obj, indent=2, default=str), encoding="utf-8")
    LOG.info("Finished. Outputs are in %s", out_dir.resolve())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
