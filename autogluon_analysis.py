#!/usr/bin/env python3
"""
autogluon_analysis.py — THE DRUM PROTOCOLS
──────────────────────────────────────────────
AutoGluon-driven protocol recommender. Runs locally — not in GitHub Actions.

Trains on:
  - survey_responses_short  (outcome_score, src, data_type)
  - survey_responses_full   (listener context features)
  - youtube_daily           (completion signal)
  - protocols.json          (design features: delta, dur, startHz, endHz, move)

Outputs:
  - autogluon.json          → repo root, fetched by adviser.html independently
  - autogluon_context.json  → repo root, injected into MOE pipeline prompt
                               via {{INJECT:AUTOGLUON_CONTEXT}}
  - autogluon_model/        → saved AutoGluon predictor folder (local only, gitignored)
  - autogluon_model_runs    → Supabase table, model versioning / drift tracking

Intentionally decoupled from moe_sphere_pipeline.py and data.json.
Each script owns its output. adviser.html fetches both files independently.

Usage:
  py -3.11 autogluon_analysis.py               # train + score + write both JSON files
  py -3.11 autogluon_analysis.py --real-only   # filter to data_type='real' only
  py -3.11 autogluon_analysis.py --dry-run     # skip all writes, print summary
  py -3.11 autogluon_analysis.py --no-train    # load saved model, skip retraining
  py -3.11 autogluon_analysis.py --time-limit 300  # AutoGluon time budget in seconds

Requirements:
  pip install autogluon.tabular[lightgbm,catboost,xgboost] supabase python-dotenv

Environment (.env):
  SUPABASE_URL         — Supabase project URL
  SUPABASE_SERVICE_KEY — service_role key (bypasses RLS)
"""

import os
import sys
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from datetime import timezone
from dotenv import load_dotenv
from supabase import create_client

# ── Configuration ─────────────────────────────────────────────────────────────

PROTOCOLS_PATH     = "protocols.json"
MODEL_PATH         = "autogluon_model"        # outcome_score model
MODEL_PATH_CHANGE  = "autogluon_model_change"  # change_rating model
MIN_TRAIN_ROWS_CHANGE = 10                     # lower threshold — 1-min survey has fewer rows
OUTPUT_PATH        = "autogluon.json"
CONTEXT_PATH       = "autogluon_context.json"  # MOE pipeline injection source
MIN_TRAIN_ROWS     = 20                        # below this, use fallback mean-based ranking
DEFAULT_TIME_LIMIT = None                      # No limit — run until all models complete

CAT_COLS = [
    "series", "entry", "move", "zone",
    "listener_type", "activity", "listen_method", "settle_time",
]
NUM_COLS = [
    "level", "dur", "startHz", "endHz", "delta", "hz_abs",
    "avg_view_percentage", "avg_view_duration", "adviser_driven",
]


# ── Protocol feature loader ───────────────────────────────────────────────────

def load_protocol_features(path=PROTOCOLS_PATH):
    """
    Parse protocols.json and return:
      proto_df  — DataFrame of 30 protocols with design features
      raw_json  — full parsed dict (needed for routing / questions)
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for p in raw["protocols"]:
        rows.append({
            "protocol_code": p["code"],
            "series":        p["series"],
            "entry":         p["entry"],
            "move":          p["move"],
            "level":         p["level"],
            "dur":           p["dur"],
            "startHz":       p["startHz"],
            "endHz":         p["endHz"],
            "delta":         p["startHz"] - p["endHz"],   # signed: + = descent, - = ascent
            "hz_abs":        abs(p["startHz"] - p["endHz"]),
        })

    # Zone lookup from routing table
    zone_map = {}
    for route in raw["routing"]:
        for code in route["codes"]:
            zone_map[code] = route["key"]

    df = pd.DataFrame(rows)
    df["zone"] = df["protocol_code"].map(zone_map)

    return df, raw


# ── Survey + YouTube data loader ──────────────────────────────────────────────

def load_training_data(supabase, real_only=False):
    """Pull survey_responses_short, survey_responses_full, youtube_daily."""
    q = supabase.table("survey_responses_short").select(
        "id, protocol_code, volume_code, outcome_score, src, data_type, submitted_at"
    )
    if real_only:
        q = q.eq("data_type", "real")
    short = pd.DataFrame(q.execute().data)

    q2 = supabase.table("survey_responses_full").select(
        "id, protocol_code, listener_type, activity, listen_method, "
        "settle_time, change_rating, music_opinion, rhythm_opinion, data_type"
    )
    if real_only:
        q2 = q2.eq("data_type", "real")
    full = pd.DataFrame(q2.execute().data)

    yt_raw = supabase.table("youtube_daily").select(
        "protocol_code, report_date, avg_view_percentage, avg_view_duration"
    ).execute().data
    yt = pd.DataFrame(yt_raw)
    if not yt.empty:
        yt = (
            yt.sort_values("report_date")
            .groupby("protocol_code")
            .last()
            .reset_index()
        )[["protocol_code", "avg_view_percentage", "avg_view_duration"]]

    return short, full, yt


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_features(short, full, yt, proto_df):
    """Join all sources into a training DataFrame ready for AutoGluon."""
    if short.empty:
        return pd.DataFrame()

    df = short.copy()
    df["protocol_code"] = df["protocol_code"].str.strip()
    df["adviser_driven"] = (df["src"] == "adviser").astype(int)

    if not full.empty:
        df = df.merge(
            full[["id", "listener_type", "activity", "listen_method", "settle_time"]],
            on="id", how="left",
        )
    else:
        for col in ["listener_type", "activity", "listen_method", "settle_time"]:
            df[col] = None

    df = df.merge(proto_df, on="protocol_code", how="left")

    if not yt.empty:
        df = df.merge(yt, on="protocol_code", how="left")
    else:
        df["avg_view_percentage"] = None
        df["avg_view_duration"]   = None

    # AutoGluon handles categoricals natively — just fill nulls
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    available = [c for c in CAT_COLS + NUM_COLS if c in df.columns] + ["outcome_score"]
    return df[available].copy()


# ── Feature matrix builder — change_rating model ────────────────────────────

# Ordinal mapping for change_rating text responses → 0-10 numeric scale
CHANGE_RATING_MAP = {
    "Significant improvement":      10.0,
    "Noticeable improvement":        8.0,
    "Moderate Improvement":          6.0,
    "Slight improvement":            4.0,
    "No effect":                     2.0,
    "It actually made things worse": 0.0,
}

def build_features_change(full, yt, proto_df):
    """
    Build training DataFrame for the change_rating model.
    Uses survey_responses_full as the primary source — these responses have
    both listener context features AND change_rating (before/after delta).
    change_rating is a categorical Tally response mapped to 0-10 numeric scale:
        Significant improvement      → 10
        Noticeable improvement       →  8
        Moderate Improvement         →  6
        Slight improvement           →  4
        No effect                    →  2
        It actually made things worse →  0
    Joined with protocol design features and YouTube engagement.
    Only rows with a mappable change_rating are included.
    """
    if full.empty:
        return pd.DataFrame()

    df = full.copy()
    df["protocol_code"] = df["protocol_code"].str.strip()

    # Map text responses to numeric — drop unmappable rows
    df["change_rating"] = df["change_rating"].map(CHANGE_RATING_MAP)
    df = df[df["change_rating"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["adviser_driven"] = 0  # full survey doesn't carry src — default to 0

    df = df.merge(proto_df, on="protocol_code", how="left")

    if not yt.empty:
        df = df.merge(yt, on="protocol_code", how="left")
    else:
        df["avg_view_percentage"] = None
        df["avg_view_duration"]   = None

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    available = [c for c in CAT_COLS + NUM_COLS if c in df.columns] + ["change_rating"]
    return df[available].copy()


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(train_df, time_limit=DEFAULT_TIME_LIMIT,
               label="outcome_score", model_path=None):
    """
    Train AutoGluon TabularPredictor on the given label.
    Supports outcome_score (1-second survey) and change_rating (1-minute survey).
    Uses lightgbm, catboost, xgboost, random forest ensemble.
    """
    from autogluon.tabular import TabularPredictor

    path = model_path or MODEL_PATH

    train_df = train_df.copy()
    train_df[label] = train_df[label].astype(float)

    feature_cols = [c for c in train_df.columns if c != label]
    print(f"  Label          : {label}")
    print(f"  Features       : {feature_cols}")
    print(f"  Time limit     : {time_limit}s" if time_limit else "  Time limit     : unlimited")

    predictor = TabularPredictor(
        label        = label,
        path         = path,
        eval_metric  = "rmse",
        problem_type = "regression",
        verbosity    = 2,
    ).fit(
        train_data            = train_df,
        **({"time_limit": time_limit} if time_limit is not None else {}),
        presets               = "best_quality",
        dynamic_stacking      = False,
        hyperparameters = {
            "GBM":      {},   # LightGBM
            "CAT":      {},   # CatBoost
            "XGB":      {},   # XGBoost
            "RF":       {},   # Random Forest
            "NN_TORCH": {},   # Tabular neural network
        },
    )

    predictor.save_space()
    return predictor


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(predictor, train_df):
    """Return AutoGluon feature importance as a dict, sorted descending."""
    try:
        fi = predictor.feature_importance(train_df, silent=True)
        # fi is a DataFrame with index=feature, columns include 'importance'
        return fi["importance"].sort_values(ascending=False).round(4).to_dict()
    except Exception as e:
        print(f"  ⚠ Feature importance failed: {e}")
        return {}


# ── Score all 81 combos ───────────────────────────────────────────────────────

def score_all_combos(predictor, proto_df, raw_json, train_df,
                     predictor_change=None, train_df_change=None) -> dict:
    """
    Score all 30 protocols for every zone × time × intensity combination.
    Returns a dict keyed by "{zone}__{time}__{intensity}" with:
      primary      — all in-bucket protocols, AutoGluon-ranked
      cross_bucket — up to 2 out-of-bucket protocols within 15% of top score
      top_score    — predicted score of the top primary recommendation
    """
    questions   = {q["id"]: q for q in raw_json["questions"]}
    zone_opts   = [o["key"] for o in questions["q1"]["options"]]
    time_opts   = [o["key"] for o in questions["q2"]["options"]]
    intens_opts = [o["key"] for o in questions["q3"]["options"]]
    routing     = {r["key"]: r["codes"] for r in raw_json["routing"]}

    # Population medians for YouTube features
    median_vals = {}
    for col in ["avg_view_percentage", "avg_view_duration"]:
        if col in train_df.columns:
            vals = train_df[col].replace(0, np.nan).dropna()
            median_vals[col] = float(vals.median()) if not vals.empty else 0.0
        else:
            median_vals[col] = 0.0

    intens_to_level = {"acute": 1, "moderate": 2, "mild": 3}
    recommendations = {}

    for zone in zone_opts:
        for time in time_opts:
            for intensity in intens_opts:

                rows = []
                for _, p in proto_df.iterrows():
                    row = {
                        "series":               p["series"],
                        "entry":                p["entry"],
                        "move":                 p["move"],
                        "zone":                 zone,
                        "listener_type":        "unknown",
                        "activity":             "unknown",
                        "listen_method":        "unknown",
                        "settle_time":          "unknown",
                        "level":                intens_to_level.get(intensity, 2),
                        "dur":                  p["dur"],
                        "startHz":              p["startHz"],
                        "endHz":                p["endHz"],
                        "delta":                p["delta"],
                        "hz_abs":               p["hz_abs"],
                        "avg_view_percentage":  median_vals["avg_view_percentage"],
                        "avg_view_duration":    median_vals["avg_view_duration"],
                        "adviser_driven":       1,
                        "protocol_code":        p["protocol_code"],
                    }
                    rows.append(row)

                score_df     = pd.DataFrame(rows)
                feature_cols = [c for c in CAT_COLS + NUM_COLS if c in score_df.columns]
                X_score      = score_df[feature_cols].fillna(0)

                pred_a = predictor.predict(X_score, as_pandas=False)

                # Blend with change_rating model if available
                if predictor_change is not None:
                    try:
                        pred_b = predictor_change.predict(X_score, as_pandas=False)
                        n_a    = len(train_df)
                        n_b    = len(train_df_change) if train_df_change is not None else 0
                        total  = n_a + n_b if (n_a + n_b) > 0 else 1
                        w_a    = n_a / total
                        w_b    = n_b / total
                        score_df["predicted_score"] = pred_a * w_a + pred_b * w_b
                    except Exception as e:
                        print(f"  ⚠ change_rating predictor failed for {zone}/{intensity}: {e}")
                        score_df["predicted_score"] = pred_a
                else:
                    score_df["predicted_score"] = pred_a

                bucket_codes = routing.get(zone, [])
                within  = (
                    score_df[score_df["protocol_code"].isin(bucket_codes)]
                    .sort_values("predicted_score", ascending=False)
                )
                outside = (
                    score_df[~score_df["protocol_code"].isin(bucket_codes)]
                    .sort_values("predicted_score", ascending=False)
                )

                max_in_bucket = float(within["predicted_score"].max()) if not within.empty else 0.0
                threshold     = max_in_bucket * 0.85
                cross_bucket  = (
                    outside[outside["predicted_score"] >= threshold]
                    ["protocol_code"].head(2).tolist()
                )

                key = f"{zone}__{time}__{intensity}"
                recommendations[key] = {
                    "primary":      within["protocol_code"].tolist(),
                    "cross_bucket": cross_bucket,
                    "top_score":    round(max_in_bucket, 3),
                }

    return recommendations


# ── Fallback: rank by protocol mean ──────────────────────────────────────────

def fallback_recommendations(proto_df, raw_json, metrics_df) -> dict:
    """Rank by mean outcome_score when insufficient training data."""
    routing     = {r["key"]: r["codes"] for r in raw_json["routing"]}
    questions   = {q["id"]: q for q in raw_json["questions"]}
    zone_opts   = [o["key"] for o in questions["q1"]["options"]]
    time_opts   = [o["key"] for o in questions["q2"]["options"]]
    intens_opts = [o["key"] for o in questions["q3"]["options"]]

    score_map = {}
    if not metrics_df.empty:
        score_map = dict(zip(
            metrics_df["protocol_code"],
            metrics_df["outcome_score_mean"].fillna(0),
        ))

    result = {}
    for zone in zone_opts:
        for time in time_opts:
            for intensity in intens_opts:
                bucket  = routing.get(zone, [])
                primary = sorted(bucket, key=lambda c: score_map.get(c, 0), reverse=True)
                result[f"{zone}__{time}__{intensity}"] = {
                    "primary":      primary,
                    "cross_bucket": [],
                    "top_score":    None,
                }
    return result


# ── Supabase write-back ───────────────────────────────────────────────────────

def writeback_model_run(supabase, run_date, n_train, leaderboard_summary,
                        top_features, real_only,
                        leaderboard_summary_change=None,
                        n_train_change=0,
                        blend_active=False):
    """Log model run metadata to autogluon_model_runs for drift tracking."""
    supabase.table("autogluon_model_runs").upsert(
        {
            "id":                          f"{run_date}-{'real' if real_only else 'all'}",
            "run_date":                    run_date,
            "n_training_rows":             n_train,
            "n_training_rows_change":      n_train_change,
            "leaderboard_summary":         leaderboard_summary,
            "leaderboard_summary_change":  leaderboard_summary_change or [],
            "top_features":                list(top_features.keys())[:5] if top_features else [],
            "model_path":                  MODEL_PATH,
            "real_only":                   real_only,
            "blend_active":                blend_active,
            "created_at":                  datetime.datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="id",
    ).execute()
    print(f"  ✓ autogluon_model_runs logged  (blend_active={blend_active})")


# ── Write autogluon.json ──────────────────────────────────────────────────────

def write_output(recommendations, feature_importance, leaderboard_summary,
                 n_train, run_date, dry_run=False,
                 feature_importance_change=None, leaderboard_summary_change=None,
                 n_train_change=0, blend_active=False):
    """
    Write autogluon.json to repo root.
    Schema:
      _meta                     — run metadata including dual-model info
      recommendations           — { "zone__time__intensity": { primary, cross_bucket, top_score } }
      feature_importance        — outcome_score model SHAP values
      leaderboard               — outcome_score model leaderboard
      feature_importance_change — change_rating model SHAP values (if trained)
      leaderboard_change        — change_rating model leaderboard (if trained)
    """
    output = {
        "_meta": {
            "generated_iso":           datetime.datetime.now(timezone.utc).isoformat(),
            "run_date":                run_date,
            "n_training_rows":         n_train,
            "n_training_rows_change":  n_train_change,
            "model_path":              MODEL_PATH,
            "model_path_change":       MODEL_PATH_CHANGE,
            "blend_active":            blend_active,
            "source":                  "autogluon_analysis.py",
            "note":                    "Decoupled from data.json. Fetched independently by adviser.html.",
        },
        "recommendations":           recommendations,
        "feature_importance":        feature_importance,
        "leaderboard":               leaderboard_summary,
        "feature_importance_change": feature_importance_change or {},
        "leaderboard_change":        leaderboard_summary_change or [],
    }

    json_str = json.dumps(output, indent=2, ensure_ascii=False)

    if dry_run:
        print("\n── DRY RUN — autogluon.json preview (first 40 lines) ────────")
        for i, line in enumerate(json_str.splitlines()):
            if i >= 40:
                print("  ... (truncated)")
                break
            print(line)
    else:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(json_str)
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f"  ✓ Written to {OUTPUT_PATH}  ({size_kb:.1f} KB)")


# ── Write autogluon_context.json ──────────────────────────────────────────────

def write_context_block(feature_importance, leaderboard_summary,
                        recommendations, n_train, run_date, dry_run=False):
    """
    Write autogluon_context.json — pre-formatted text block for injection
    into moe_sphere_pipeline.py via {{INJECT:AUTOGLUON_CONTEXT}}.

    Called after write_output() on every successful run. The context_block
    field contains a plain-text summary ready to drop into the Sphere user
    prompt so all 5 analyzer models receive the current AutoGluon findings.
    """
    # Top 5 models from leaderboard
    lb_lines = []
    for row in leaderboard_summary[:5]:
        lb_lines.append(
            f"  {row['model']:<40} val_rmse={abs(row['score_val']):.3f}"
        )

    # Feature importance ranked highest → lowest
    fi_lines = []
    for feat, val in sorted(feature_importance.items(),
                            key=lambda x: x[1], reverse=True):
        fi_lines.append(f"  {feat:<28} {val:.4f}")

    # Zone predictions — one row per zone (acute intensity only, first seen)
    seen_zones = set()
    zone_lines = []
    for key, rec in recommendations.items():
        zone = key.split("__")[0]
        if zone in seen_zones:
            continue
        seen_zones.add(zone)
        top  = rec["primary"][0] if rec["primary"] else "n/a"
        cb   = ", ".join(rec.get("cross_bucket", []))
        line = f"  {zone:<16} -> {top:<12} score={rec['top_score']:.3f}"
        if cb:
            line += f"  cross-bucket: {cb}"
        zone_lines.append(line)

    sep   = "=" * 70
    block = "\n".join([
        sep,
        "AUTOGLUON ENSEMBLE ANALYSIS",
        f"Run date: {run_date}  |  Training rows: {n_train}",
        sep,
        "",
        "MODEL PERFORMANCE — Top models (val RMSE, lower is better):",
        *lb_lines,
        "",
        "FEATURE IMPORTANCE — Mean absolute SHAP values",
        "  (Higher = stronger predictor of outcome_score)",
        *fi_lines,
        "",
        "PREDICTED OUTCOME SCORES BY ZONE (acute intensity):",
        "  AutoGluon ensemble predictions, not observed means.",
        *zone_lines,
        "",
        sep,
        "END AUTOGLUON CONTEXT",
        sep,
    ])

    output = {
        "_meta": {
            "generated_iso":   datetime.datetime.now(timezone.utc).isoformat(),
            "run_date":        run_date,
            "n_training_rows": n_train,
            "source":          "autogluon_analysis.py",
            "consumed_by":     "moe_sphere_pipeline.py -> {{INJECT:AUTOGLUON_CONTEXT}}",
        },
        "context_block": block,
    }

    if dry_run:
        print("\n── DRY RUN — autogluon_context.json preview ─────────────────")
        print(block[:800])
        print("  ... (truncated)")
    else:
        with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Written to {CONTEXT_PATH}  ({len(block)} chars)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(real_only=False, dry_run=False, no_train=False,
        time_limit=DEFAULT_TIME_LIMIT):

    print("── THE DRUM PROTOCOLS — AutoGluon Recommender ────────────────")
    run_date = datetime.date.today().isoformat()

    load_dotenv()
    supa_url = os.environ.get("SUPABASE_URL")
    supa_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supa_url or not supa_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY.")
        sys.exit(1)

    supabase           = create_client(supa_url, supa_key)
    proto_df, raw_json = load_protocol_features()

    # ── Load data ─────────────────────────────────────────────────────────────
    short, full, yt = load_training_data(supabase, real_only=real_only)
    n_total = len(short) if not short.empty else 0
    n_real  = int((short["data_type"] == "real").sum()) if not short.empty else 0
    print(f"  Training data  : {n_total} rows total  (real={n_real})")

    train_df = build_features(short, full, yt, proto_df)

    predictor          = None
    predictor_change   = None
    feature_importance = {}
    leaderboard_summary= []
    feature_importance_change = {}
    leaderboard_summary_change= []

    # ── Build change_rating feature matrix ────────────────────────────────────
    train_df_change = build_features_change(full, yt, proto_df)
    n_change = len(train_df_change)
    print(f"  Change rating  : {n_change} rows with valid change_rating")

    # ── Load saved models ─────────────────────────────────────────────────────
    if no_train:
        from autogluon.tabular import TabularPredictor
        if os.path.exists(MODEL_PATH):
            print(f"  Loading outcome_score model from {MODEL_PATH}/")
            predictor           = TabularPredictor.load(MODEL_PATH)
            feature_importance  = get_feature_importance(predictor, train_df)
            lb                  = predictor.leaderboard(silent=True)
            leaderboard_summary = lb[["model", "score_val", "pred_time_val"]].head(8).to_dict("records")
            print(f"  Outcome model  : {len(leaderboard_summary)} models, RMSE {abs(leaderboard_summary[0]['score_val']):.3f}")
        if os.path.exists(MODEL_PATH_CHANGE) and not train_df_change.empty:
            print(f"  Loading change_rating model from {MODEL_PATH_CHANGE}/")
            predictor_change           = TabularPredictor.load(MODEL_PATH_CHANGE)
            feature_importance_change  = get_feature_importance(predictor_change, train_df_change)
            lb_c                       = predictor_change.leaderboard(silent=True)
            leaderboard_summary_change = lb_c[["model", "score_val", "pred_time_val"]].head(8).to_dict("records")
            print(f"  Change model   : {len(leaderboard_summary_change)} models, RMSE {abs(leaderboard_summary_change[0]['score_val']):.3f}")
        elif n_change >= MIN_TRAIN_ROWS_CHANGE:
            print(f"  No change_rating model saved — training now on {n_change} rows...")
            predictor_change           = train_model(train_df_change, time_limit=time_limit,
                                                     label="change_rating", model_path=MODEL_PATH_CHANGE)
            feature_importance_change  = get_feature_importance(predictor_change, train_df_change)
            lb_c                       = predictor_change.leaderboard(silent=True)
            leaderboard_summary_change = lb_c[["model", "score_val", "pred_time_val"]].head(8).to_dict("records")
        else:
            print(f"  ⚠ Only {n_change} change_rating rows (min={MIN_TRAIN_ROWS_CHANGE}) — change_rating model skipped")

    # ── Train ─────────────────────────────────────────────────────────────────
    elif len(train_df) >= MIN_TRAIN_ROWS:
        # Model A — outcome_score (1-second survey)
        print(f"\n[A] Training outcome_score model on {len(train_df)} rows...")
        predictor           = train_model(train_df, time_limit=time_limit,
                                          label="outcome_score", model_path=MODEL_PATH)
        feature_importance  = get_feature_importance(predictor, train_df)
        lb                  = predictor.leaderboard(silent=True)
        leaderboard_summary = lb[["model", "score_val", "pred_time_val"]].head(8).to_dict("records")
        print(f"\n  Outcome model leaderboard:")
        for row in leaderboard_summary:
            print(f"    {row['model']:<40} val_rmse={abs(row['score_val']):.3f}")
        print("\n  Feature importance (top 8):")
        for feat, val in list(feature_importance.items())[:8]:
            print(f"    {feat:<28} {val:.4f}")

        # Model B — change_rating (1-minute survey)
        if n_change >= MIN_TRAIN_ROWS_CHANGE:
            print(f"\n[B] Training change_rating model on {n_change} rows...")
            predictor_change           = train_model(train_df_change, time_limit=time_limit,
                                                     label="change_rating", model_path=MODEL_PATH_CHANGE)
            feature_importance_change  = get_feature_importance(predictor_change, train_df_change)
            lb_c                       = predictor_change.leaderboard(silent=True)
            leaderboard_summary_change = lb_c[["model", "score_val", "pred_time_val"]].head(8).to_dict("records")
            print(f"\n  Change model leaderboard:")
            for row in leaderboard_summary_change:
                print(f"    {row['model']:<40} val_rmse={abs(row['score_val']):.3f}")
        else:
            print(f"  ⚠ Only {n_change} change_rating rows (min={MIN_TRAIN_ROWS_CHANGE}) — change_rating model skipped")

    # ── Fallback ──────────────────────────────────────────────────────────────
    else:
        print(f"  ⚠ Only {len(train_df)} rows (min={MIN_TRAIN_ROWS}) — using fallback mean ranking")

    # ── Score or fallback ─────────────────────────────────────────────────────
    blend_active = predictor is not None and predictor_change is not None
    if predictor is None:
        metrics  = pd.DataFrame(
            supabase.table("protocol_metrics")
            .select("protocol_code, outcome_score_mean").execute().data
        )
        recommendations = fallback_recommendations(proto_df, raw_json, metrics)
        mode = "FALLBACK"
    else:
        recommendations = score_all_combos(
            predictor, proto_df, raw_json, train_df,
            predictor_change=predictor_change,
            train_df_change=train_df_change if blend_active else None,
        )
        if blend_active:
            n_a   = len(train_df)
            n_b   = len(train_df_change) if train_df_change is not None else 0
            total = n_a + n_b if (n_a + n_b) > 0 else 1
            pct_a = round(n_a / total * 100)
            pct_b = round(n_b / total * 100)
            mode = f"AutoGluon dual-model (outcome_score {pct_a}% + change_rating {pct_b}%)"
        elif no_train:
            mode = "AutoGluon outcome_score only (loaded)"
        else:
            mode = "AutoGluon outcome_score only"

    print(f"\n  Mode           : {mode}")
    print(f"  Combos scored  : {len(recommendations)}")

    # ── Write autogluon.json ──────────────────────────────────────────────────
    write_output(
        recommendations             = recommendations,
        feature_importance          = feature_importance,
        leaderboard_summary         = leaderboard_summary,
        n_train                     = len(train_df),
        run_date                    = run_date,
        dry_run                     = dry_run,
        feature_importance_change   = feature_importance_change,
        leaderboard_summary_change  = leaderboard_summary_change,
        n_train_change              = len(train_df_change),
        blend_active                = blend_active,
    )

    # ── Write autogluon_context.json ──────────────────────────────────────────
    write_context_block(
        feature_importance  = feature_importance,
        leaderboard_summary = leaderboard_summary,
        recommendations     = recommendations,
        n_train             = len(train_df),
        run_date            = run_date,
        dry_run             = dry_run,
    )

    # ── Supabase write-back ───────────────────────────────────────────────────
    if not dry_run and predictor is not None and len(train_df) >= MIN_TRAIN_ROWS:
        try:
            writeback_model_run(
                supabase, run_date, len(train_df),
                leaderboard_summary, feature_importance, real_only,
                leaderboard_summary_change = leaderboard_summary_change,
                n_train_change             = len(train_df_change) if train_df_change is not None else 0,
                blend_active               = blend_active,
            )
        except Exception as e:
            print(f"  ⚠ Supabase write-back failed: {e}")

    # ── Sample output ─────────────────────────────────────────────────────────
    print("\n── Sample recommendations ────────────────────────────────────")
    for key in list(recommendations.keys())[:3]:
        r = recommendations[key]
        print(f"  {key}")
        print(f"    primary      : {r['primary']}")
        print(f"    cross_bucket : {r['cross_bucket']}")
        print(f"    top_score    : {r['top_score']}")

    print("\n  ✓ Done.")
    if not dry_run:
        print(f"  Next step: git add {OUTPUT_PATH} {CONTEXT_PATH} && git commit && git push")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THE DRUM PROTOCOLS — AutoGluon recommender (local only)"
    )
    parser.add_argument("--real-only",   action="store_true",
                        help="Filter to data_type='real' only")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Skip all writes, print summary only")
    parser.add_argument("--no-train",    action="store_true",
                        help="Load saved model from autogluon_model/ instead of retraining")
    parser.add_argument("--time-limit",  type=int, default=DEFAULT_TIME_LIMIT,
                        help="AutoGluon training time budget in seconds (default: unlimited)")
    args = parser.parse_args()

    run(
        real_only  = args.real_only,
        dry_run    = args.dry_run,
        no_train   = args.no_train,
        time_limit = args.time_limit,
    )
