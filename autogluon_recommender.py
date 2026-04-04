#!/usr/bin/env python3
"""
autogluon_recommender.py — THE DRUM PROTOCOLS
──────────────────────────────────────────────
AutoGluon-driven protocol recommender. Runs locally — not in GitHub Actions.

Trains on:
  - survey_responses_short  (outcome_score, src, data_type)
  - survey_responses_full   (listener context features)
  - youtube_daily           (completion signal)
  - protocols.json          (design features: delta, dur, startHz, endHz, move)

Outputs:
  - autogluon.json          → repo root, fetched by adviser.html independently
  - autogluon_model/        → saved AutoGluon predictor folder (local only, gitignored)
  - autogluon_model_runs    → Supabase table, model versioning / drift tracking

Intentionally decoupled from moe_sphere_pipeline.py and data.json.
Each script owns its output. adviser.html fetches both files independently.

Usage:
  python autogluon_recommender.py               # train + score + write autogluon.json
  python autogluon_recommender.py --real-only   # filter to data_type='real' only
  python autogluon_recommender.py --dry-run     # skip all writes, print summary
  python autogluon_recommender.py --no-train    # load saved model, skip retraining
  python autogluon_recommender.py --time-limit 300  # AutoGluon time budget in seconds

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

PROTOCOLS_PATH = "protocols.json"
MODEL_PATH     = "autogluon_model"        # folder, not a single file
OUTPUT_PATH    = "autogluon.json"
MIN_TRAIN_ROWS = 20                        # below this, use fallback mean-based ranking
DEFAULT_TIME_LIMIT = None                  # No limit — run until all models complete

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


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(train_df, time_limit=DEFAULT_TIME_LIMIT):
    """
    Train AutoGluon TabularPredictor on outcome_score.
    Uses lightgbm, catboost, xgboost ensemble — no deep learning, no GPU needed.
    Calls save_space() after fit to minimise the model folder size.
    """
    from autogluon.tabular import TabularPredictor

    # Cast outcome_score to float — prevents AutoGluon misdetecting 0-10 integers as multiclass
    train_df = train_df.copy()
    train_df["outcome_score"] = train_df["outcome_score"].astype(float)

    feature_cols = [c for c in train_df.columns if c != "outcome_score"]
    print(f"  Features       : {feature_cols}")
    print(f"  Time limit     : {time_limit}s" if time_limit else "  Time limit     : unlimited")

    predictor = TabularPredictor(
        label        = "outcome_score",
        path         = MODEL_PATH,
        eval_metric  = "rmse",
        problem_type = "regression",   # explicit — prevents multiclass misdetection on 0-10 integers
        verbosity    = 2,
    ).fit(
        train_data            = train_df,
        **({"time_limit": time_limit} if time_limit is not None else {}),
        presets               = "best_quality",
        dynamic_stacking      = False,     # skip DyStack — saves ~150s on small datasets
        hyperparameters = {
            "GBM":  {},   # LightGBM
            "CAT":  {},   # CatBoost
            "XGB":  {},   # XGBoost
            "RF":   {},   # Random Forest (fast, good baseline)
        },

    )

    # Shrink saved model — no impact on prediction accuracy
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

def score_all_combos(predictor, proto_df, raw_json, train_df) -> dict:
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

                score_df["predicted_score"] = predictor.predict(X_score, as_pandas=False)

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
                        top_features, real_only):
    """Log model run metadata to autogluon_model_runs for drift tracking."""
    supabase.table("autogluon_model_runs").upsert(
        {
            "id":                  f"{run_date}-{'real' if real_only else 'all'}",
            "run_date":            run_date,
            "n_training_rows":     n_train,
            "leaderboard_summary": leaderboard_summary,
            "top_features":        list(top_features.keys())[:5] if top_features else [],
            "model_path":          MODEL_PATH,
            "real_only":           real_only,
            "created_at":          datetime.datetime.now(timezone.utc).isoformat(),
        },
        on_conflict="id",
    ).execute()
    print(f"  ✓ autogluon_model_runs logged")


# ── Write autogluon.json ──────────────────────────────────────────────────────

def write_output(recommendations, feature_importance, leaderboard_summary,
                 n_train, run_date, dry_run=False):
    """
    Write autogluon.json to repo root.
    Schema:
      _meta          — run metadata
      recommendations — { "zone__time__intensity": { primary, cross_bucket, top_score } }
      feature_importance — { feature: shap_value, ... }
      leaderboard    — top models from AutoGluon leaderboard
    """
    output = {
        "_meta": {
            "generated_iso":   datetime.datetime.now(timezone.utc).isoformat(),
            "run_date":        run_date,
            "n_training_rows": n_train,
            "model_path":      MODEL_PATH,
            "source":          "autogluon_recommender.py",
            "note":            "Decoupled from data.json. Fetched independently by adviser.html.",
        },
        "recommendations":   recommendations,
        "feature_importance": feature_importance,
        "leaderboard":        leaderboard_summary,
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
    feature_importance = {}
    leaderboard_summary= []

    # ── Load saved model ──────────────────────────────────────────────────────
    if no_train and os.path.exists(MODEL_PATH):
        from autogluon.tabular import TabularPredictor
        print(f"  Loading saved model from {MODEL_PATH}/")
        predictor = TabularPredictor.load(MODEL_PATH)

    # ── Train ─────────────────────────────────────────────────────────────────
    elif len(train_df) >= MIN_TRAIN_ROWS:
        print(f"  Training AutoGluon on {len(train_df)} rows  (time_limit={time_limit}s)...")
        predictor          = train_model(train_df, time_limit=time_limit)
        feature_importance = get_feature_importance(predictor, train_df)
        lb                 = predictor.leaderboard(silent=True)
        leaderboard_summary = lb[["model", "score_val", "pred_time_val"]].head(8).to_dict("records")

        print(f"\n  AutoGluon leaderboard (top models by val RMSE):")
        for row in leaderboard_summary:
            print(f"    {row['model']:<40} val_rmse={abs(row['score_val']):.3f}")

        print("\n  Feature importance (top 8):")
        for feat, val in list(feature_importance.items())[:8]:
            print(f"    {feat:<28} {val:.4f}")

    # ── Fallback ──────────────────────────────────────────────────────────────
    else:
        print(f"  ⚠ Only {len(train_df)} rows (min={MIN_TRAIN_ROWS}) — using fallback mean ranking")

    # ── Score or fallback ─────────────────────────────────────────────────────
    if predictor is None:
        metrics  = pd.DataFrame(
            supabase.table("protocol_metrics")
            .select("protocol_code, outcome_score_mean").execute().data
        )
        recommendations = fallback_recommendations(proto_df, raw_json, metrics)
        mode = "FALLBACK"
    else:
        recommendations = score_all_combos(predictor, proto_df, raw_json, train_df)
        mode = "AutoGluon" if len(train_df) >= MIN_TRAIN_ROWS else "AutoGluon (loaded)"

    print(f"\n  Mode           : {mode}")
    print(f"  Combos scored  : {len(recommendations)}")

    # ── Write autogluon.json ──────────────────────────────────────────────────
    write_output(
        recommendations    = recommendations,
        feature_importance = feature_importance,
        leaderboard_summary= leaderboard_summary,
        n_train            = len(train_df),
        run_date           = run_date,
        dry_run            = dry_run,
    )

    # ── Supabase write-back ───────────────────────────────────────────────────
    if not dry_run and predictor is not None and len(train_df) >= MIN_TRAIN_ROWS:
        try:
            writeback_model_run(
                supabase, run_date, len(train_df),
                leaderboard_summary, feature_importance, real_only,
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
        print(f"  Next step: git add {OUTPUT_PATH} && git commit && git push")


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
