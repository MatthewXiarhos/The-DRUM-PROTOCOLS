#!/usr/bin/env python3
"""
moe_sphere_pipeline.py — THE DRUM PROTOCOLS
──────────────────────────────────────────────
Parallel MOE pipeline: Supabase → stats → Sphere MOE (Nebius) → data.json

This is a standalone parallel to sphere_pipeline.py. It does not modify
or import from sphere_pipeline.py. The two pipelines share only the
Supabase database and the prompts/ directory.

Differences from sphere_pipeline.py:
  - LLM layer: 5 parallel Nebius analyzer calls + 1 synthesizer call
    instead of a single Claude call
  - Adds embedding_analysis field to sphere_commentary and data.json
  - Adds combo_string field to sphere_commentary and data.json
  - Writes 6 rows to moe_runs table (one per model call) in addition to
    all existing write-backs
  - Requires NEBIUS_API_KEY instead of ANTHROPIC_API_KEY

Everything else — stat computation, Supabase write-backs, data.json
schema, YouTube ingestion, embedding generation — is identical.

Usage:
  python moe_sphere_pipeline.py                    # all data + MOE call
  python moe_sphere_pipeline.py --real-only        # real data + MOE call
  python moe_sphere_pipeline.py --no-sphere        # stats only, skip MOE
  python moe_sphere_pipeline.py --dry-run          # print JSON, don't write
  python moe_sphere_pipeline.py --real-only --no-sphere

Requirements:
  pip install supabase scipy numpy python-dotenv openai

Environment (.env or GitHub secrets):
  SUPABASE_URL          — Supabase project URL
  SUPABASE_SERVICE_KEY  — service_role key (bypasses RLS)
  NEBIUS_API_KEY        — Nebius Token Factory API key
  OPENAI_VECTOR_KEY     — OpenAI API key for text-embedding-3-small

Output:
  ./data.json — same file as sphere_pipeline.py; commit message uses [moe]
                tag for traceability in git history.

Confidence scale (matches sphere.html CONF_PCT):
  early       n < 5    22%
  developing  n 5–19   55%
  meaningful  n 20–49  80%
  strong      n >= 50  95%
"""

import os
import sys
import json
import random
import hashlib
import argparse
import datetime
import concurrent.futures
from datetime import timezone
from collections import Counter

import numpy as np
from scipy import stats as scipy_stats
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# ── CONFIGURATION ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "."
PROMPTS_DIR = "prompts"

N_DEVELOPING = 5
N_MEANINGFUL = 20
N_STRONG     = 50

BIMODALITY_BC_THRESHOLD = 0.555

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMS  = 1536

PROTOCOL_META = {
    "H-OS-L1": {"name": "Pebble",     "series": "HEALING",      "entry": "Over-stimulation",        "move": "DESCENT"},
    "H-OS-L2": {"name": "Rainfall",   "series": "HEALING",      "entry": "Over-stimulation",        "move": "DESCENT"},
    "H-OS-L3": {"name": "Glacier",    "series": "HEALING",      "entry": "Over-stimulation",        "move": "DESCENT"},
    "H-AX-L1": {"name": "Meadow",     "series": "HEALING",      "entry": "Anxiety & Stress",        "move": "DESCENT"},
    "H-AX-L2": {"name": "Cottage",    "series": "HEALING",      "entry": "Anxiety & Stress",        "move": "DESCENT"},
    "H-AX-L3": {"name": "Valley",     "series": "HEALING",      "entry": "Anxiety & Stress",        "move": "DESCENT"},
    "H-CR-L1": {"name": "Anchor",     "series": "HEALING",      "entry": "Acute Distress / Crisis", "move": "DESCENT"},
    "H-CR-L2": {"name": "Harbor",     "series": "HEALING",      "entry": "Acute Distress / Crisis", "move": "DESCENT"},
    "H-CR-L3": {"name": "Horizon",    "series": "HEALING",      "entry": "Acute Distress / Crisis", "move": "DESCENT"},
    "H-SL-L1": {"name": "Lantern",    "series": "HEALING",      "entry": "Sleep & Rest",            "move": "DESCENT"},
    "H-SL-L2": {"name": "Hammock",    "series": "HEALING",      "entry": "Sleep & Rest",            "move": "DESCENT"},
    "H-SL-L3": {"name": "Midnight",   "series": "HEALING",      "entry": "Sleep & Rest",            "move": "DESCENT"},
    "T-FC-L1": {"name": "Compass",    "series": "THRIVING",     "entry": "Focus & Clarity",         "move": "HOLD"},
    "T-FC-L2": {"name": "Waypoint",   "series": "THRIVING",     "entry": "Focus & Clarity",         "move": "HOLD"},
    "T-FC-L3": {"name": "Summit",     "series": "THRIVING",     "entry": "Focus & Clarity",         "move": "HOLD"},
    "T-CF-L1": {"name": "Ember",      "series": "THRIVING",     "entry": "Creative Flow",           "move": "HOLD"},
    "T-CF-L2": {"name": "Current",    "series": "THRIVING",     "entry": "Creative Flow",           "move": "HOLD"},
    "T-CF-L3": {"name": "Aurora",     "series": "THRIVING",     "entry": "Creative Flow",           "move": "HOLD"},
    "T-MB-L1": {"name": "Stone",      "series": "THRIVING",     "entry": "Maintenance & Balance",   "move": "HOLD"},
    "T-MB-L2": {"name": "Axis",       "series": "THRIVING",     "entry": "Maintenance & Balance",   "move": "HOLD"},
    "T-MB-L3": {"name": "Orbit",      "series": "THRIVING",     "entry": "Maintenance & Balance",   "move": "HOLD"},
    "X-MD-L1": {"name": "Footpath",   "series": "TRANSFORMING", "entry": "Motivation & Drive",      "move": "ASCENT"},
    "X-MD-L2": {"name": "Trail",      "series": "TRANSFORMING", "entry": "Motivation & Drive",      "move": "ASCENT"},
    "X-MD-L3": {"name": "Switchback", "series": "TRANSFORMING", "entry": "Motivation & Drive",      "move": "ASCENT"},
    "X-ME-L1": {"name": "Breeze",     "series": "TRANSFORMING", "entry": "Mood & Energy",           "move": "ASCENT"},
    "X-ME-L2": {"name": "Sunrise",    "series": "TRANSFORMING", "entry": "Mood & Energy",           "move": "ASCENT"},
    "X-ME-L3": {"name": "Updraft",    "series": "TRANSFORMING", "entry": "Mood & Energy",           "move": "ASCENT"},
    "X-MC-L1": {"name": "Cavern",     "series": "TRANSFORMING", "entry": "Mind & Clarity",          "move": "DESCENT"},
    "X-MC-L2": {"name": "Starlight",  "series": "TRANSFORMING", "entry": "Mind & Clarity",          "move": "DESCENT"},
    "X-MC-L3": {"name": "Cosmos",     "series": "TRANSFORMING", "entry": "Mind & Clarity",          "move": "DESCENT"},
}

# Active prompt filenames
MOE_SYSTEM_FILE      = "sphere__system__v1.md"
MOE_USER_FILE        = "sphere__user__v1.md"
MOE_ANALYZER_FILE    = "sphere__analyzer__v1.md"
MOE_SYNTHESIZER_FILE = "sphere__synthesizer__v1.md"

# ── MOE MODEL POOL ────────────────────────────────────────────────────────────────────
# All model IDs verified against https://nebius.com/token-factory/prices (April 2026).
# Base URL updated to current Nebius Token Factory endpoint.
# cost_in / cost_out: USD per million tokens (base tier).
# synthesizer_eligible: whether the model may serve as the synthesizer.

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

MODEL_POOL = {
    # Code  | Model ID (exact Nebius API id)                              | In $/M | Out $/M | Synth?
    # IDs confirmed via client.models.list() against api.tokenfactory.nebius.com/v1/
    "LLM": {"name": "meta-llama/Meta-Llama-3.1-70B-Instruct",            "cost_in": 0.13, "cost_out": 0.40, "synthesizer_eligible": False},
    "QWN": {"name": "Qwen/Qwen3-235B-A22B",                              "cost_in": 0.20, "cost_out": 0.60, "synthesizer_eligible": False},
    "DSK": {"name": "deepseek-ai/DeepSeek-V3",                           "cost_in": 0.50, "cost_out": 1.50, "synthesizer_eligible": False},
    "HRM": {"name": "NousResearch/Hermes-3-Llama-3.1-70B-Instruct",      "cost_in": 0.13, "cost_out": 0.40, "synthesizer_eligible": False},
    "GPT": {"name": "openai/gpt-4o-mini",                                "cost_in": 0.15, "cost_out": 0.60, "synthesizer_eligible": False},
    "GMM": {"name": "google/gemma-2-9b-it",                              "cost_in": 0.03, "cost_out": 0.09, "synthesizer_eligible": False},
    "GLM": {"name": "THUDM/glm-4-9b-chat",                              "cost_in": 0.03, "cost_out": 0.09, "synthesizer_eligible": False},
    "KIM": {"name": "moonshotai/Kimi-K2-Instruct",                       "cost_in": 0.50, "cost_out": 2.40, "synthesizer_eligible": False},
    "NEM": {"name": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",        "cost_in": 0.13, "cost_out": 0.40, "synthesizer_eligible": True},
    "MST": {"name": "mistralai/Mistral-Nemo-Instruct-2407",              "cost_in": 0.07, "cost_out": 0.07, "synthesizer_eligible": False},
}


def validate_model_pool(client: OpenAI) -> list[str]:
    """
    Query the live Nebius model list and validate every MODEL_POOL entry.
    Prints a warning for any model ID that doesn't appear in the live catalogue.
    Returns list of invalid model codes so callers can remove them from the pool.

    This runs once at pipeline startup — it's the definitive check against 404s.
    If the live API call fails, we warn and continue (don't block the run).
    """
    print("  Validating model pool against live Nebius catalogue...")
    try:
        live_models = {m.id for m in client.models.list().data}
        invalid     = []
        for code, meta in MODEL_POOL.items():
            model_id = meta["name"]
            if model_id not in live_models:
                print(f"    ✗ {code}: '{model_id}' NOT in live catalogue — will be skipped")
                invalid.append(code)
            else:
                print(f"    ✓ {code}: '{model_id}'")
        if not invalid:
            print(f"  All {len(MODEL_POOL)} models validated.")
        else:
            print(f"  WARNING: {len(invalid)} model(s) invalid. Run 'python moe_sphere_pipeline.py --list-models' to see full catalogue.")
        return invalid
    except Exception as e:
        print(f"  ⚠ Model validation failed ({e}) — proceeding without validation")
        return []

N_ANALYZERS            = 5
ANALYZER_MAX_TOKENS    = 4000
SYNTHESIZER_MAX_TOKENS = 16000


# ── PROMPT LOADING ────────────────────────────────────────────────────────────────────

def _strip_frontmatter(text):
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text

def load_prompt(filename):
    filepath = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"ERROR: Prompt file not found: {filepath}")
        sys.exit(1)
    raw          = open(filepath, encoding="utf-8").read()
    content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return _strip_frontmatter(raw), content_hash, filepath


# ── STAT HELPERS ──────────────────────────────────────────────────────────────────────

def bimodality_coefficient(data):
    n = len(data)
    if n < 5:
        return None
    skew = scipy_stats.skew(data)
    kurt = scipy_stats.kurtosis(data)
    bc   = (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
    return round(float(bc), 3)

def score_dist(scores):
    counts = [0] * 11
    for s in scores:
        if s is not None:
            idx = max(0, min(10, int(round(s))))
            counts[idx] += 1
    return counts

def pct_positive(dist):
    total = sum(dist)
    if total == 0:
        return 0
    return round(sum(dist[7:]) / total * 100)

def confidence_level(n):
    if n >= N_STRONG:     return "strong"
    if n >= N_MEANINGFUL: return "meaningful"
    if n >= N_DEVELOPING: return "developing"
    return "early"

def compute_protocol_stats(scores):
    valid = [s for s in scores if s is not None]
    n     = len(valid)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "dist": [0]*11,
                "pct_positive": 0, "confidence": "early",
                "bimodality_coefficient": None, "bimodal_flag": False}
    arr   = np.array(valid, dtype=float)
    dist  = score_dist(valid)
    mean  = round(float(np.mean(arr)), 2)
    std   = round(float(np.std(arr, ddof=1)) if n > 1 else 0.0, 2)
    bc    = bimodality_coefficient(arr)
    return {
        "n": n, "mean": mean, "std": std, "dist": dist,
        "pct_positive":           pct_positive(dist),
        "confidence":             confidence_level(n),
        "bimodality_coefficient": bc,
        "bimodal_flag":           bc is not None and bc > BIMODALITY_BC_THRESHOLD,
    }

def compute_series_summary(protocols_out, short_by_protocol):
    series_scores = {}
    series_count  = {}
    for code, p in protocols_out.items():
        series = p["series"]
        scores = [r["outcome_score"] for r in short_by_protocol.get(code, [])
                  if r["outcome_score"] is not None]
        series_scores.setdefault(series, []).extend(scores)
        series_count[series] = series_count.get(series, 0) + 1
    result = {}
    for series, scores in series_scores.items():
        arr = np.array(scores, dtype=float) if scores else np.array([])
        result[series] = {
            "n":         len(arr),
            "mean":      round(float(np.mean(arr)), 2) if len(arr) else 0.0,
            "std":       round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
            "protocols": series_count.get(series, 0),
        }
    return result


# ── CONTEXT BUILDERS ─────────────────────────────────────────────────────────────────

def build_full_survey_context(full_rows, protocols_out):
    full_by_protocol = {}
    for r in full_rows:
        full_by_protocol.setdefault(r["protocol_code"], []).append(r)
    lines = []
    for code in sorted(protocols_out.keys()):
        rows = full_by_protocol.get(code, [])
        if not rows:
            continue
        def top(field, top_n=3):
            vals = [r.get(field) for r in rows if r.get(field)]
            if not vals: return "n/a"
            counts = Counter(vals)
            return " / ".join(f"{v}({c})" for v, c in counts.most_common(top_n))
        listener_counts = Counter(r.get("listener_type") for r in rows if r.get("listener_type"))
        listener_str    = " | ".join(f"{k}:{v}" for k, v in listener_counts.most_common()) or "n/a"
        lines.append(
            f"{code} | n={len(rows)} | listener_type: {listener_str} | "
            f"change_rating: {top('change_rating')} | settle_time: {top('settle_time', 2)} | "
            f"activity: {top('activity', 2)} | music_opinion: {top('music_opinion', 2)} | "
            f"rhythm_opinion: {top('rhythm_opinion', 2)}"
        )
    return "\n".join(lines) if lines else "No 1-minute survey responses available yet."

def fetch_youtube_context(supabase) -> dict:
    try:
        rows = supabase.table("youtube_daily").select(
            "protocol_code, report_date, views, avg_view_duration, avg_view_percentage"
        ).execute().data
        if not rows:
            return {}
        latest = {}
        for r in rows:
            code = r["protocol_code"]
            if code not in latest or r["report_date"] > latest[code]["report_date"]:
                latest[code] = r
        return {
            code: {
                "views":               row.get("views"),
                "avg_view_duration":   row.get("avg_view_duration"),
                "avg_view_percentage": row.get("avg_view_percentage"),
                "report_date":         row.get("report_date"),
            }
            for code, row in latest.items()
        }
    except Exception as e:
        print(f"  ⚠ YouTube context fetch failed: {e}")
        return {}

def build_youtube_context_block(youtube_data: dict, protocols_out: dict) -> str:
    if not youtube_data:
        return "No YouTube data available yet."
    lines = []
    for code in sorted(protocols_out.keys()):
        yt = youtube_data.get(code)
        if not yt:
            continue
        views    = yt["views"] if yt["views"] is not None else "NULL"
        duration = f"{yt['avg_view_duration']:.0f}s" if yt["avg_view_duration"] is not None else "NULL"
        pct      = f"{yt['avg_view_percentage']:.1f}%" if yt["avg_view_percentage"] is not None else "NULL"
        lines.append(
            f"{code} | views={views} | avg_view_duration={duration} | "
            f"avg_view_percentage={pct} | as_of={yt.get('report_date', '')}"
        )
    return "\n".join(lines) if lines else "No YouTube data available yet."

def fetch_embedding_context(supabase, protocols_out: dict) -> str:
    """
    Cosine similarity drift (earliest vs most recent run per protocol) and
    cross-protocol clustering (latest run). Handles low-data states gracefully.
    """
    try:
        rows = supabase.table("llm_outputs").select(
            "protocol_code, run_date, embedding, analysis_scope"
        ).eq("analysis_scope", "by_protocol").execute().data

        rows_with_emb = [r for r in rows if r.get("embedding")]
        if not rows_with_emb:
            return "No embeddings stored yet — first pipeline run."

        def parse_emb(raw):
            """Supabase returns pgvector as a string '[0.1,...]' — parse to list."""
            if isinstance(raw, str):
                import json
                return json.loads(raw)
            return raw

        def cosine_sim(a, b):
            a, b  = np.array(parse_emb(a), dtype=float), np.array(parse_emb(b), dtype=float)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / denom) if denom else 0.0

        by_protocol = {}
        for r in rows_with_emb:
            code = r.get("protocol_code")
            if code:
                by_protocol.setdefault(code, []).append(r)
        for code in by_protocol:
            by_protocol[code].sort(key=lambda x: x["run_date"])

        all_dates = sorted(set(r["run_date"] for r in rows_with_emb))
        n_runs    = len(all_dates)
        lines     = []

        if n_runs < 4:
            lines.append(
                f"LOW DATA CAVEAT: Only {n_runs} run(s) have embeddings. "
                "All patterns are provisional — treat as early signal only."
            )

        lines.append("\nCROSS-RUN DRIFT (cosine similarity, earliest vs most recent):")
        drift_items = []
        for code, proto_rows in sorted(by_protocol.items()):
            if len(proto_rows) < 2:
                continue
            sim = round(cosine_sim(proto_rows[0]["embedding"], proto_rows[-1]["embedding"]), 3)
            drift_items.append((code, sim, proto_rows[0]["run_date"], proto_rows[-1]["run_date"]))

        if not drift_items:
            lines.append("  Insufficient history (need ≥ 2 runs per protocol).")
        else:
            for code, sim, d0, d1 in sorted(drift_items, key=lambda x: x[1]):
                flag = " ← NOTABLE DRIFT" if sim < 0.85 else ""
                lines.append(f"  {code:<12} sim={sim:.3f}  ({d0} → {d1}){flag}")

        latest_date = all_dates[-1]
        latest_rows = [r for r in rows_with_emb if r["run_date"] == latest_date]
        lines.append(f"\nCROSS-PROTOCOL SEMANTIC CLUSTERS (latest run: {latest_date}):")

        if len(latest_rows) < 3:
            lines.append("  Too few protocols with embeddings for cluster analysis.")
        else:
            codes = [r["protocol_code"] for r in latest_rows]
            embs  = [r["embedding"]     for r in latest_rows]
            sim_matrix = {}
            for i, ci in enumerate(codes):
                for j, cj in enumerate(codes):
                    if i < j:
                        sim_matrix[(ci, cj)] = cosine_sim(embs[i], embs[j])

            lines.append("  Highest similarity pairs:")
            for (ci, cj), sim in sorted(sim_matrix.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"    {ci} ↔ {cj}  sim={sim:.3f}")

            mean_sims = {}
            for code in codes:
                sims = [v for (ci, cj), v in sim_matrix.items() if ci == code or cj == code]
                mean_sims[code] = round(sum(sims) / len(sims), 3) if sims else 0.0

            lines.append("  Most analytically distinct protocols (lowest mean pairwise similarity):")
            for code, sim in sorted(mean_sims.items(), key=lambda x: x[1])[:3]:
                lines.append(f"    {code:<12} mean_sim={sim:.3f}")

        return "\n".join(lines)

    except Exception as e:
        print(f"  ⚠ Embedding context computation failed: {e}")
        return f"Embedding context unavailable: {e}"


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────────────

def build_data_payload(protocols_out, series_summary, total_n,
                       total_n_full=0, full_rows=None,
                       youtube_data=None, embedding_context=""):
    """Renders sphere__user__v1.md with all {{INJECT:*}} placeholders."""
    template, _, _ = load_prompt(MOE_USER_FILE)

    series_lines = [
        f"{s}: n={v['n']}, mean={v['mean']}, std={v['std']}, protocols={v['protocols']}"
        for s, v in series_summary.items()
    ]

    proto_lines = []
    for code, p in sorted(protocols_out.items()):
        meta     = PROTOCOL_META.get(code, {})
        bc       = p.get("bimodality_coefficient")
        flags    = []
        if p["bimodal_flag"]:          flags.append(f"BIMODAL (BC={bc})")
        if p["confidence"] == "early": flags.append("EARLY DATA")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        src_str  = " | ".join(f"{k}:{v}" for k, v in p.get("src_counts", {}).items())
        vol_str  = " | ".join(f"{k}:{v}" for k, v in p.get("vol_counts", {}).items())
        proto_lines.append(
            f"{code} | {meta.get('name','?')} | {meta.get('series','?')} | "
            f"{meta.get('entry','?')} | {meta.get('move','?')} | "
            f"n={p['n']} | mean={p['mean']} | std={p['std']} | "
            f"pct_pos={p['pct_positive']}% | confidence={p['confidence']}{flag_str}"
        )
        proto_lines.append(f"  dist: {p['dist']}")
        if src_str: proto_lines.append(f"  src: {src_str}")
        if vol_str: proto_lines.append(f"  vol: {vol_str}")

    rendered = template
    rendered = rendered.replace("{{INJECT:TODAY}}",               datetime.date.today().isoformat())
    rendered = rendered.replace("{{INJECT:TOTAL_N_SHORT}}",       str(total_n))
    rendered = rendered.replace("{{INJECT:TOTAL_N_FULL}}",        str(total_n_full))
    rendered = rendered.replace("{{INJECT:SERIES_SUMMARY}}",      "\n".join(series_lines))
    rendered = rendered.replace("{{INJECT:PROTOCOL_DATA}}",       "\n".join(proto_lines))
    rendered = rendered.replace("{{INJECT:YOUTUBE_CONTEXT}}",     build_youtube_context_block(youtube_data or {}, protocols_out))
    rendered = rendered.replace("{{INJECT:FULL_SURVEY_CONTEXT}}",  build_full_survey_context(full_rows or [], protocols_out))
    rendered = rendered.replace("{{INJECT:EMBEDDING_CONTEXT}}",   embedding_context or "Not available for this run.")
    return rendered


# ── MOE MODEL SELECTION ───────────────────────────────────────────────────────────────

def draw_panel_from(valid_pool: dict):
    """
    Draw 5 distinct analyzer codes and 1 synthesizer code from valid_pool.
    Synthesizer preference: NEM if available and valid.
    Falls back to any synthesizer_eligible model, then any remaining model.
    Returns (analyzer_codes, synthesizer_code).
    """
    pool = list(valid_pool.keys())
    random.shuffle(pool)
    analyzer_codes = pool[:N_ANALYZERS]
    remaining      = pool[N_ANALYZERS:]
    # Prefer NEM as synthesizer if it is valid and not an analyzer
    if "NEM" in valid_pool and "NEM" not in analyzer_codes:
        synthesizer_code = "NEM"
    else:
        eligible = [c for c in remaining if valid_pool[c]["synthesizer_eligible"]]
        synthesizer_code = random.choice(eligible if eligible else remaining)
    return analyzer_codes, synthesizer_code

def combination_string(analyzer_codes, synthesizer_code):
    return "-".join(analyzer_codes) + ":" + synthesizer_code

def model_cost(code, input_tokens, output_tokens):
    m = MODEL_POOL[code]
    return round(input_tokens / 1_000_000 * m["cost_in"] +
                 output_tokens / 1_000_000 * m["cost_out"], 6)


# ── MOE API CALLS ─────────────────────────────────────────────────────────────────────

def get_nebius_client():
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY not set.")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=NEBIUS_BASE_URL)

def call_model(client, model_code, system, user, max_tokens, label=""):
    """Single model call. Returns result dict."""
    tag = f"[{label or model_code}]"
    try:
        response = client.chat.completions.create(
            model      = MODEL_POOL[model_code]["name"],
            max_tokens = max_tokens,
            messages   = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        out   = response.choices[0].message.content.strip()
        i_tok = response.usage.prompt_tokens
        o_tok = response.usage.completion_tokens
        cost  = model_cost(model_code, i_tok, o_tok)
        print(f"  {tag} ✓  {i_tok}in/{o_tok}out  (~${cost:.4f})")
        return {"model_code": model_code, "output_text": out,
                "input_tokens": i_tok, "output_tokens": o_tok,
                "cost_usd": cost, "error": None}
    except Exception as e:
        print(f"  {tag} ✗  ERROR: {e}")
        return {"model_code": model_code, "output_text": "",
                "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "error": str(e)}

def run_analyzers(client, analyzer_codes, system_text, analyzer_text, data_payload):
    """
    Run N_ANALYZERS in parallel via ThreadPoolExecutor.
    System: Sphere persona only. User: analyzer role instructions + data payload.
    """
    print(f"\n  Running {N_ANALYZERS} analyzers in parallel...")
    # Analyzer instructions prepended to data payload in the user message
    user = analyzer_text + "\n\n" + data_payload
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_ANALYZERS) as ex:
        futures = {
            ex.submit(call_model, client, code, system_text, user,
                      ANALYZER_MAX_TOKENS, f"ANALYZER-{code}"): code
            for code in analyzer_codes
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

def run_synthesizer(client, synthesizer_code, system_text, synthesizer_text,
                    data_payload, analyzer_results):
    """
    Build synthesizer user prompt and call the model.

    System prompt: Sphere persona only (system_text).
    User prompt:   data payload + all analyzer outputs + synthesizer instructions.
    synthesizer_text appears ONCE — in the user message — so the model is not
    confused by receiving identical instructions in both turns.
    """
    print(f"\n  Running synthesizer ({synthesizer_code})...")
    analyzer_block = "\n".join(
        f"=== ANALYZER {r['model_code']} ===\n"
        + (r["output_text"] or "[NO OUTPUT — model error]") + "\n"
        for r in analyzer_results
    )
    user = (
        "=== ORIGINAL DATA PAYLOAD ===\n" + data_payload +
        "\n\n=== ANALYZER OUTPUTS ===\n" + analyzer_block +
        "\n\n=== YOUR TASK ===\n" + synthesizer_text
    )
    # System is Sphere persona only — instructions are in the user message above
    result = call_model(client, synthesizer_code, system_text, user,
                        SYNTHESIZER_MAX_TOKENS, f"SYNTHESIZER-{synthesizer_code}")
    # Log first 300 chars of raw output to help diagnose empty/malformed responses
    preview = result["output_text"][:300].replace("\n", " ") if result["output_text"] else "[EMPTY]"
    print(f"  Synthesizer raw preview: {preview}")
    return result

def parse_json_output(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  WARNING: Synthesizer returned invalid JSON ({e}). Saving raw.")
        return {"raw": raw, "parse_error": str(e)}

def call_sphere_moe(protocols_out, series_summary, total_n,
                    total_n_full=0, full_rows=None, youtube_data=None,
                    embedding_context=""):
    """
    Full MOE pipeline: draw panel → 5 parallel analyzers → 1 synthesizer.
    Returns (commentary_dict, usage_dict, prompt_versions_info,
             analyzer_results, synthesizer_result, combo).
    """
    client = get_nebius_client()

    # Validate model IDs against live catalogue — removes invalid codes from pool
    invalid_codes = validate_model_pool(client)
    valid_pool    = {k: v for k, v in MODEL_POOL.items() if k not in invalid_codes}
    if len(valid_pool) < N_ANALYZERS + 1:
        print(f"ERROR: Only {len(valid_pool)} valid models available, need at least {N_ANALYZERS + 1}.")
        print("       Run: python moe_sphere_pipeline.py --list-models")
        sys.exit(1)

    system_text,      system_hash,      _ = load_prompt(MOE_SYSTEM_FILE)
    analyzer_text,    analyzer_hash,    _ = load_prompt(MOE_ANALYZER_FILE)
    synthesizer_text, synthesizer_hash, _ = load_prompt(MOE_SYNTHESIZER_FILE)
    _,                user_hash,        _ = load_prompt(MOE_USER_FILE)

    prompt_versions_info = [
        {"filename": MOE_SYSTEM_FILE,      "content_hash": system_hash},
        {"filename": MOE_ANALYZER_FILE,    "content_hash": analyzer_hash},
        {"filename": MOE_SYNTHESIZER_FILE, "content_hash": synthesizer_hash},
        {"filename": MOE_USER_FILE,        "content_hash": user_hash},
    ]

    data_payload = build_data_payload(
        protocols_out, series_summary, total_n, total_n_full,
        full_rows=full_rows, youtube_data=youtube_data,
        embedding_context=embedding_context,
    )

    analyzer_codes, synthesizer_code = draw_panel_from(valid_pool)
    combo = combination_string(analyzer_codes, synthesizer_code)
    print(f"\n  MOE panel  : analyzers={analyzer_codes}  synthesizer={synthesizer_code}")
    print(f"  Combo str  : {combo}")

    analyzer_results   = run_analyzers(client, analyzer_codes, system_text, analyzer_text, data_payload)

    # Hard-fail if every analyzer errored — synthesizer has nothing to work with
    successful_analyzers = [r for r in analyzer_results if not r["error"]]
    if not successful_analyzers:
        print("ERROR: All analyzers failed. Check model IDs and NEBIUS_API_KEY.")
        sys.exit(1)
    if len(successful_analyzers) < N_ANALYZERS:
        print(f"  WARNING: {N_ANALYZERS - len(successful_analyzers)} analyzer(s) failed — proceeding with {len(successful_analyzers)} outputs")

    synthesizer_result = run_synthesizer(client, synthesizer_code, system_text,
                                         synthesizer_text, data_payload, analyzer_results)

    if synthesizer_result["error"]:
        print(f"ERROR: Synthesizer failed: {synthesizer_result['error']}")
        sys.exit(1)

    all_results = analyzer_results + [synthesizer_result]
    usage = {
        "input_tokens":  sum(r["input_tokens"]  for r in all_results),
        "output_tokens": sum(r["output_tokens"] for r in all_results),
        "cost_usd":      round(sum(r["cost_usd"] for r in all_results), 6),
    }
    print(f"\n  MOE total  : {usage['input_tokens']}in / {usage['output_tokens']}out  (~${usage['cost_usd']:.4f})")

    commentary           = parse_json_output(synthesizer_result["output_text"])
    commentary["_combo"] = combo

    return commentary, usage, prompt_versions_info, analyzer_results, synthesizer_result, combo


# ── SUPABASE WRITE-BACK ───────────────────────────────────────────────────────────────

def mode_value(values):
    clean = [v for v in values if v is not None]
    return max(set(clean), key=clean.count) if clean else None

def writeback_metrics(supabase, protocols_out, short_by_protocol, full_rows, run_date):
    now              = datetime.datetime.now(timezone.utc).isoformat()
    full_by_protocol = {}
    for r in full_rows:
        full_by_protocol.setdefault(r["protocol_code"], []).append(r)

    print("  → Upserting protocol_metrics...")
    proto_rows = []
    for code, p in protocols_out.items():
        full = full_by_protocol.get(code, [])
        proto_rows.append({
            "protocol_code":        code,
            "short_survey_count":   p["n"],
            "full_survey_count":    len(full),
            "outcome_score_mean":   float(p["mean"]) if p["mean"] else None,
            "outcome_score_stddev": float(p["std"])  if p["std"]  else None,
            "change_rating_mode":   mode_value([r.get("change_rating") for r in full]),
            "music_opinion_mode":   mode_value([r.get("music_opinion") for r in full]),
            "rhythm_opinion_mode":  mode_value([r.get("rhythm_opinion") for r in full]),
            "confidence_level":     p["confidence"],
            "last_updated":         now,
        })
    supabase.table("protocol_metrics").upsert(proto_rows, on_conflict="protocol_code").execute()
    print(f"    ✓ {len(proto_rows)} protocol_metrics rows upserted")

    print("  → Upserting protocol_volume_metrics...")
    pv_rows = []
    for code, p in protocols_out.items():
        for vol_code, _ in p.get("vol_counts", {}).items():
            if vol_code == "unknown":
                continue
            pvid       = f"{code}-{vol_code}"
            vol_scores = [r["outcome_score"] for r in short_by_protocol.get(code, [])
                          if r.get("volume_code") == vol_code and r["outcome_score"] is not None]
            s    = compute_protocol_stats(vol_scores)
            full = [r for r in full_by_protocol.get(code, []) if r.get("volume_code") == vol_code]
            pv_rows.append({
                "pvid":                 pvid,
                "protocol_code":        code,
                "volume_code":          vol_code,
                "short_survey_count":   s["n"],
                "full_survey_count":    len(full),
                "outcome_score_mean":   float(s["mean"]) if s["mean"] else None,
                "outcome_score_stddev": float(s["std"])  if s["std"]  else None,
                "change_rating_mode":   mode_value([r.get("change_rating") for r in full]),
                "music_opinion_mode":   mode_value([r.get("music_opinion") for r in full]),
                "rhythm_opinion_mode":  mode_value([r.get("rhythm_opinion") for r in full]),
                "confidence_level":     s["confidence"],
                "last_updated":         now,
            })
    if pv_rows:
        supabase.table("protocol_volume_metrics").upsert(pv_rows, on_conflict="pvid").execute()
        print(f"    ✓ {len(pv_rows)} protocol_volume_metrics rows upserted")

    print("  → Upserting volume_metrics...")
    vol_scores_map, vol_full_map = {}, {}
    for code, p in protocols_out.items():
        for r in short_by_protocol.get(code, []):
            vol = r.get("volume_code") or "unknown"
            if vol == "unknown": continue
            vol_scores_map.setdefault(vol, [])
            if r["outcome_score"] is not None:
                vol_scores_map[vol].append(r["outcome_score"])
        for r in full_by_protocol.get(code, []):
            vol = r.get("volume_code") or "unknown"
            if vol != "unknown":
                vol_full_map.setdefault(vol, []).append(r)
    vol_rows = []
    for vol_code, scores in vol_scores_map.items():
        s    = compute_protocol_stats(scores)
        full = vol_full_map.get(vol_code, [])
        vol_rows.append({
            "volume_code":          vol_code,
            "short_survey_count":   s["n"],
            "full_survey_count":    len(full),
            "outcome_score_mean":   float(s["mean"]) if s["mean"] else None,
            "outcome_score_stddev": float(s["std"])  if s["std"]  else None,
            "music_opinion_mode":   mode_value([r.get("music_opinion") for r in full]),
            "rhythm_opinion_mode":  mode_value([r.get("rhythm_opinion") for r in full]),
            "confidence_level":     s["confidence"],
            "last_updated":         now,
        })
    if vol_rows:
        supabase.table("volume_metrics").upsert(vol_rows, on_conflict="volume_code").execute()
        print(f"    ✓ {len(vol_rows)} volume_metrics rows upserted")

def generate_embeddings(rows):
    api_key = os.environ.get("OPENAI_VECTOR_KEY")
    if not api_key:
        print("  ⚠ OPENAI_VECTOR_KEY not set — skipping embeddings")
        return 0
    texts, indices = [], []
    for i, row in enumerate(rows):
        text = row.get("output_text", "")
        if text:
            texts.append(text)
            indices.append(i)
    if not texts:
        return 0
    client   = OpenAI(api_key=api_key)
    print(f"  → Generating {len(texts)} embeddings (text-embedding-3-small)...")
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL,
                                        input=texts, dimensions=OPENAI_EMBEDDING_DIMS)
    for result, idx in zip(response.data, indices):
        rows[idx]["embedding"] = result.embedding
    cost_usd = round(response.usage.total_tokens / 1_000_000 * 0.02, 6)
    print(f"    ✓ {len(texts)} embeddings  ({response.usage.total_tokens} tokens, ~${cost_usd:.6f})")
    return len(texts)

def writeback_llm_outputs(supabase, protocols_out, sphere_commentary,
                          run_date, model_used, prompt_version="v1"):
    now  = datetime.datetime.now(timezone.utc).isoformat()
    rows = []
    for scope_key in ("overview", "cross_series", "anomalies",
                      "development_signals", "embedding_analysis"):
        text = sphere_commentary.get(scope_key, "")
        if not text: continue
        rows.append({
            "id":                 f"{run_date}-global-{scope_key}",
            "pvid":               None,
            "protocol_code":      None,
            "volume_code":        None,
            "analysis_scope":     scope_key,
            "run_date":           run_date,
            "model_used":         model_used,
            "prompt_version":     prompt_version,
            "output_text":        text,
            "outcome_score_mean": None,
            "confidence_level":   None,
            "sphere_signal":      None,
            "anomaly_flagged":    scope_key == "anomalies" and bool(text),
            "embedding":          None,
            "created_at":         now,
        })
    by_protocol       = sphere_commentary.get("by_protocol", {})
    by_protocol_plain = sphere_commentary.get("by_protocol_plain", {})
    for code, p in protocols_out.items():
        tech_text  = by_protocol.get(code, "")
        plain_text = by_protocol_plain.get(code, "")
        if not tech_text and not plain_text: continue
        combined = json.dumps({"technical": tech_text, "plain": plain_text}, ensure_ascii=False)
        vol_code  = next((v for v in p.get("vol_counts", {}) if v != "unknown"), "VOL001")
        pvid      = f"{code}-{vol_code}"
        rows.append({
            "id":                 f"{run_date}-{pvid}",
            "pvid":               pvid,
            "protocol_code":      code,
            "volume_code":        vol_code,
            "analysis_scope":     "by_protocol",
            "run_date":           run_date,
            "model_used":         model_used,
            "prompt_version":     prompt_version,
            "output_text":        combined,
            "outcome_score_mean": float(p["mean"]) if p["mean"] else None,
            "confidence_level":   p["confidence"],
            "sphere_signal":      tech_text[:500] if tech_text else None,
            "anomaly_flagged":    p.get("bimodal_flag", False),
            "embedding":          None,
            "created_at":         now,
        })
    if rows:
        generate_embeddings(rows)
        supabase.table("llm_outputs").upsert(rows, on_conflict="id").execute()
        print(f"    ✓ {len(rows)} llm_outputs rows written "
              f"({sum(1 for r in rows if r.get('embedding'))} with embeddings)")

def writeback_moe_runs(supabase, run_date, analyzer_results,
                       synthesizer_result, combo):
    now  = datetime.datetime.now(timezone.utc).isoformat()
    rows = []
    for r in analyzer_results:
        code = r["model_code"]
        rows.append({
            "id":            f"{run_date}-{code}",
            "run_date":      run_date,
            "model_code":    code,
            "model_name":    MODEL_POOL[code]["name"],
            "role":          "analyzer",
            "output_text":   r["output_text"],
            "input_tokens":  r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "cost_usd":      r["cost_usd"],
            "created_at":    now,
        })
    code = synthesizer_result["model_code"]
    rows.append({
        "id":            f"{run_date}-{code}-synth",
        "run_date":      run_date,
        "model_code":    code,
        "model_name":    MODEL_POOL[code]["name"],
        "role":          "synthesizer",
        "output_text":   synthesizer_result["output_text"],
        "input_tokens":  synthesizer_result["input_tokens"],
        "output_tokens": synthesizer_result["output_tokens"],
        "cost_usd":      synthesizer_result["cost_usd"],
        "created_at":    now,
    })
    supabase.table("moe_runs").upsert(rows, on_conflict="id").execute()
    print(f"    ✓ moe_runs: {len(rows)} rows written (combo: {combo})")

def writeback_prompt_versions(supabase, prompt_versions_info, run_date):
    now  = datetime.datetime.now(timezone.utc).isoformat()
    rows = []
    for info in prompt_versions_info:
        filename  = info["filename"]
        parts     = filename.replace(".md", "").split("__")
        record_id = "__".join(parts[:3]) if len(parts) >= 3 else filename.replace(".md", "")
        rows.append({
            "id":             record_id,
            "version":        parts[2] if len(parts) > 2 else "v1",
            "task_type":      parts[0] if len(parts) > 0 else "sphere",
            "analysis_scope": parts[1] if len(parts) > 1 else "system",
            "prompt_text":    open(os.path.join(PROMPTS_DIR, filename), encoding="utf-8").read(),
            "is_active":      True,
            "created_at":     now,
        })
    if rows:
        supabase.table("prompt_versions").upsert(rows, on_conflict="id").execute()
        print(f"    ✓ prompt_versions logged ({len(rows)} prompts: "
              + ", ".join(r["id"] for r in rows) + ")")

def writeback_pipeline_run(supabase, run_id, run_date, run_type,
                           tasks, status, duration_seconds, error_message=None):
    supabase.table("pipeline_runs").upsert({
        "id":               run_id,
        "run_date":         run_date,
        "run_type":         run_type,
        "tasks_executed":   tasks,
        "status":           status,
        "error_message":    error_message,
        "duration_seconds": round(duration_seconds, 2),
        "created_at":       datetime.datetime.now(timezone.utc).isoformat(),
    }, on_conflict="id").execute()
    print(f"    ✓ pipeline_runs logged (status={status}, {duration_seconds:.1f}s)")

def writeback_budget(supabase, run_date, input_tokens, output_tokens, cost_usd):
    import calendar
    today          = datetime.date.fromisoformat(run_date)
    month_start    = today.replace(day=1).isoformat()
    existing       = supabase.table("budget_tracking").select("llm_cost_usd") \
                             .gte("record_date", month_start).execute().data
    monthly_so_far = sum(r["llm_cost_usd"] or 0 for r in existing)
    monthly_total  = round(monthly_so_far + cost_usd, 6)
    days_elapsed   = today.day
    days_in_month  = calendar.monthrange(today.year, today.month)[1]
    forecast       = round(monthly_total / days_elapsed * days_in_month, 4) if days_elapsed else monthly_total
    budget_state   = "healthy"
    if forecast > 50 * 0.9: budget_state = "caution"
    if forecast > 50:        budget_state = "over_budget"
    supabase.table("budget_tracking").upsert({
        "id":                    f"{run_date}-moe",
        "record_date":           run_date,
        "llm_cost_usd":          round(cost_usd, 6),
        "tokens_used":           input_tokens + output_tokens,
        "budget_state":          budget_state,
        "monthly_spend_to_date": monthly_total,
        "monthly_forecast_usd":  forecast,
        "created_at":            datetime.datetime.now(timezone.utc).isoformat(),
    }, on_conflict="id").execute()
    print(f"    ✓ budget_tracking logged  (${cost_usd:.4f} this run, "
          f"${monthly_total:.4f} MTD, forecast ${forecast:.2f}, state={budget_state})")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────────────

def run_pipeline(real_only=False, skip_sphere=False, dry_run=False, output_path=None):
    import time as _time
    run_start     = _time.time()
    print("── THE DRUM PROTOCOLS — Sphere MOE Pipeline ─────────────────")
    today         = datetime.date.today().isoformat()
    generated_iso = datetime.datetime.now(timezone.utc).isoformat()
    run_id        = f"{today}-moe-{'real' if real_only else 'all'}"
    mode_label    = "REAL DATA ONLY" if real_only else "ALL DATA (test + real)"
    usage         = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    run_status    = "success"
    run_error     = None

    print(f"  Mode    : {mode_label}")
    print(f"  Sphere  : {'SKIP (--no-sphere)' if skip_sphere else 'ENABLED (MOE)'}")
    print()

    # ── 1. Connect ─────────────────────────────────────────────────────────
    load_dotenv()
    supa_url = os.environ.get("SUPABASE_URL")
    supa_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supa_url or not supa_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY.")
        sys.exit(1)
    if not skip_sphere and not os.environ.get("NEBIUS_API_KEY"):
        print("ERROR: NEBIUS_API_KEY not set. Use --no-sphere to skip the MOE call.")
        sys.exit(1)
    supabase = create_client(supa_url, supa_key)

    # ── 2. Registry ────────────────────────────────────────────────────────
    print("[1/5] Loading registry from Supabase...")
    protocols_raw     = supabase.table("protocol_registry").select("*").execute().data
    volumes_raw       = supabase.table("volumes").select("*").execute().data
    protocols_by_code = {p["protocol_code"]: p for p in protocols_raw}
    print(f"  {len(protocols_raw)} protocols, {len(volumes_raw)} volumes")

    # ── 3. Survey responses ────────────────────────────────────────────────
    print("\n[2/5] Loading survey responses from Supabase...")
    short_q = supabase.table("survey_responses_short").select(
        "protocol_code, volume_code, outcome_score, src, data_type"
    )
    if real_only:
        short_q = short_q.eq("data_type", "real")
    short_rows = short_q.execute().data
    test_short = sum(1 for r in short_rows if r.get("data_type") == "test")
    real_short = sum(1 for r in short_rows if r.get("data_type") == "real")
    print(f"  short survey : {len(short_rows)} rows  (real={real_short}, test={test_short})")

    full_q = supabase.table("survey_responses_full").select(
        "protocol_code, volume_code, data_type, "
        "state_before, state_after, change_rating, settle_time, activity, "
        "listen_method, listen_frequency, music_opinion, rhythm_opinion, "
        "listener_type, open_feedback"
    )
    if real_only:
        full_q = full_q.eq("data_type", "real")
    full_rows     = full_q.execute().data
    contains_test = any(r.get("data_type") == "test" for r in short_rows)
    print(f"  full survey  : {len(full_rows)} rows")

    short_by_protocol = {}
    for r in short_rows:
        short_by_protocol.setdefault(r["protocol_code"], []).append(r)

    # ── 4. YouTube + embedding context ────────────────────────────────────
    print("\n[3/5] Loading YouTube data from Supabase...")
    youtube_data = fetch_youtube_context(supabase)
    yt_count     = sum(1 for v in youtube_data.values() if v.get("views") is not None)
    print(f"  {len(youtube_data)} protocols with YouTube data ({yt_count} with view counts)")

    print("\n[3b/5] Computing embedding context...")
    embedding_context = fetch_embedding_context(supabase, {})
    print(f"  Embedding context: {embedding_context.count(chr(10)) + 1} lines")

    # ── 5. Stats ───────────────────────────────────────────────────────────
    print("\n[4/5] Computing protocol statistics...")
    protocols_out = {}
    for code, p in protocols_by_code.items():
        rows   = short_by_protocol.get(code, [])
        scores = [r["outcome_score"] for r in rows if r["outcome_score"] is not None]
        s      = compute_protocol_stats(scores)

        src_counts, vol_counts = {}, {}
        for r in rows:
            src = r.get("src") or "unknown"
            vol = r.get("volume_code") or "unknown"
            src_counts[src] = src_counts.get(src, 0) + 1
            vol_counts[vol] = vol_counts.get(vol, 0) + 1

        meta   = PROTOCOL_META.get(code, {})
        name   = p.get("full_name")   or meta.get("name",   code)
        series = (p.get("series")     or meta.get("series", "UNKNOWN")).upper()
        entry  = p.get("entry_point") or meta.get("entry",  "")

        bimod_flag = " [BIMODAL]" if s["bimodal_flag"] else ""
        print(f"  {code:<12} n={s['n']:<4} mean={s['mean']:.1f}  {s['confidence']}{bimod_flag}")

        protocols_out[code] = {
            "name": name, "series": series, "entry": entry,
            "n": s["n"], "mean": s["mean"], "std": s["std"],
            "dist": s["dist"], "pct_positive": s["pct_positive"],
            "confidence": s["confidence"], "bimodal_flag": s["bimodal_flag"],
            "bimodality_coefficient": s["bimodality_coefficient"],
            "sphere": "", "sphere_plain": "",
            "src_counts": src_counts, "vol_counts": vol_counts,
        }

    series_summary = compute_series_summary(protocols_out, short_by_protocol)
    total_n        = sum(p["n"] for p in protocols_out.values())

    # ── 6. MOE commentary ──────────────────────────────────────────────────
    print()
    prompt_versions_info = []

    if skip_sphere:
        print("[5/5] Skipping Sphere MOE call (--no-sphere)")
        sphere_commentary = {
            "overview": "[MOE commentary not generated — run without --no-sphere]",
            "cross_series": "", "anomalies": "", "development_signals": "",
            "embedding_analysis": "",
            "by_protocol":       {code: "" for code in protocols_out},
            "by_protocol_plain": {code: "" for code in protocols_out},
        }
        analyzer_results   = []
        synthesizer_result = {}
        combo              = ""
    else:
        print("[5/5] Generating Sphere MOE commentary...")
        (sphere_commentary, usage, prompt_versions_info,
         analyzer_results, synthesizer_result, combo) = call_sphere_moe(
            protocols_out, series_summary, total_n, total_n_full=len(full_rows),
            full_rows=full_rows, youtube_data=youtube_data,
            embedding_context=embedding_context,
        )
        by_protocol       = sphere_commentary.get("by_protocol", {})
        by_protocol_plain = sphere_commentary.get("by_protocol_plain", {})
        for code in protocols_out:
            protocols_out[code]["sphere"]       = by_protocol.get(code, "")
            protocols_out[code]["sphere_plain"] = by_protocol_plain.get(code, "")
        print("  Sphere MOE commentary generated.")

    # ── 7. Assemble data.json ──────────────────────────────────────────────
    protocols_with_data = sum(1 for p in protocols_out.values() if p["n"] > 0)
    total_all   = len(short_rows) + len(full_rows)
    total_test  = (sum(1 for r in short_rows if r.get("data_type") == "test") +
                   sum(1 for r in full_rows  if r.get("data_type") == "test"))
    test_pct    = round(total_test / total_all * 100, 1) if total_all else 0.0

    def public_protocol(p):
        return {k: v for k, v in p.items()
                if k not in ("src_counts", "vol_counts", "bimodality_coefficient")}

    data_json = {
        "_meta": {
            "generated_iso":      generated_iso,
            "mode":               "real_only" if real_only else "all",
            "contains_test_data": contains_test,
            "total_responses":    total_n,
            "test_pct":           test_pct,
            "real_pct":           round(100 - test_pct, 1),
        },
        "generated":           today,
        "analysis_date":       today,
        "protocols_with_data": protocols_with_data,
        "protocols":           {code: public_protocol(p) for code, p in protocols_out.items()},
        "series_summary":      series_summary,
        "sphere_commentary": {
            "overview":            sphere_commentary.get("overview", ""),
            "cross_series":        sphere_commentary.get("cross_series", ""),
            "anomalies":           sphere_commentary.get("anomalies", ""),
            "development_signals": sphere_commentary.get("development_signals", ""),
            "embedding_analysis":  sphere_commentary.get("embedding_analysis", ""),
            "combo_string":        sphere_commentary.get("_combo", ""),
            "by_protocol":         sphere_commentary.get("by_protocol",       {code: "" for code in protocols_out}),
            "by_protocol_plain":   sphere_commentary.get("by_protocol_plain", {code: "" for code in protocols_out}),
        },
    }

    # ── 8. Output ──────────────────────────────────────────────────────────
    json_str    = json.dumps(data_json, indent=2, ensure_ascii=False)
    output_path = output_path or os.path.join(OUTPUT_DIR, "data.json")

    if dry_run:
        print("\n── DRY RUN — output (first 80 lines) ────────────────────────")
        for i, line in enumerate(json_str.splitlines()):
            if i >= 80: print("  ... (truncated)"); break
            print(line)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\n  ✓ Written to {output_path}  ({os.path.getsize(output_path)/1024:.1f} KB)")

    # ── 9. Write-back to Supabase ──────────────────────────────────────────
    if not dry_run:
        print("\n[5/5] Writing results back to Supabase...")
        try:
            writeback_metrics(supabase, protocols_out, short_by_protocol, full_rows, today)

            if not skip_sphere and not sphere_commentary.get("parse_error"):
                writeback_llm_outputs(
                    supabase, protocols_out, sphere_commentary,
                    run_date=today, model_used=combo,
                    prompt_version=MOE_SYSTEM_FILE.replace(".md", ""),
                )
                writeback_moe_runs(supabase, today, analyzer_results, synthesizer_result, combo)
                writeback_prompt_versions(supabase, prompt_versions_info, today)
                writeback_budget(supabase, today,
                                 usage["input_tokens"], usage["output_tokens"], usage["cost_usd"])
        except Exception as e:
            run_status = "partial"
            run_error  = f"write-back error: {e}"
            print(f"  WARNING: write-back failed — {e}")

    # ── 10. Log pipeline run ───────────────────────────────────────────────
    if not dry_run:
        import time as _time2
        duration = _time2.time() - run_start
        tasks    = ["load_registry", "load_surveys", "load_youtube",
                    "compute_embedding_context", "compute_stats"]
        if not skip_sphere:
            tasks += ["moe_analyzers", "moe_synthesizer", "writeback_moe_runs",
                      "writeback_llm_outputs", "writeback_budget"]
        tasks += ["writeback_metrics", "write_data_json"]
        try:
            writeback_pipeline_run(
                supabase, run_id, today,
                run_type="moe_full" if not skip_sphere else "moe_stats_only",
                tasks=tasks, status=run_status, duration_seconds=duration,
                error_message=run_error,
            )
        except Exception as e:
            print(f"  WARNING: pipeline_runs logging failed — {e}")

    # ── 11. Summary ────────────────────────────────────────────────────────
    SERIES_ORDER = {"HEALING": 0, "THRIVING": 1, "TRANSFORMING": 2}
    print()
    print("=" * 60)
    print(f"  Mode              : {mode_label}")
    print(f"  Contains test data: {contains_test}")
    print(f"  Total responses   : {total_n}")
    print(f"  Protocols w/ data : {protocols_with_data} / {len(protocols_out)}")
    if not skip_sphere:
        print(f"  MOE combo         : {combo}")
        print(f"  LLM cost this run : ${usage['cost_usd']:.4f}  "
              f"({usage['input_tokens']} in / {usage['output_tokens']} out tokens)")
    print()
    print("  Series summary:")
    for series, s in series_summary.items():
        print(f"    {series:<14} n={s['n']:<5} mean={s['mean']:.2f}  "
              f"std={s['std']:.2f}  protocols={s['protocols']}")
    print()
    print("  Protocol breakdown (n / mean / confidence):")
    for code, p in sorted(protocols_out.items(),
                          key=lambda x: (SERIES_ORDER.get(x[1]["series"], 9), x[0])):
        bimod = " [BIMODAL]" if p["bimodal_flag"] else ""
        print(f"    {code:<12} n={p['n']:<4} mean={p['mean']:.1f}  {p['confidence']}{bimod}")

    if youtube_data:
        print()
        print("  YouTube data summary (most recent row per protocol):")
        print(f"    {'CODE':<12} {'VIEWS':>6}  {'AVG DURATION':>13}  {'AVG RETENTION':>14}  {'AS OF'}")
        print(f"    {'-'*12} {'-'*6}  {'-'*13}  {'-'*14}  {'-'*10}")
        series_yt = {}
        for code, p in sorted(protocols_out.items(),
                               key=lambda x: (SERIES_ORDER.get(x[1]["series"], 9), x[0])):
            yt     = youtube_data.get(code)
            series = p["series"]
            views  = yt["views"]               if yt and yt.get("views")               is not None else None
            dur    = yt["avg_view_duration"]   if yt and yt.get("avg_view_duration")   is not None else None
            pct    = yt["avg_view_percentage"] if yt and yt.get("avg_view_percentage") is not None else None
            as_of  = yt["report_date"]         if yt else "—"
            views_s = f"{views:>6}"    if views is not None else "  NULL"
            dur_s   = f"{dur:>10.0f}s" if dur   is not None else "         NULL"
            pct_s   = f"{pct:>11.1f}%" if pct   is not None else "         NULL"
            print(f"    {code:<12} {views_s}  {dur_s:>13}  {pct_s:>14}  {as_of}")
            if views is not None:
                series_yt.setdefault(series, {"views": [], "dur": [], "pct": []})
                series_yt[series]["views"].append(views)
                if dur is not None: series_yt[series]["dur"].append(dur)
                if pct is not None: series_yt[series]["pct"].append(pct)
        if series_yt:
            print()
            print("  YouTube by series:")
            for series in ["HEALING", "THRIVING", "TRANSFORMING"]:
                yt_s = series_yt.get(series)
                if not yt_s: continue
                total_views = sum(yt_s["views"])
                # Guard against empty lists before formatting
                mean_dur = round(sum(yt_s["dur"]) / len(yt_s["dur"]), 0) if yt_s["dur"] else None
                mean_pct = round(sum(yt_s["pct"]) / len(yt_s["pct"]), 1) if yt_s["pct"] else None
                dur_s    = f"{mean_dur:.0f}s" if mean_dur is not None else "NULL"
                pct_s    = f"{mean_pct:.1f}%" if mean_pct is not None else "NULL"
                print(f"    {series:<14} total_views={total_views:<5}  "
                      f"mean_duration={dur_s:<8}  mean_retention={pct_s}")
    else:
        print("  YouTube data summary: no data available")

    print()
    if not dry_run:
        print("  Next step: data.json committed automatically by moe_sphere.yml.")
    print("=" * 60)
    print()

    return data_json


# ── CLI ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THE DRUM PROTOCOLS — MOE pipeline: Supabase → stats → Nebius MOE → data.json"
    )
    parser.add_argument("--real-only",    action="store_true", help="Filter to data_type='real' only")
    parser.add_argument("--no-sphere",    action="store_true", help="Skip MOE call — stats only")
    parser.add_argument("--output",       default=None,        help="Output path (default: ./data.json)")
    parser.add_argument("--dry-run",      action="store_true", help="Print output, don't write file")
    parser.add_argument("--list-models",  action="store_true", help="Print live Nebius model catalogue and exit")
    args = parser.parse_args()

    if args.list_models:
        load_dotenv()
        client = get_nebius_client()
        print("\nFetching live model list from Nebius Token Factory...\n")
        try:
            models = sorted(m.id for m in client.models.list().data)
            for m in models:
                print(f"  {m}")
            print(f"\n{len(models)} models available.")
            print("\nCurrent MODEL_POOL status:")
            live_set = set(models)
            for code, meta in MODEL_POOL.items():
                status = "✓" if meta["name"] in live_set else "✗ NOT FOUND"
                print(f"  {code}: {meta['name']}  {status}")
        except Exception as e:
            print(f"Error: {e}")
        sys.exit(0)

    run_pipeline(
        real_only   = args.real_only,
        skip_sphere = args.no_sphere,
        dry_run     = args.dry_run,
        output_path = args.output,
    )
