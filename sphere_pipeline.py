#!/usr/bin/env python3
"""
sphere_pipeline.py — THE DRUM PROTOCOLS
──────────────────────────────────────────
Unified pipeline: Supabase → stats → Sphere (Claude API) → data.json

Replaces the old CSV-based sphere_pipeline.py and the stat-only
generate_json.py. Supabase is the single source of truth for all survey
data.

Usage:
  py -3.11 sphere_pipeline.py                          # all data + Sphere LLM call
  py -3.11 sphere_pipeline.py --real-only              # real data + Sphere LLM call
  py -3.11 sphere_pipeline.py --no-sphere              # all data + skip LLM call
  py -3.11 sphere_pipeline.py --dry-run                # print JSON, don't write file
  py -3.11 sphere_pipeline.py --real-only --no-sphere

Requirements:
  pip install supabase anthropic scipy numpy python-dotenv openai

Environment (.env or GitHub secrets):
  SUPABASE_URL          — Supabase project URL
  SUPABASE_SERVICE_KEY  — service_role key (bypasses RLS)
  ANTHROPIC_API_KEY     — Claude API key (not needed with --no-sphere)
  OPENAI_VECTOR_KEY     — OpenAI API key for text-embedding-3-small (not needed with --no-sphere)

Output:
  ./data.json — matches GitHub repo root; upload directly

Confidence scale (matches sphere.html CONF_PCT):
  early       n < 5    22%
  developing  n 5–19   55%
  meaningful  n 20–49  80%
  strong      n >= 50  95%
"""

import os
import sys
import json
import math
import argparse
import datetime
from datetime import timezone

import numpy as np
from scipy import stats as scipy_stats
from dotenv import load_dotenv
from supabase import create_client
import hashlib
import anthropic
from openai import OpenAI

# ── CONFIGURATION ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "."        # data.json writes to project root — matches GitHub Pages expectation
PROMPTS_DIR = "prompts"  # versioned prompt markdown files

# Confidence thresholds — must match sphere.html CONF_PCT
N_DEVELOPING = 5
N_MEANINGFUL = 20
N_STRONG     = 50

# Sarle's BC threshold for bimodality
BIMODALITY_BC_THRESHOLD = 0.555

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMS  = 1536

# Protocol metadata — movement type (DESCENT / HOLD / ASCENT) used in Sphere prompt
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

# ── PROMPT LOADING ────────────────────────────────────────────────────────────────────

def _strip_frontmatter(text):
    """Strip YAML frontmatter (--- … ---) from a markdown prompt file."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text

def load_prompt(filename):
    """
    Load a prompt from the prompts/ directory, stripping YAML frontmatter.
    Returns (prompt_text, content_hash, filepath).
    Exits with error if file not found.
    """
    filepath = os.path.join(PROMPTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"ERROR: Prompt file not found: {filepath}")
        sys.exit(1)
    raw          = open(filepath, encoding="utf-8").read()
    content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    prompt_text  = _strip_frontmatter(raw)
    return prompt_text, content_hash, filepath

# Active prompt filenames — change these to switch versions
SPHERE_SYSTEM_FILE = "sphere__system__v1.md"
SPHERE_USER_FILE   = "sphere__user__v1.md"

# ── STAT HELPERS ──────────────────────────────────────────────────────────────────────

def bimodality_coefficient(data):
    """
    Sarle's bimodality coefficient (0–1). Values > 0.555 suggest bimodality.
    BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
    Returns None if n < 5.
    """
    n = len(data)
    if n < 5:
        return None
    skew = scipy_stats.skew(data)
    kurt = scipy_stats.kurtosis(data)  # excess kurtosis
    bc   = (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
    return round(float(bc), 3)

def score_dist(scores):
    """11-element list: count of scores at each integer 0–10."""
    counts = [0] * 11
    for s in scores:
        if s is not None:
            idx = max(0, min(10, int(round(s))))
            counts[idx] += 1
    return counts

def pct_positive(dist):
    """Percentage of responses scoring 7 or above."""
    total = sum(dist)
    if total == 0:
        return 0
    return round(sum(dist[7:]) / total * 100)

def confidence_level(n):
    """Maps to sphere.html CONF_PCT thresholds."""
    if n >= N_STRONG:     return "strong"
    if n >= N_MEANINGFUL: return "meaningful"
    if n >= N_DEVELOPING: return "developing"
    return "early"

def compute_protocol_stats(scores):
    """
    Given a list of outcome scores (0–10 floats), return a stats dict.
    Uses numpy/scipy for accuracy (Sarle's BC).
    """
    valid = [s for s in scores if s is not None]
    n     = len(valid)
    if n == 0:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "dist": [0]*11,
            "pct_positive": 0, "confidence": "early",
            "bimodality_coefficient": None, "bimodal_flag": False,
        }

    arr   = np.array(valid, dtype=float)
    dist  = score_dist(valid)
    mean  = round(float(np.mean(arr)), 2)
    std   = round(float(np.std(arr, ddof=1)) if n > 1 else 0.0, 2)
    pct   = pct_positive(dist)
    bc    = bimodality_coefficient(arr)
    bimod = bc is not None and bc > BIMODALITY_BC_THRESHOLD

    return {
        "n":                     n,
        "mean":                  mean,
        "std":                   std,
        "dist":                  dist,
        "pct_positive":          pct,
        "confidence":            confidence_level(n),
        "bimodality_coefficient": bc,
        "bimodal_flag":          bimod,
    }

def compute_series_summary(protocols_out, short_by_protocol):
    """
    Roll up per-series aggregates from raw score lists (accurate mean/std).
    """
    series_scores         = {}
    series_protocol_count = {}

    for code, p in protocols_out.items():
        series = p["series"]
        rows   = short_by_protocol.get(code, [])
        scores = [r["outcome_score"] for r in rows if r["outcome_score"] is not None]
        series_scores.setdefault(series, []).extend(scores)
        series_protocol_count[series] = series_protocol_count.get(series, 0) + 1

    result = {}
    for series, scores in series_scores.items():
        arr = np.array(scores, dtype=float) if scores else np.array([])
        result[series] = {
            "n":         len(arr),
            "mean":      round(float(np.mean(arr)), 2) if len(arr) else 0.0,
            "std":       round(float(np.std(arr, ddof=1)), 2) if len(arr) > 1 else 0.0,
            "protocols": series_protocol_count.get(series, 0),
        }
    return result

def build_full_survey_context(full_rows, protocols_out):
    """
    Summarise full survey (1-minute) responses per protocol into a compact
    text block for the Sphere prompt.
    """
    from collections import Counter

    full_by_protocol = {}
    for r in full_rows:
        full_by_protocol.setdefault(r["protocol_code"], []).append(r)

    lines = []
    for code in sorted(protocols_out.keys()):
        rows = full_by_protocol.get(code, [])
        if not rows:
            continue
        n = len(rows)

        def top(field, top_n=3):
            vals = [r.get(field) for r in rows if r.get(field)]
            if not vals:
                return "n/a"
            counts = Counter(vals)
            return " / ".join(f"{v}({c})" for v, c in counts.most_common(top_n))

        listener_counts = Counter(r.get("listener_type") for r in rows if r.get("listener_type"))
        listener_str    = " | ".join(f"{k}:{v}" for k, v in listener_counts.most_common()) or "n/a"

        lines.append(
            f"{code} | n={n} | "
            f"listener_type: {listener_str} | "
            f"change_rating: {top('change_rating')} | "
            f"settle_time: {top('settle_time', 2)} | "
            f"activity: {top('activity', 2)} | "
            f"music_opinion: {top('music_opinion', 2)} | "
            f"rhythm_opinion: {top('rhythm_opinion', 2)}"
        )

    return "\n".join(lines) if lines else "No 1-minute survey responses available yet."


# ── YOUTUBE CONTEXT ───────────────────────────────────────────────────────────────────

def fetch_youtube_context(supabase, protocols_out) -> dict:
    """
    Fetch the most recent youtube_daily row per protocol_code.
    Returns {protocol_code: {views, avg_view_duration, avg_view_percentage}}.
    Only includes protocols that have at least one youtube_daily row.
    """
    try:
        rows = supabase.table("youtube_daily").select(
            "protocol_code, report_date, views, avg_view_duration, avg_view_percentage"
        ).execute().data

        if not rows:
            return {}

        # Keep only the most recent row per protocol_code
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
    """
    Build the {{INJECT:YOUTUBE_CONTEXT}} block for the Sphere prompt.
    """
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
        date     = yt.get("report_date", "")

        lines.append(
            f"{code} | views={views} | avg_view_duration={duration} | "
            f"avg_view_percentage={pct} | as_of={date}"
        )

    return "\n".join(lines) if lines else "No YouTube data available yet."


# ── SPHERE PROMPT BUILDER ─────────────────────────────────────────────────────────────

def build_sphere_prompt(protocols_out, series_summary, total_n,
                        total_n_full=0, full_rows=None, youtube_data=None):
    """
    Build the user-turn prompt by loading sphere__user__v1.md and rendering
    the {{INJECT:*}} placeholders with live data.
    """
    template, _, _ = load_prompt(SPHERE_USER_FILE)

    # ── Series summary block ───────────────────────────────────────────────
    series_lines = []
    for series, s in series_summary.items():
        series_lines.append(
            f"{series}: n={s['n']}, mean={s['mean']}, std={s['std']}, protocols={s['protocols']}"
        )

    # ── Protocol data block ────────────────────────────────────────────────
    proto_lines = []
    for code, p in sorted(protocols_out.items()):
        meta    = PROTOCOL_META.get(code, {})
        bc      = p.get("bimodality_coefficient")
        flags   = []
        if p["bimodal_flag"]:
            flags.append(f"BIMODAL (BC={bc})")
        if p["confidence"] == "early":
            flags.append("EARLY DATA")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""

        src_str = " | ".join(f"{k}:{v}" for k, v in p.get("src_counts", {}).items())
        vol_str = " | ".join(f"{k}:{v}" for k, v in p.get("vol_counts", {}).items())

        proto_lines.append(
            f"{code} | {meta.get('name','?')} | {meta.get('series','?')} | "
            f"{meta.get('entry','?')} | {meta.get('move','?')} | "
            f"n={p['n']} | mean={p['mean']} | std={p['std']} | "
            f"pct_pos={p['pct_positive']}% | confidence={p['confidence']}{flag_str}"
        )
        proto_lines.append(f"  dist: {p['dist']}")
        if src_str:
            proto_lines.append(f"  src: {src_str}")
        if vol_str:
            proto_lines.append(f"  vol: {vol_str}")

    # ── YouTube context block ──────────────────────────────────────────────
    youtube_block = build_youtube_context_block(youtube_data or {}, protocols_out)

    # ── Render template ────────────────────────────────────────────────────
    rendered = template
    rendered = rendered.replace("{{INJECT:TODAY}}",              datetime.date.today().isoformat())
    rendered = rendered.replace("{{INJECT:TOTAL_N_SHORT}}",      str(total_n))
    rendered = rendered.replace("{{INJECT:TOTAL_N_FULL}}",       str(total_n_full))
    rendered = rendered.replace("{{INJECT:SERIES_SUMMARY}}",     "\n".join(series_lines))
    rendered = rendered.replace("{{INJECT:PROTOCOL_DATA}}",      "\n".join(proto_lines))
    rendered = rendered.replace("{{INJECT:YOUTUBE_CONTEXT}}",    youtube_block)

    full_context = build_full_survey_context(full_rows or [], protocols_out)
    rendered = rendered.replace("{{INJECT:FULL_SURVEY_CONTEXT}}", full_context)

    return rendered


# ── SPHERE API CALL ───────────────────────────────────────────────────────────────────

# Anthropic pricing for claude-sonnet-4-6 ($/million tokens)
SONNET_INPUT_COST_PER_MTOK  = 3.00
SONNET_OUTPUT_COST_PER_MTOK = 15.00

def call_sphere(protocols_out, series_summary, total_n,
                total_n_full=0, full_rows=None, youtube_data=None):
    """
    Call Claude API to generate Sphere commentary.
    Returns (commentary_dict, usage_dict, prompt_versions_info).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    system_text, system_hash, system_path = load_prompt(SPHERE_SYSTEM_FILE)
    prompt                                = build_sphere_prompt(
        protocols_out, series_summary, total_n, total_n_full,
        full_rows=full_rows, youtube_data=youtube_data
    )
    _, user_hash, user_path = load_prompt(SPHERE_USER_FILE)

    prompt_versions_info = [
        {"filename": SPHERE_SYSTEM_FILE, "content_hash": system_hash},
        {"filename": SPHERE_USER_FILE,   "content_hash": user_hash},
    ]

    client = anthropic.Anthropic(api_key=api_key)

    print("  → Calling Sphere (Claude API)...")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=system_text,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    usage = {
        "input_tokens":  message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "cost_usd":      round(
            message.usage.input_tokens  / 1_000_000 * SONNET_INPUT_COST_PER_MTOK +
            message.usage.output_tokens / 1_000_000 * SONNET_OUTPUT_COST_PER_MTOK,
            6
        ),
    }
    print(f"  → Tokens: {usage['input_tokens']} in / {usage['output_tokens']} out  "
          f"(~${usage['cost_usd']:.4f})")

    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    raw = raw.strip()

    try:
        commentary = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"WARNING: Sphere returned invalid JSON ({e}). Saving raw text.")
        commentary = {"raw": raw, "parse_error": str(e)}

    return commentary, usage, prompt_versions_info


# ── SUPABASE WRITE-BACK ───────────────────────────────────────────────────────────────

def mode_value(values):
    """Return the most common non-null value in a list, or None."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return max(set(clean), key=clean.count)

def writeback_metrics(supabase, protocols_out, short_by_protocol, full_rows, run_date):
    now = datetime.datetime.now(timezone.utc).isoformat()

    full_by_protocol = {}
    for r in full_rows:
        full_by_protocol.setdefault(r["protocol_code"], []).append(r)

    print("  → Upserting protocol_metrics...")
    proto_rows = []
    for code, p in protocols_out.items():
        full       = full_by_protocol.get(code, [])
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
        for vol_code, vol_n in p.get("vol_counts", {}).items():
            if vol_code == "unknown":
                continue
            pvid       = f"{code}-{vol_code}"
            vol_scores = [
                r["outcome_score"]
                for r in short_by_protocol.get(code, [])
                if r.get("volume_code") == vol_code and r["outcome_score"] is not None
            ]
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
    vol_scores_map = {}
    vol_full_map   = {}
    for code, p in protocols_out.items():
        for r in short_by_protocol.get(code, []):
            vol = r.get("volume_code") or "unknown"
            if vol == "unknown":
                continue
            vol_scores_map.setdefault(vol, [])
            if r["outcome_score"] is not None:
                vol_scores_map[vol].append(r["outcome_score"])
        for r in full_by_protocol.get(code, []):
            vol = r.get("volume_code") or "unknown"
            if vol == "unknown":
                continue
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

    texts   = []
    indices = []
    for i, row in enumerate(rows):
        text = row.get("output_text", "")
        if text:
            texts.append(text)
            indices.append(i)

    if not texts:
        return 0

    client = OpenAI(api_key=api_key)
    print(f"  → Generating {len(texts)} embeddings (text-embedding-3-small)...")

    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
        dimensions=OPENAI_EMBEDDING_DIMS,
    )

    for result, idx in zip(response.data, indices):
        rows[idx]["embedding"] = result.embedding

    cost_usd = round(response.usage.total_tokens / 1_000_000 * 0.02, 6)
    print(f"    ✓ {len(texts)} embeddings generated  "
          f"({response.usage.total_tokens} tokens, ~${cost_usd:.6f})")
    return len(texts)

def writeback_llm_outputs(supabase, protocols_out, sphere_commentary,
                          run_date, model_used, prompt_version="v1"):
    now  = datetime.datetime.now(timezone.utc).isoformat()
    rows = []

    for scope_key in ("overview", "cross_series", "anomalies", "development_signals"):
        text = sphere_commentary.get(scope_key, "")
        if not text:
            continue
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
        if not tech_text and not plain_text:
            continue
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

def writeback_prompt_versions(supabase, prompt_versions_info, run_date):
    now  = datetime.datetime.now(timezone.utc).isoformat()
    rows = []
    for info in prompt_versions_info:
        filename = info["filename"]
        parts    = filename.replace(".md", "").split("__")
        task_type      = parts[0] if len(parts) > 0 else "sphere"
        analysis_scope = parts[1] if len(parts) > 1 else "system"
        version        = parts[2] if len(parts) > 2 else "v1"
        filepath       = os.path.join(PROMPTS_DIR, filename)
        full_content   = open(filepath, encoding="utf-8").read()
        record_id      = f"{task_type}__{analysis_scope}__{version}"
        rows.append({
            "id":             record_id,
            "version":        version,
            "task_type":      task_type,
            "analysis_scope": analysis_scope,
            "prompt_text":    full_content,
            "is_active":      True,
            "created_at":     now,
        })
    if rows:
        supabase.table("prompt_versions").upsert(rows, on_conflict="id").execute()
        print(f"    ✓ prompt_versions logged ({len(rows)} prompts: "
              + ",".join(r['id'] for r in rows) + ")")

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
    today       = datetime.date.fromisoformat(run_date)
    month_start = today.replace(day=1).isoformat()
    existing    = supabase.table("budget_tracking") \
        .select("llm_cost_usd") \
        .gte("record_date", month_start) \
        .execute().data
    monthly_so_far = sum(r["llm_cost_usd"] or 0 for r in existing)
    monthly_total  = round(monthly_so_far + cost_usd, 6)

    days_elapsed  = today.day
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    forecast      = round(monthly_total / days_elapsed * days_in_month, 4) if days_elapsed else monthly_total

    budget_state = "healthy"
    if forecast > 50 * 0.9: budget_state = "caution"
    if forecast > 50:        budget_state = "over_budget"

    supabase.table("budget_tracking").upsert({
        "id":                    f"{run_date}-sphere",
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
    run_start      = _time.time()
    print("── THE DRUM PROTOCOLS — Sphere Pipeline (unified) ────────────")
    today          = datetime.date.today().isoformat()
    generated_iso  = datetime.datetime.now(timezone.utc).isoformat()
    run_id         = f"{today}-sphere-{'real' if real_only else 'all'}"
    mode_label     = "REAL DATA ONLY" if real_only else "ALL DATA (test + real)"
    usage          = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    run_status     = "success"
    run_error      = None

    print(f"  Mode    : {mode_label}")
    print(f"  Sphere  : {'SKIP (--no-sphere)' if skip_sphere else 'ENABLED'}")
    print()

    # ── 1. Connect to Supabase ─────────────────────────────────────────────
    load_dotenv()
    supa_url = os.environ.get("SUPABASE_URL")
    supa_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not supa_url or not supa_key:
        print("ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env / environment.")
        sys.exit(1)

    supabase = create_client(supa_url, supa_key)

    # ── 2. Load registry ───────────────────────────────────────────────────
    print("[1/5] Loading registry from Supabase...")
    protocols_raw     = supabase.table("protocol_registry").select("*").execute().data
    volumes_raw       = supabase.table("volumes").select("*").execute().data
    protocols_by_code = {p["protocol_code"]: p for p in protocols_raw}
    print(f"  {len(protocols_raw)} protocols, {len(volumes_raw)} volumes")

    # ── 3. Load survey responses ───────────────────────────────────────────
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
    full_rows = full_q.execute().data
    print(f"  full survey  : {len(full_rows)} rows")

    contains_test = any(r.get("data_type") == "test" for r in short_rows)

    short_by_protocol = {}
    for r in short_rows:
        short_by_protocol.setdefault(r["protocol_code"], []).append(r)

    # ── 4. Load YouTube context ────────────────────────────────────────────
    print("\n[3/5] Loading YouTube data from Supabase...")
    youtube_data = fetch_youtube_context(supabase, {})
    yt_count     = sum(1 for v in youtube_data.values() if v.get("views") is not None)
    print(f"  {len(youtube_data)} protocols with YouTube data ({yt_count} with view counts)")

    # ── 5. Compute per-protocol stats ──────────────────────────────────────
    print("\n[4/5] Computing protocol statistics...")

    protocols_out = {}
    for code, p in protocols_by_code.items():
        rows   = short_by_protocol.get(code, [])
        scores = [r["outcome_score"] for r in rows if r["outcome_score"] is not None]
        s      = compute_protocol_stats(scores)

        src_counts = {}
        vol_counts = {}
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
            "name":                   name,
            "series":                 series,
            "entry":                  entry,
            "n":                      s["n"],
            "mean":                   s["mean"],
            "std":                    s["std"],
            "dist":                   s["dist"],
            "pct_positive":           s["pct_positive"],
            "confidence":             s["confidence"],
            "bimodal_flag":           s["bimodal_flag"],
            "bimodality_coefficient": s["bimodality_coefficient"],
            "sphere":                 "",
            "sphere_plain":           "",
            "src_counts":             src_counts,
            "vol_counts":             vol_counts,
        }

    series_summary = compute_series_summary(protocols_out, short_by_protocol)

    # ── 6. Sphere commentary ───────────────────────────────────────────────
    print()
    total_n = sum(p["n"] for p in protocols_out.values())

    if skip_sphere:
        print("[5/5] Skipping Sphere API call (--no-sphere)")
        sphere_commentary = {
            "overview":            "[Sphere commentary not generated — run without --no-sphere]",
            "cross_series":        "",
            "anomalies":           "",
            "development_signals": "",
            "by_protocol":         {code: "" for code in protocols_out},
            "by_protocol_plain":   {code: "" for code in protocols_out},
        }
    else:
        print("[5/5] Generating Sphere commentary...")
        sphere_commentary, usage, prompt_versions_info = call_sphere(
            protocols_out, series_summary, total_n, total_n_full=len(full_rows),
            full_rows=full_rows, youtube_data=youtube_data
        )

        by_protocol       = sphere_commentary.get("by_protocol", {})
        by_protocol_plain = sphere_commentary.get("by_protocol_plain", {})
        for code in protocols_out:
            protocols_out[code]["sphere"]       = by_protocol.get(code, "")
            protocols_out[code]["sphere_plain"] = by_protocol_plain.get(code, "")

        print("  Sphere commentary generated.")

    # ── 7. Assemble data.json ──────────────────────────────────────────────
    protocols_with_data = sum(1 for p in protocols_out.values() if p["n"] > 0)

    def public_protocol(p):
        return {k: v for k, v in p.items()
                if k not in ("src_counts", "vol_counts", "bimodality_coefficient")}

    data_json = {
        "_meta": {
            "generated_iso":     generated_iso,
            "mode":              "real_only" if real_only else "all",
            "contains_test_data": contains_test,
            "total_responses":   total_n,
        },
        "generated":           today,
        "analysis_date":       today,
        "protocols_with_data": protocols_with_data,
        "protocols":           {code: public_protocol(p) for code, p in protocols_out.items()},
        "series_summary":      series_summary,
        "sphere_commentary":   {
            "overview":            sphere_commentary.get("overview", ""),
            "cross_series":        sphere_commentary.get("cross_series", ""),
            "anomalies":           sphere_commentary.get("anomalies", ""),
            "development_signals": sphere_commentary.get("development_signals", ""),
            "by_protocol":         sphere_commentary.get("by_protocol",       {code: "" for code in protocols_out}),
            "by_protocol_plain":   sphere_commentary.get("by_protocol_plain", {code: "" for code in protocols_out}),
        },
    }

    # ── 8. Output ──────────────────────────────────────────────────────────
    json_str = json.dumps(data_json, indent=2, ensure_ascii=False)

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "data.json")

    if dry_run:
        print("\n── DRY RUN — output (first 80 lines) ────────────────────────")
        for i, line in enumerate(json_str.splitlines()):
            if i >= 80:
                print("  ... (truncated)")
                break
            print(line)
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"\n  ✓ Written to {output_path}  ({size_kb:.1f} KB)")

    # ── 9. Write-back to Supabase ──────────────────────────────────────────
    if not dry_run:
        print("\n[5/5] Writing results back to Supabase...")
        try:
            writeback_metrics(supabase, protocols_out, short_by_protocol, full_rows, today)

            if not skip_sphere and not sphere_commentary.get("parse_error"):
                writeback_llm_outputs(
                    supabase, protocols_out, sphere_commentary,
                    run_date=today, model_used="claude-sonnet-4-6",
                    prompt_version=SPHERE_SYSTEM_FILE.replace(".md", "")
                )
                writeback_prompt_versions(supabase, prompt_versions_info, today)
                writeback_budget(
                    supabase, today,
                    usage["input_tokens"], usage["output_tokens"], usage["cost_usd"]
                )
        except Exception as e:
            run_status = "partial"
            run_error  = f"write-back error: {e}"
            print(f"  WARNING: write-back failed — {e}")

    # ── 10. Log pipeline run ───────────────────────────────────────────────
    if not dry_run:
        import time as _time2
        duration = _time2.time() - run_start
        tasks    = ["load_registry", "load_surveys", "load_youtube", "compute_stats"]
        if not skip_sphere:
            tasks += ["sphere_llm", "writeback_llm_outputs", "writeback_budget"]
        tasks += ["writeback_metrics", "write_data_json"]
        try:
            writeback_pipeline_run(
                supabase, run_id, today,
                run_type="sphere_full" if not skip_sphere else "sphere_stats_only",
                tasks=tasks, status=run_status, duration_seconds=duration,
                error_message=run_error,
            )
        except Exception as e:
            print(f"  WARNING: pipeline_runs logging failed — {e}")

    # ── 11. Summary report ─────────────────────────────────────────────────
    SERIES_ORDER = {"HEALING": 0, "THRIVING": 1, "TRANSFORMING": 2}
    print()
    print("=" * 60)
    print(f"  Mode              : {mode_label}")
    print(f"  Contains test data: {contains_test}")
    print(f"  Total responses   : {total_n}")
    print(f"  Protocols w/ data : {protocols_with_data} / {len(protocols_out)}")
    if not skip_sphere:
        print(f"  LLM cost this run : ${usage['cost_usd']:.4f}  "
              f"({usage['input_tokens']} in / {usage['output_tokens']} out tokens)")
    print()
    print("  Series summary:")
    for series, s in series_summary.items():
        print(f"    {series:<14} n={s['n']:<5} mean={s['mean']:.2f}  std={s['std']:.2f}  protocols={s['protocols']}")
    print()
    print("  Protocol breakdown (n / mean / confidence):")
    for code, p in sorted(protocols_out.items(),
                          key=lambda x: (SERIES_ORDER.get(x[1]["series"], 9), x[0])):
        bimod = " [BIMODAL]" if p["bimodal_flag"] else ""
        print(f"    {code:<12} n={p['n']:<4} mean={p['mean']:.1f}  {p['confidence']}{bimod}")
    print()
    if not dry_run:
        print("  Next step: upload data.json to your GitHub repo root.")
    print("=" * 60)
    print()

    return data_json


# ── CLI ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THE DRUM PROTOCOLS — unified Sphere pipeline: Supabase → stats → LLM → data.json"
    )
    parser.add_argument("--real-only", action="store_true",
                        help="Filter to data_type = 'real' only")
    parser.add_argument("--no-sphere", action="store_true",
                        help="Skip the Claude API call — generate stats only")
    parser.add_argument("--output",    default=None,
                        help="Output path for data.json (default: ./data.json)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print output to terminal, don't write file")
    args = parser.parse_args()

    run_pipeline(
        real_only   = args.real_only,
        skip_sphere = args.no_sphere,
        dry_run     = args.dry_run,
        output_path = args.output,
    )