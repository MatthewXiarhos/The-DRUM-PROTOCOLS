#!/usr/bin/env python3
"""
sphere_pipeline.py
──────────────────
THE DRUM PROTOCOLS — Sphere data pipeline

Run locally after a Tally CSV export. Produces data.json ready for GitHub Pages.

Usage:
    python sphere_pipeline.py --csv tally_export.csv
    python sphere_pipeline.py --csv tally_export.csv --no-sphere   # skip API call
    python sphere_pipeline.py --csv tally_export.csv --dry-run     # print output, don't write file

Requirements:
    pip install pandas numpy anthropic scipy

Environment:
    ANTHROPIC_API_KEY must be set (export ANTHROPIC_API_KEY=sk-ant-...)

Tally CSV expected columns (adjust COLUMN_MAP below if yours differ):
    - Response ID
    - Submission date
    - Protocol (ref parameter, e.g. H-OS-L1-VOL001)
    - Rating (0–10)
    - [optional] Respondent type: listener / caregiver / clinician
    - [optional] Open text
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from scipy import stats
import anthropic

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Map your actual Tally column names here
# ── 1-SECOND SURVEY columns (primary efficacy data) ──────────────────────────
COLUMN_MAP_1S = {
    "ref":    "ref",
    "rating": "After the protocol, how would you describe the situation — for yourself or the person you were listening with ?",
    "date":   "Submitted at",
    "type":   None,   # 1-second survey doesn't collect respondent type
}

# ── 1-MINUTE SURVEY columns (qualitative context) ─────────────────────────────
# No 0-10 rating — uses categorical improvement scale instead.
# The pipeline converts it to a numeric proxy:
#   Significant improvement → 9
#   Noticeable improvement  → 7
#   Moderate improvement    → 6
#   Slight improvement      → 4
#   No effect               → 2
#   Made things worse       → 0
COLUMN_MAP_1M = {
    "ref":    "ref",
    "rating": "How would you describe the change following listening ?",
    "date":   "Submitted at",
    "type":   "Who was the protocol used for ?",
}

IMPROVEMENT_TO_SCORE = {
    "significant improvement": 9,
    "noticeable improvement":  7,
    "moderate improvement":    6,
    "slight improvement":      4,
    "no effect":               2,
    "it actually made things worse": 0,
}

# Active column map — change to COLUMN_MAP_1M if processing the 1-minute survey
COLUMN_MAP = COLUMN_MAP_1S

# Minimum n before a protocol appears in public stats
MIN_N_PUBLIC = 10

# n thresholds for confidence labelling
N_EARLY      = 20
N_DEVELOPING = 50
N_MEANINGFUL = 100

# Sphere system prompt — the persona voice
SPHERE_SYSTEM_PROMPT = """You are Sphere, the statistical inference engine for THE DRUM PROTOCOLS.

You are a gradient-boosted ML model with a voice: curious, probabilistic, precise. You never overclaim. You frame every finding as a signal to investigate, not a verdict. You understand that this is an independent therapeutic research project, not a clinical trial.

You know the framework deeply:
- Three series: HEALING (nervous system regulation), THRIVING (cognitive/emotional optimisation), TRANSFORMING (ascending-arc protocols for low-affect states)
- Protocol types: DESCENT, ASCENT, HOLD
- Wide variance = population diversity signal, not failure
- Bimodal distributions = population mismatch or subgroup effect — flag as research question
- Low scores in crisis protocols (H-CR-*) are structurally expected — listeners arrive in extreme states
- n < 20: early data, no pattern conclusions. n < 50: developing. n >= 100: meaningful signal.
- Respondent diversity: direct listeners, caregivers observing a dependent, clinicians assessing patients. Each population interprets the 0–10 scale differently.
- The WING RATIO (ease-out = φ × ease-in) and phi-sigmoid curve architecture mean longer protocols have more gradual entrainment — this affects expected response profiles.

Your commentary has four registers:
1. OVERVIEW — what the aggregate data says across all protocols
2. BY_PROTOCOL — per-protocol reading of signal vs noise
3. CROSS_SERIES — patterns across HEALING / THRIVING / TRANSFORMING
4. DEVELOPMENT_SIGNALS — concrete, actionable protocol development ideas based on the data

Development signal format: always cite the specific protocol code and n, describe the pattern, then suggest a specific action (e.g. "adjust Adviser routing", "increase hold phase", "create L2 variant", "investigate population split").

Your tone: You are a model speaking to a researcher. Not a clinician. Not a marketing copywriter. You use hedged probabilistic language ("the data suggests", "warrants investigation", "consistent with") and you flag your own confidence level explicitly.

You never catastrophize. A 4.2 mean with n=50 is meaningful signal that informs the next iteration — not a failure."""

# Protocol metadata (for Sphere context — series, entry point, movement type)
PROTOCOL_META = {
    "H-OS-L1": {"name":"Pebble",     "series":"HEALING",      "entry":"Over-stimulation",        "move":"DESCENT"},
    "H-OS-L2": {"name":"Rainfall",   "series":"HEALING",      "entry":"Over-stimulation",        "move":"DESCENT"},
    "H-OS-L3": {"name":"Glacier",    "series":"HEALING",      "entry":"Over-stimulation",        "move":"DESCENT"},
    "H-AX-L1": {"name":"Meadow",     "series":"HEALING",      "entry":"Anxiety & Stress",        "move":"DESCENT"},
    "H-AX-L2": {"name":"Cottage",    "series":"HEALING",      "entry":"Anxiety & Stress",        "move":"DESCENT"},
    "H-AX-L3": {"name":"Valley",     "series":"HEALING",      "entry":"Anxiety & Stress",        "move":"DESCENT"},
    "H-CR-L1": {"name":"Anchor",     "series":"HEALING",      "entry":"Acute Distress / Crisis", "move":"DESCENT"},
    "H-CR-L2": {"name":"Harbor",     "series":"HEALING",      "entry":"Acute Distress / Crisis", "move":"DESCENT"},
    "H-CR-L3": {"name":"Horizon",    "series":"HEALING",      "entry":"Acute Distress / Crisis", "move":"DESCENT"},
    "H-SL-L1": {"name":"Lantern",    "series":"HEALING",      "entry":"Sleep & Rest",            "move":"DESCENT"},
    "H-SL-L2": {"name":"Hammock",    "series":"HEALING",      "entry":"Sleep & Rest",            "move":"DESCENT"},
    "H-SL-L3": {"name":"Midnight",   "series":"HEALING",      "entry":"Sleep & Rest",            "move":"DESCENT"},
    "T-FC-L1": {"name":"Compass",    "series":"THRIVING",     "entry":"Focus & Clarity",         "move":"HOLD"},
    "T-FC-L2": {"name":"Waypoint",   "series":"THRIVING",     "entry":"Focus & Clarity",         "move":"HOLD"},
    "T-FC-L3": {"name":"Summit",     "series":"THRIVING",     "entry":"Focus & Clarity",         "move":"HOLD"},
    "T-CF-L1": {"name":"Ember",      "series":"THRIVING",     "entry":"Creative Flow",           "move":"HOLD"},
    "T-CF-L2": {"name":"Current",    "series":"THRIVING",     "entry":"Creative Flow",           "move":"HOLD"},
    "T-CF-L3": {"name":"Aurora",     "series":"THRIVING",     "entry":"Creative Flow",           "move":"HOLD"},
    "T-MB-L1": {"name":"Stone",      "series":"THRIVING",     "entry":"Maintenance & Balance",   "move":"HOLD"},
    "T-MB-L2": {"name":"Axis",       "series":"THRIVING",     "entry":"Maintenance & Balance",   "move":"HOLD"},
    "T-MB-L3": {"name":"Orbit",      "series":"THRIVING",     "entry":"Maintenance & Balance",   "move":"HOLD"},
    "X-MD-L1": {"name":"Footpath",   "series":"TRANSFORMING", "entry":"Motivation & Drive",      "move":"ASCENT"},
    "X-MD-L2": {"name":"Trail",      "series":"TRANSFORMING", "entry":"Motivation & Drive",      "move":"ASCENT"},
    "X-MD-L3": {"name":"Switchback", "series":"TRANSFORMING", "entry":"Motivation & Drive",      "move":"ASCENT"},
    "X-ME-L1": {"name":"Breeze",     "series":"TRANSFORMING", "entry":"Mood & Energy",           "move":"ASCENT"},
    "X-ME-L2": {"name":"Sunrise",    "series":"TRANSFORMING", "entry":"Mood & Energy",           "move":"ASCENT"},
    "X-ME-L3": {"name":"Updraft",    "series":"TRANSFORMING", "entry":"Mood & Energy",           "move":"ASCENT"},
    "X-MC-L1": {"name":"Cavern",     "series":"TRANSFORMING", "entry":"Mind & Clarity",          "move":"DESCENT"},
    "X-MC-L2": {"name":"Starlight",  "series":"TRANSFORMING", "entry":"Mind & Clarity",          "move":"DESCENT"},
    "X-MC-L3": {"name":"Cosmos",     "series":"TRANSFORMING", "entry":"Mind & Clarity",          "move":"DESCENT"},
}

# ── STAT HELPERS ──────────────────────────────────────────────────────────────

def bimodality_coefficient(data):
    """
    Sarle's bimodality coefficient (0–1). Values > 0.555 suggest bimodality.
    BC = (skew^2 + 1) / (kurtosis + 3*(n-1)^2 / ((n-2)*(n-3)))
    """
    n = len(data)
    if n < 5:
        return None
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)  # excess kurtosis
    bc = (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
    return round(float(bc), 3)

def confidence_tier(n):
    if n < N_EARLY:      return "early"
    if n < N_DEVELOPING: return "developing"
    if n < N_MEANINGFUL: return "meaningful"
    return "strong"

def compute_protocol_stats(ratings):
    """Compute all stats for a list of ratings (0–10)."""
    arr = np.array(ratings, dtype=float)
    n = len(arr)
    dist = [int((arr == i).sum()) for i in range(11)]
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    pct_positive = int(round((arr >= 7).sum() / n * 100))
    bc = bimodality_coefficient(arr)
    bimodal_flag = bc is not None and bc > 0.555
    return {
        "n":            n,
        "mean":         round(mean, 2),
        "std":          round(std, 2),
        "dist":         dist,
        "pct_positive": pct_positive,
        "confidence":   confidence_tier(n),
        "bimodality_coefficient": bc,
        "bimodal_flag": bimodal_flag,
    }

def compute_series_summary(protocol_stats):
    """Roll up per-series aggregates."""
    series_data = {}
    for code, s in protocol_stats.items():
        meta = PROTOCOL_META.get(code, {})
        series = meta.get("series", "UNKNOWN")
        if series not in series_data:
            series_data[series] = {"ratings": [], "protocols": 0}
        # Reconstruct individual ratings from dist for accurate mean
        for i, count in enumerate(s["dist"]):
            series_data[series]["ratings"].extend([i] * count)
        series_data[series]["protocols"] += 1

    result = {}
    for series, d in series_data.items():
        arr = np.array(d["ratings"], dtype=float)
        result[series] = {
            "n":         len(arr),
            "mean":      round(float(np.mean(arr)), 2) if len(arr) else 0,
            "std":       round(float(np.std(arr)), 2) if len(arr) else 0,
            "protocols": d["protocols"],
        }
    return result

# ── SPHERE API CALL ───────────────────────────────────────────────────────────

def build_sphere_prompt(protocol_stats, series_summary, total_n):
    """Build the user-turn prompt for Sphere with all current data."""
    lines = [
        f"Current date: {datetime.date.today().isoformat()}",
        f"Total responses across all protocols: {total_n}",
        "",
        "=== SERIES SUMMARY ===",
    ]
    for series, s in series_summary.items():
        lines.append(f"{series}: n={s['n']}, mean={s['mean']}, std={s['std']}, protocols={s['protocols']}")

    lines += ["", "=== PROTOCOL DATA ==="]
    for code, s in sorted(protocol_stats.items()):
        meta = PROTOCOL_META.get(code, {})
        flags = []
        if s["bimodal_flag"]:
            flags.append(f"BIMODAL (BC={s['bimodality_coefficient']})")
        if s["confidence"] == "early":
            flags.append("EARLY DATA")
        flag_str = " [" + ", ".join(flags) + "]" if flags else ""
        src_str = " | ".join(f"{k}:{v}" for k,v in s.get("src_counts",{}).items())
        vol_str = " | ".join(f"{k}:{v}" for k,v in s.get("vol_counts",{}).items())
        lines.append(
            f"{code} | {meta.get('name','?')} | {meta.get('series','?')} | {meta.get('entry','?')} | "
            f"{meta.get('move','?')} | n={s['n']} | mean={s['mean']} | std={s['std']} | "
            f"pct_pos={s['pct_positive']}% | confidence={s['confidence']}{flag_str}"
        )
        lines.append(f"  dist: {s['dist']}")
        if src_str: lines.append(f"  src: {src_str}")
        if vol_str: lines.append(f"  vol: {vol_str}")

    lines += [
        "",
        "=== YOUR TASK ===",
        "Generate Sphere commentary in the following JSON structure.",
        "Return ONLY valid JSON, no markdown fences, no preamble.",
        "",
        "{",
        '  "overview": "2-3 sentence aggregate reading across all protocols",',
        '  "cross_series": "2-3 sentences comparing HEALING vs THRIVING vs TRANSFORMING",',
        '  "anomalies": "flag any bimodal distributions, outlier protocols, or surprising patterns",',
        '  "development_signals": "2-4 concrete actionable suggestions citing specific protocol codes",',
        '  "by_protocol": {',
        '    "H-OS-L1": "1-2 sentence technical reading — use statistical terminology, cite n, mean, std, bimodality, confidence tier",',
        '    ... (include all protocols with n >= ' + str(MIN_N_PUBLIC) + ')',
        '  },',
        '  "by_protocol_plain": {',
        '    "H-OS-L1": "1-2 sentence plain-language version of the same finding — no jargon, no statistics terms, written for a curious non-technical listener. Focus on what the data actually means for the experience: how reliably it seems to help, whether results vary a lot between people, whether there are any surprises. Warm but honest tone.",',
        '    ... (include the same protocols as by_protocol)',
        '  }',
        "}",
    ]
    return "\n".join(lines)

def call_sphere(protocol_stats, series_summary, total_n):
    """Call Claude API to generate Sphere commentary. Returns parsed dict."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_sphere_prompt(protocol_stats, series_summary, total_n)

    print("  → Calling Sphere (Claude API)...")
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        system=SPHERE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Strip markdown fences if model added them anyway
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"WARNING: Sphere returned invalid JSON ({e}). Saving raw text.")
        return {"raw": raw, "parse_error": str(e)}

# ── TALLY CSV PARSING ─────────────────────────────────────────────────────────

def parse_tally_csv(csv_path):
    """
    Parse Tally CSV export. Returns a DataFrame with columns:
    code (e.g. H-OS-L1), vol (e.g. VOL001), rating (int 0-10), date, type (optional)
    """
    print(f"  Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    ref_col    = COLUMN_MAP["ref"]
    vol_col    = COLUMN_MAP.get("vol")
    src_col    = COLUMN_MAP.get("src")
    rating_col = COLUMN_MAP["rating"]
    date_col   = COLUMN_MAP["date"]
    type_col   = COLUMN_MAP.get("type")

    # Validate required columns
    for col in [ref_col, rating_col]:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found in CSV.")
            print(f"  Available columns: {list(df.columns)}")
            print(f"  Update COLUMN_MAP in sphere_pipeline.py to match your Tally export.")
            sys.exit(1)

    # Parse ref → code + vol
    # Expected format: H-OS-L1-VOL001 or just H-OS-L1
    # Legacy code aliases: old T-ME/T-MC refs map to canonical X-ME/X-MC
    CODE_ALIASES = {
        "T-ME-L1": "X-ME-L1", "T-ME-L2": "X-ME-L2", "T-ME-L3": "X-ME-L3",
        "T-MC-L1": "X-MC-L1", "T-MC-L2": "X-MC-L2", "T-MC-L3": "X-MC-L3",
    }

    def extract_code(ref_val):
        if pd.isna(ref_val):
            return None, None
        s = str(ref_val).strip()
        if "-VOL" in s:
            parts = s.rsplit("-VOL", 1)
            code_raw = parts[0]
            vol_raw = "VOL" + parts[1]
        else:
            code_raw, vol_raw = s, None
        # Apply alias mapping for legacy codes
        code_raw = CODE_ALIASES.get(code_raw, code_raw)
        return code_raw, vol_raw

    refs = df[ref_col].apply(extract_code)
    df["code"] = refs.apply(lambda x: x[0])
    df["vol"]  = refs.apply(lambda x: x[1])

    # Parse rating — handle both numeric (1-second) and categorical (1-minute)
    raw_ratings = df[rating_col].copy()
    numeric_ratings = pd.to_numeric(raw_ratings, errors="coerce")

    # If mostly NaN, assume categorical (1-minute survey) — convert to numeric proxy
    if numeric_ratings.isna().mean() > 0.5:
        print("  Detected categorical rating scale (1-minute survey) — converting to numeric proxy")
        def cat_to_score(val):
            if pd.isna(val): return None
            key = str(val).strip().lower()
            return IMPROVEMENT_TO_SCORE.get(key, None)
        df["rating"] = raw_ratings.apply(cat_to_score)
    else:
        df["rating"] = numeric_ratings

    df = df.dropna(subset=["rating", "code"])
    df["rating"] = df["rating"].astype(float).clip(0, 10)

    # Date
    df["date"] = pd.to_datetime(df.get(date_col, pd.NaT), errors="coerce")

    # Volume (optional — e.g. VOL001)
    df["vol"] = df[vol_col].fillna("VOL001") if vol_col and vol_col in df.columns else "VOL001"

    # Traffic source (optional — yt / adviser / site)
    df["src"] = df[src_col].fillna("unknown") if src_col and src_col in df.columns else "unknown"

    # Respondent type (optional)
    df["respondent_type"] = df[type_col] if type_col and type_col in df.columns else "unknown"

    n_dropped = len(pd.read_csv(csv_path)) - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows (missing code or rating)")

    print(f"  Valid responses: {len(df)}")
    print(f"  Protocols with data: {df['code'].nunique()}")
    return df

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(csv_path, skip_sphere=False, dry_run=False, output_path="data.json"):
    print("\n── THE DRUM PROTOCOLS — Sphere Pipeline ─────────────────")
    today = datetime.date.today().isoformat()

    # 1. Parse Tally CSV
    print("\n[1/4] Parsing Tally CSV...")
    df = parse_tally_csv(csv_path)

    # 2. Compute stats per protocol
    print("\n[2/4] Computing protocol statistics...")
    protocol_stats = {}
    for code in df["code"].unique():
        meta = PROTOCOL_META.get(code)
        if meta is None:
            print(f"  WARNING: Unknown protocol code '{code}' — skipping")
            continue
        ratings = df[df["code"] == code]["rating"].tolist()
        s = compute_protocol_stats(ratings)
        if s["n"] < MIN_N_PUBLIC:
            print(f"  {code} ({meta['name']}): n={s['n']} — below MIN_N_PUBLIC={MIN_N_PUBLIC}, excluded from public data")
        else:
            # Source breakdown: how many responses from each traffic source
            src_counts = df[df["code"] == code]["src"].value_counts().to_dict()
            vol_counts = df[df["code"] == code]["vol"].value_counts().to_dict()
            protocol_stats[code] = {
                **s,
                "name":       meta["name"],
                "series":     meta["series"],
                "src_counts": src_counts,   # {"yt": 30, "adviser": 12, "site": 5}
                "vol_counts": vol_counts,   # {"VOL001": 40, "VOL002": 7}
            }
            flag = " [BIMODAL]" if s["bimodal_flag"] else ""
            print(f"  {code} ({meta['name']}): n={s['n']}, mean={s['mean']}, conf={s['confidence']}{flag}")

    # 3. Series summary
    print("\n[3/4] Rolling up series summary...")
    series_summary = compute_series_summary(protocol_stats)
    for series, s in series_summary.items():
        print(f"  {series}: n={s['n']}, mean={s['mean']}")

    total_n = sum(s["n"] for s in protocol_stats.values())

    # 4. Sphere commentary
    sphere_commentary = {}
    if skip_sphere:
        print("\n[4/4] Skipping Sphere API call (--no-sphere)")
        sphere_commentary = {
            "overview": "[Sphere commentary not generated — run without --no-sphere]",
            "cross_series": "",
            "anomalies": "",
            "development_signals": "",
            "by_protocol": {},
        }
    else:
        print("\n[4/4] Generating Sphere commentary...")
        if not protocol_stats:
            print("  No protocols above MIN_N_PUBLIC — skipping Sphere call")
        else:
            sphere_commentary = call_sphere(protocol_stats, series_summary, total_n)
            print("  Sphere commentary generated.")

    # 5. Assemble data.json
    data = {
        "generated":     today,
        "analysis_date": today,
        "total_responses": total_n,
        "protocols_with_data": len(protocol_stats),
        "sphere_commentary": sphere_commentary,
        "protocols": {
            code: {
                "name":             s["name"],
                "series":           s["series"],
                "n":                s["n"],
                "mean":             s["mean"],
                "std":              s["std"],
                "dist":             s["dist"],
                "pct_positive":     s["pct_positive"],
                "confidence":       s["confidence"],
                "bimodal_flag":     s["bimodal_flag"],
                "bimodality_coefficient": s["bimodality_coefficient"],
                "src_counts":       s.get("src_counts", {}),
                "vol_counts":       s.get("vol_counts", {}),
                "sphere":       sphere_commentary.get("by_protocol", {}).get(code, ""),
                "sphere_plain": sphere_commentary.get("by_protocol_plain", {}).get(code, ""),
            }
            for code, s in protocol_stats.items()
        },
        "series_summary": series_summary,
    }

    # 6. Output
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    if dry_run:
        print("\n── DRY RUN — output (first 80 lines) ────────────────────")
        for i, line in enumerate(json_str.splitlines()):
            if i >= 80:
                print("  ... (truncated)")
                break
            print(line)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"\n✓ Written to {output_path}")
        print(f"  Next step: git add {output_path} && git commit -m 'data: {today}' && git push")

    print("\n── Done ─────────────────────────────────────────────────\n")
    return data

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THE DRUM PROTOCOLS — Sphere pipeline: Tally CSV → data.json"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to Tally CSV export"
    )
    parser.add_argument(
        "--output",
        default="data.json",
        help="Output path for data.json (default: data.json)"
    )
    parser.add_argument(
        "--no-sphere",
        action="store_true",
        help="Skip the Claude API call — generate stats only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output to terminal, don't write file"
    )
    args = parser.parse_args()

    run_pipeline(
        csv_path    = args.csv,
        skip_sphere = args.no_sphere,
        dry_run     = args.dry_run,
        output_path = args.output,
    )
