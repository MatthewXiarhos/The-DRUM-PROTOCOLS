#!/usr/bin/env python3
"""
tally_ingestion.py — THE DRUM PROTOCOLS
──────────────────────────────────────────
Fetches Tally form submissions via REST API and upserts to Supabase.
Replaces manual CSV export + ingest_short_survey.py / ingest_full_survey.py.

DATA TYPE LOGIC:
  data field blank or absent → data_type = 'real'   (live listener URL has no data param)
  data field = 'test'        → data_type = 'test'   (test pipeline URL has data=test)

PVID CONSTRUCTION:
  ref = 'H-OS-L1' + vol = 'VOL001'   → pvid = 'H-OS-L1-VOL001'
  ref = 'H-OS-L1-VOL001' (full pvid) → use directly, ignore vol
  ref blank                           → pvid = None (unattributable, still ingested)

HIDDEN FIELD RESOLUTION:
  Tally REST API omits hidden fields from responses[] when their value is null.
  We map by UUID directly (UUID_FIELD_MAP) — confirmed from --inspect-submission.
  UUIDs are permanent and never change unless the form is rebuilt from scratch.

Usage:
  python tally_ingestion.py                     # ingest all forms
  python tally_ingestion.py --dry-run           # print what would be upserted, don't write
  python tally_ingestion.py --form 1AMk7b       # single form only
  python tally_ingestion.py --since 2026-04-01  # submissions on/after this date only
  python tally_ingestion.py --inspect-submission 1AMk7b
  python tally_ingestion.py --inspect-submission 1AMk7b --submission-id q54NvDk

Requirements:
  pip install supabase requests python-dotenv

Environment (.env or GitHub secrets):
  SUPABASE_URL          — Supabase project URL
  SUPABASE_SERVICE_KEY  — service_role key (bypasses RLS)
  TALLY_API_KEY         — Tally API key (Settings → API keys)
"""

import os
import sys
import argparse
import time
import requests
from dotenv import load_dotenv
from supabase import create_client

# ── FORM REGISTRY ─────────────────────────────────────────────────────────────

FORMS = [
    {"form_id": "1AMk7b", "survey_type": "short", "name": "1 SECOND SURVEY"},
    {"form_id": "rjKPY5", "survey_type": "full",  "name": "1 MINUTE SURVEY"},
]

TALLY_BASE_URL = "https://api.tally.so"
PAGE_SIZE      = 100
RETRY_WAIT     = 10

# ── UUID → COLUMN MAP ─────────────────────────────────────────────────────────
# Tally omits hidden fields from the API response when their value is null.
# We map by UUID directly. These are permanent field identifiers.
#
# src and data UUIDs still needed — run these to find them:
#   python tally_ingestion.py --inspect-submission 1AMk7b --submission-id q54NvDk
#   python tally_ingestion.py --inspect-submission 1AMk7b --submission-id laayAao
#   python tally_ingestion.py --inspect-submission rjKPY5 --submission-id xVXJ6X5

UUID_FIELD_MAP = {
    # 1 SECOND SURVEY (1AMk7b) — hidden fields at top of form
    "dYLLMr": "ref",
    "1KA9lW": "vol",
    "D1kMLb": "src",
    "R40vVv": "data",
    # survey answer field
    "YZEE6N": "outcome_score",

    # 1 MINUTE SURVEY (rjKPY5) — hidden fields at bottom of form
    "7d80ya": "ref",
    "8k1Zdx": "vol",
    "oBbG9N": "src",
    "OPzdrM": "data",
    # survey answer fields
    "lN5Qzv": "state_before",
    "Zdgpae": "state_after",
    "Rzgpjj": "change_rating",
    "NAgeoO": "settle_time",
    "qbp2Vg": "activity",
    "QAX6VX": "listen_method",
    "9dazQ5": "listen_frequency",
    "eBpdRl": "music_opinion",
    "WAgvzk": "rhythm_opinion",
    "zKOala": "listener_type",
    # open_feedback and future_styles UUIDs not yet seen — title matching handles them
}

# ── SURVEY ANSWER FIELD MAPS ──────────────────────────────────────────────────
# Tally question titles already match Supabase column names exactly.

FIELD_MAP_SHORT = {
    "outcome_score": "outcome_score",
}

FIELD_MAP_FULL = {
    "state_before":     "state_before",
    "state_after":      "state_after",
    "change_rating":    "change_rating",
    "settle_time":      "settle_time",
    "activity":         "activity",
    "listen_method":    "listen_method",
    "listen_frequency": "listen_frequency",
    "music_opinion":    "music_opinion",
    "rhythm_opinion":   "rhythm_opinion",
    "listener_type":    "listener_type",
    "open_feedback":    "open_feedback",
    # future_styles has no Supabase column yet — silently ignored
}

FLOAT_COLUMNS = {"outcome_score"}


# ── TALLY API CLIENT ──────────────────────────────────────────────────────────

class TallyClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._req_count  = 0
        self._window_start = time.time()

    def _rate_check(self):
        self._req_count += 1
        if self._req_count >= 90:
            elapsed = time.time() - self._window_start
            if elapsed < 60:
                print(f"  Rate limit guard: sleeping {61 - elapsed:.1f}s...")
                time.sleep(61 - elapsed)
            self._req_count    = 0
            self._window_start = time.time()

    def get(self, path: str, params: dict = None) -> dict:
        self._rate_check()
        resp = self.session.get(f"{TALLY_BASE_URL}{path}", params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  Rate limited — waiting {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
            return self.get(path, params)
        resp.raise_for_status()
        return resp.json()

    def get_form_schema(self, form_id: str) -> tuple[dict, list]:
        """Questions live on the submissions endpoint, not the form endpoint."""
        data      = self.get(f"/forms/{form_id}/submissions", params={"page": 1, "limit": 1})
        form_info = self.get(f"/forms/{form_id}")
        return form_info, data.get("questions", [])

    def get_submissions_page(self, form_id: str, page: int) -> dict:
        return self.get(
            f"/forms/{form_id}/submissions",
            params={"page": page, "limit": PAGE_SIZE},
        )

    def get_submission(self, form_id: str, submission_id: str) -> dict:
        return self.get(f"/forms/{form_id}/submissions/{submission_id}")

    def get_all_submissions(self, form_id: str, since: str = None) -> list[dict]:
        """Paginate all completed submissions. Stops early if since date provided."""
        all_subs = []
        page     = 1
        while True:
            data = self.get_submissions_page(form_id, page)
            subs = data.get("submissions", [])
            if not subs:
                break
            for sub in subs:
                # Skip partial submissions
                if not sub.get("isCompleted", True):
                    continue
                submitted = sub.get("submittedAt", "") or ""
                if since and submitted[:10] < since:
                    return all_subs   # newest-first — safe to stop here
                all_subs.append(sub)
            if not data.get("hasMore"):
                break
            page += 1
        return all_subs


# ── FIELD HELPERS ─────────────────────────────────────────────────────────────

def normalise(s) -> str:
    return " ".join(str(s or "").lower().split()).strip("?:.")

def extract_value(answer) -> str | None:
    """
    Handle all Tally API answer shapes:
      {'ref': 'T-FC-L3'}          — hidden field: dict keyed by field name
      ['Noticeable improvement']    — multiple choice: single-item list
      ['opt1', 'opt2']             — multi-select: list
      10                            — linear scale: number
      'some text'                   — textarea: string
      None                          — unanswered
    """
    if answer is None:
        return None
    # Hidden field dict e.g. {'ref': 'T-FC-L3'} or {'data': ''}
    if isinstance(answer, dict):
        values = [str(v) for v in answer.values() if v is not None and str(v).strip()]
        return values[0] if len(values) == 1 else (", ".join(values) if values else None)
    # List: multiple choice or multi-select — unwrap single items
    if isinstance(answer, list):
        clean = [str(v).strip() for v in answer if v is not None and str(v).strip()]
        return clean[0] if len(clean) == 1 else (", ".join(clean) if clean else None)
    # Scalar: number or string
    val = str(answer).strip()
    return val if val else None

def build_question_index(questions: list) -> dict:
    """
    {fieldUuid: (normalised_title, field_type)}
    UUID_FIELD_MAP entries take priority over title-derived entries.
    """
    index = {uuid: (col, "HIDDEN_FIELD") for uuid, col in UUID_FIELD_MAP.items()}
    for q in questions:
        qid      = q.get("id", "")
        qtype    = q.get("type", "")
        raw_title = q.get("title") or ""
        for field in q.get("fields", []):
            fid = field.get("uuid", qid)
            if fid not in index:
                ftitle     = field.get("title") or raw_title or ""
                index[fid] = (normalise(ftitle), field.get("type", ""))
        if qid not in index:
            best = raw_title or (q["fields"][0].get("title", "") if q.get("fields") else "")
            index[qid] = (normalise(best), qtype)
    return index


# ── DATA TYPE LOGIC ───────────────────────────────────────────────────────────

def determine_data_type(data_field_value: str | None) -> str:
    """
    blank / None → 'real'   (live URL has no data param)
    'test'       → 'test'   (test pipeline URL has data=test)
    """
    if not data_field_value or data_field_value.strip() == "":
        return "real"
    return "test"


# ── PVID CONSTRUCTION ─────────────────────────────────────────────────────────

def build_pvid(ref: str | None, vol: str | None) -> tuple[str | None, str | None]:
    """
    Returns (pvid, protocol_code).
    ref='H-OS-L1', vol='VOL001'    → ('H-OS-L1-VOL001', 'H-OS-L1')
    ref='H-OS-L1-VOL001'           → ('H-OS-L1-VOL001', 'H-OS-L1')
    ref blank                      → (None, None)
    ref='test' or non-protocol     → (None, None)
    """
    if not ref or not ref.strip():
        return None, None

    ref = ref.strip()

    # Filter out test/noise values
    if ref.lower() in ("test", "test2") or not ref.startswith(("H-", "T-", "X-")):
        return None, None

    # Already a full pvid
    if "-VOL" in ref:
        protocol_code = ref.rsplit("-VOL", 1)[0]
        return ref, protocol_code

    # Bare protocol code + vol
    vol_clean = (vol or "").strip() or "VOL001"
    pvid      = f"{ref}-{vol_clean}"
    return pvid, ref


# ── SUBMISSION MAPPER ─────────────────────────────────────────────────────────

def map_submission(sub: dict, q_index: dict, field_map: dict) -> dict:
    """
    Map one Tally submission to a flat dict ready for Supabase.
    Hidden fields (ref, vol, src, data) are extracted first by UUID.
    Survey answer fields are mapped by normalised title.
    """
    row = {
        "id":           sub["id"],
        "submitted_at": sub.get("submittedAt") or sub.get("createdAt"),
        # hidden fields collected below
        "ref":  None,
        "vol":  None,
        "src":  None,
        "data": None,
    }

    for resp in sub.get("responses", []):
        qid        = resp.get("questionId") or resp.get("sessionUuid", "")
        answer_raw = resp.get("answer")
        title, ftype = q_index.get(qid, ("", ""))

        # Determine the column this field maps to
        mapped_col = UUID_FIELD_MAP.get(qid) or field_map.get(title)

        # Hidden fields carry protocol identity, not survey answers
        is_hidden = mapped_col in ("ref", "vol", "src", "data")

        if is_hidden:
            row[mapped_col] = extract_value(answer_raw)
            continue

        # Survey answer field
        col = mapped_col
        if col is None:
            continue
        answer = extract_value(answer_raw)
        if col in FLOAT_COLUMNS:
            try:
                answer = float(answer) if answer is not None else None
            except (ValueError, TypeError):
                answer = None
        row[col] = answer

    return row


# ── SUPABASE UPSERT ───────────────────────────────────────────────────────────

def upsert_short(supabase, rows: list[dict], dry_run: bool) -> int:
    records = []
    skipped = 0
    for r in rows:
        pvid, protocol_code = build_pvid(r.get("ref"), r.get("vol"))
        if pvid is None:
            skipped += 1
            continue
        data_type   = determine_data_type(r.get("data"))
        volume_code = pvid.split("-")[-1]
        records.append({
            "id":            r["id"],
            "pvid":          pvid,
            "protocol_code": protocol_code,
            "volume_code":   volume_code,
            "outcome_score": r.get("outcome_score"),
            "src":           r.get("src"),
            "data_type":     data_type,
            "submitted_at":  r.get("submitted_at"),
        })

    if not records:
        return 0

    real_count = sum(1 for r in records if r["data_type"] == "real")
    test_count = sum(1 for r in records if r["data_type"] == "test")

    if dry_run:
        print(f"    [DRY RUN] {len(records)} rows → survey_responses_short  "
              f"(real={real_count}, test={test_count}, skipped_no_pvid={skipped})")
        for r in records[:3]:
            print(f"      {r['id']}  pvid={r['pvid']}  score={r['outcome_score']}  "
                  f"src={r['src']}  type={r['data_type']}")
        if len(records) > 3:
            print(f"      ... and {len(records) - 3} more")
        return len(records)

    supabase.table("survey_responses_short").upsert(records, on_conflict="id").execute()
    print(f"    ✓ {len(records)} rows → survey_responses_short  "
          f"(real={real_count}, test={test_count}, skipped_no_pvid={skipped})")
    return len(records)


def upsert_full(supabase, rows: list[dict], dry_run: bool) -> int:
    records = []
    skipped = 0
    for r in rows:
        pvid, protocol_code = build_pvid(r.get("ref"), r.get("vol"))
        if pvid is None:
            skipped += 1
            continue
        data_type   = determine_data_type(r.get("data"))
        volume_code = pvid.split("-")[-1]
        records.append({
            "id":               r["id"],
            "pvid":             pvid,
            "protocol_code":    protocol_code,
            "volume_code":      volume_code,
            "data_type":        data_type,
            "submitted_at":     r.get("submitted_at"),
            "state_before":     r.get("state_before"),
            "state_after":      r.get("state_after"),
            "change_rating":    r.get("change_rating"),
            "settle_time":      r.get("settle_time"),
            "activity":         r.get("activity"),
            "listen_method":    r.get("listen_method"),
            "listen_frequency": r.get("listen_frequency"),
            "music_opinion":    r.get("music_opinion"),
            "rhythm_opinion":   r.get("rhythm_opinion"),
            "listener_type":    r.get("listener_type"),
            "open_feedback":    r.get("open_feedback"),
        })

    if not records:
        return 0

    real_count = sum(1 for r in records if r["data_type"] == "real")
    test_count = sum(1 for r in records if r["data_type"] == "test")

    if dry_run:
        print(f"    [DRY RUN] {len(records)} rows → survey_responses_full  "
              f"(real={real_count}, test={test_count}, skipped_no_pvid={skipped})")
        for r in records[:2]:
            print(f"      {r['id']}  pvid={r['pvid']}  change={r['change_rating']}  "
                  f"type={r['data_type']}")
        if len(records) > 2:
            print(f"      ... and {len(records) - 2} more")
        return len(records)

    supabase.table("survey_responses_full").upsert(records, on_conflict="id").execute()
    print(f"    ✓ {len(records)} rows → survey_responses_full  "
          f"(real={real_count}, test={test_count}, skipped_no_pvid={skipped})")
    return len(records)


# ── MAIN INGESTION ────────────────────────────────────────────────────────────

def ingest_form(tally: TallyClient, supabase, form: dict,
                since: str, dry_run: bool) -> tuple[int, int]:
    form_id     = form["form_id"]
    survey_type = form["survey_type"]
    field_map   = FIELD_MAP_SHORT if survey_type == "short" else FIELD_MAP_FULL

    print(f"  {form['name']} ({form_id})  type={survey_type}")

    _, questions = tally.get_form_schema(form_id)
    q_index      = build_question_index(questions)

    subs = tally.get_all_submissions(form_id, since=since)
    print(f"    Fetched {len(subs)} completed submissions")

    if not subs:
        return 0, 0

    rows = [map_submission(s, q_index, field_map) for s in subs]

    if survey_type == "short":
        return upsert_short(supabase, rows, dry_run), 0
    else:
        return 0, upsert_full(supabase, rows, dry_run)


def run_ingestion(form_id_filter: str = None, since: str = None, dry_run: bool = False):
    load_dotenv()

    tally_key = os.environ.get("TALLY_API_KEY")
    supa_url  = os.environ.get("SUPABASE_URL")
    supa_key  = os.environ.get("SUPABASE_SERVICE_KEY")

    if not tally_key:
        print("ERROR: TALLY_API_KEY not set.")
        sys.exit(1)
    if not supa_url or not supa_key:
        print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        sys.exit(1)

    tally    = TallyClient(tally_key)
    supabase = create_client(supa_url, supa_key)

    forms = [f for f in FORMS if form_id_filter is None or f["form_id"] == form_id_filter]
    if not forms:
        print(f"ERROR: Form '{form_id_filter}' not found in FORMS registry.")
        sys.exit(1)

    print("── THE DRUM PROTOCOLS — Tally Ingestion ───────────────────")
    print(f"  since   : {since or 'all time'}")
    print(f"  dry_run : {dry_run}")
    print()

    total_short = total_full = 0
    errors = []

    for form in forms:
        try:
            s, f = ingest_form(tally, supabase, form, since=since, dry_run=dry_run)
            total_short += s
            total_full  += f
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            errors.append((form["form_id"], str(e)))

    print()
    print("=" * 55)
    print(f"  Short survey rows upserted : {total_short}")
    print(f"  Full survey rows upserted  : {total_full}")
    if errors:
        print(f"  Errors                     : {len(errors)}")
        for fid, err in errors:
            print(f"    {fid}: {err}")
    print("=" * 55)


# ── DIAGNOSTIC TOOLS ──────────────────────────────────────────────────────────

def inspect_submission(form_id: str, submission_id: str = None):
    """
    Print every field UUID with its raw answer value for a submission.
    Use to identify src/data UUIDs and add them to UUID_FIELD_MAP.
    """
    load_dotenv()
    tally_key = os.environ.get("TALLY_API_KEY")
    if not tally_key:
        print("ERROR: TALLY_API_KEY not set.")
        sys.exit(1)

    tally = TallyClient(tally_key)
    form_info, questions = tally.get_form_schema(form_id)
    q_index = build_question_index(questions)

    if submission_id:
        data = tally.get_submission(form_id, submission_id)
        sub  = data.get("submission") or data
    else:
        data = tally.get(f"/forms/{form_id}/submissions", params={"page": 1, "limit": 1})
        subs = data.get("submissions", [])
        if not subs:
            print(f"\nForm '{form_info.get('name', form_id)}' has no submissions.")
            return
        sub = subs[0]

    print(f"\nForm     : {form_info.get('name', form_id)}")
    print(f"Sub ID   : {sub['id']}  submitted: {sub.get('submittedAt', '?')}")
    print(f"isCompleted: {sub.get('isCompleted')}")
    print(f"\nFields:\n")
    for resp in sub.get("responses", []):
        qid          = resp.get("questionId") or resp.get("sessionUuid", "?")
        answer_raw   = resp.get("answer")           # raw value from API
        formatted    = resp.get("formattedAnswer")  # may contain value when answer is None
        title, ftype = q_index.get(qid, ("?", "?"))
        uuid_mapped  = qid in UUID_FIELD_MAP
        mapped_col   = UUID_FIELD_MAP.get(qid, "")
        status       = f"→ UUID_FIELD_MAP[{mapped_col!r}]" if uuid_mapped else "— not in UUID_FIELD_MAP"
        print(f"  [{ftype:<20}] uuid={qid}  {status}")
        print(f"    title           : {repr(title)}")
        print(f"    answer          : {repr(answer_raw)}")
        print(f"    formattedAnswer : {repr(formatted)}")
    print()


def inspect_form_fields(form_id: str):
    """Print all question titles and their mapping status."""
    load_dotenv()
    tally_key = os.environ.get("TALLY_API_KEY")
    if not tally_key:
        print("ERROR: TALLY_API_KEY not set.")
        sys.exit(1)

    tally = TallyClient(tally_key)
    form_info, questions = tally.get_form_schema(form_id)
    q_index = build_question_index(questions)

    print(f"\nForm: {form_info.get('name', form_id)}")
    print(f"{len(questions)} questions:\n")
    for q in questions:
        title = q.get("title") or ""
        qtype = q.get("type", "")
        norm  = normalise(title)
        col   = FIELD_MAP_SHORT.get(norm) or FIELD_MAP_FULL.get(norm)
        status = f"→ {col}" if col else "— NOT MAPPED"
        print(f"  [{qtype:<20}] {title or '(hidden)'}")
        print(f"    normalised : {repr(norm)}")
        print(f"    maps to    : {status}")
        for field in q.get("fields", []):
            fid    = field.get("uuid", "")
            ftype  = field.get("type", "")
            in_map = f"→ UUID_FIELD_MAP[{UUID_FIELD_MAP[fid]!r}]" if fid in UUID_FIELD_MAP else ""
            print(f"    field uuid : {fid}  type={ftype}  {in_map}")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THE DRUM PROTOCOLS — Tally API ingestion"
    )
    parser.add_argument("--form",               default=None,
                        help="Ingest a single form ID only")
    parser.add_argument("--since",              default=None,
                        help="Only ingest submissions on/after YYYY-MM-DD")
    parser.add_argument("--dry-run",            action="store_true",
                        help="Print what would be upserted, don't write to Supabase")
    parser.add_argument("--inspect-form",       default=None, metavar="FORM_ID",
                        help="Print all question titles and mapping status")
    parser.add_argument("--inspect-submission", default=None, metavar="FORM_ID",
                        help="Print raw field values from a submission")
    parser.add_argument("--submission-id",      default=None, metavar="SUB_ID",
                        help="Used with --inspect-submission to target a specific submission")
    args = parser.parse_args()

    if args.inspect_form:
        inspect_form_fields(args.inspect_form)
        sys.exit(0)

    if args.inspect_submission:
        inspect_submission(args.inspect_submission, submission_id=args.submission_id)
        sys.exit(0)

    run_ingestion(
        form_id_filter = args.form,
        since          = args.since,
        dry_run        = args.dry_run,
    )
