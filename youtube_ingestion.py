#!/usr/bin/env python3
"""
youtube_ingestion.py

Fetches daily YouTube stats for all 30 protocol videos and writes
to the youtube_daily table in Supabase.

Reads YouTube credentials from .env:
  - YOUTUBE_CLIENT_ID
  - YOUTUBE_CLIENT_SECRET
  - YOUTUBE_REFRESH_TOKEN

Usage:
  py -3.11 youtube_ingestion.py                     # fetches yesterday
  py -3.11 youtube_ingestion.py --date 2026-04-01   # specific date
  py -3.11 youtube_ingestion.py --days 7            # last N days
"""

import argparse
import json
import os
from datetime import date, timedelta
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from supabase import create_client

load_dotenv()

SUPABASE_URL         = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def build_youtube_clients():
    """Build YouTube API clients from .env credentials."""
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        scopes=[
            "https://www.googleapis.com/auth/youtube.readonly",
            "https://www.googleapis.com/auth/yt-analytics.readonly",
        ],
    )
    yt      = build("youtube",          "v3", credentials=creds)
    yt_anal = build("youtubeAnalytics", "v2", credentials=creds)
    return yt, yt_anal


def get_pvid_map(supabase) -> dict:
    """Return {youtube_video_id: {pvid, protocol_code, volume_code}} for all 30 videos."""
    rows = supabase.table("protocol_volumes").select(
        "pvid, protocol_code, volume_code, youtube_video_id"
    ).execute().data
    return {r["youtube_video_id"]: r for r in rows if r.get("youtube_video_id")}


def extract_video_id(url_or_id: str) -> str:
    """Strip full YouTube URL to bare video ID if needed."""
    if "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[-1].strip("/")
    if "youtube.com/watch" in url_or_id:
        return parse_qs(urlparse(url_or_id).query)["v"][0]
    return url_or_id  # already a bare ID


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_video_stats(yt, video_ids: list) -> dict:
    """
    YouTube Data API v3 — videos.list
    Returns {bare_video_id: {views, likes}}
    """
    stats    = {}
    bare_ids = [extract_video_id(vid) for vid in video_ids]
    for i in range(0, len(bare_ids), 50):
        chunk = bare_ids[i:i+50]
        resp  = yt.videos().list(
            part="statistics",
            id=",".join(chunk)
        ).execute()
        for item in resp.get("items", []):
            vid = item["id"]
            s   = item.get("statistics", {})
            stats[vid] = {
                "views": int(s.get("viewCount", 0)),
                "likes": int(s.get("likeCount", 0)),
            }
    return stats


def fetch_analytics(yt_anal, video_id: str, report_date: date) -> dict:
    """
    YouTube Analytics API v2 — avg view duration and percentage for a single video/date.
    """
    date_str = report_date.isoformat()
    try:
        resp = yt_anal.reports().query(
            ids=f"video=={video_id}",
            startDate=date_str,
            endDate=date_str,
            metrics="averageViewDuration,averageViewPercentage",
        ).execute()
        rows = resp.get("rows", [])
        if rows:
            return {
                "avg_view_duration":   float(rows[0][0]),
                "avg_view_percentage": float(rows[0][1]),
            }
    except Exception as e:
        print(f"  ⚠ Analytics fetch failed for {video_id}: {e}")
    return {"avg_view_duration": None, "avg_view_percentage": None}


def fetch_retention_curve(yt_anal, video_id: str, report_date: date) -> list | None:
    """
    Audience retention curve — elapsedVideoTimeRatio vs audienceWatchRatio.
    Returns a list of {elapsed, retained} dicts, or None if unavailable.
    """
    date_str = report_date.isoformat()
    try:
        resp = yt_anal.reports().query(
            ids=f"video=={video_id}",
            startDate=date_str,
            endDate=date_str,
            metrics="audienceWatchRatio",
            dimensions="elapsedVideoTimeRatio",
        ).execute()
        rows = resp.get("rows", [])
        if rows:
            return [{"elapsed": r[0], "retained": r[1]} for r in rows]
    except Exception as e:
        print(f"  ⚠ Retention fetch failed for {video_id}: {e}")
    return None


# ── Write-back ────────────────────────────────────────────────────────────────

def upsert_rows(supabase, rows: list):
    """Upsert rows into youtube_daily. PK is id = {pvid}-{report_date}."""
    if not rows:
        print("No rows to upsert.")
        return
    result = supabase.table("youtube_daily").upsert(rows, on_conflict="id").execute()
    print(f"  ✓ Upserted {len(rows)} rows to youtube_daily.")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run(report_dates: list[date]):
    supabase = get_supabase()

    print("Connecting to YouTube APIs via .env credentials...")
    yt, yt_anal = build_youtube_clients()

    pvid_map  = get_pvid_map(supabase)
    video_ids = list(pvid_map.keys())
    print(f"Found {len(video_ids)} videos in protocol_volumes.")

    for report_date in report_dates:
        print(f"\n── Fetching stats for {report_date} ──")

        stats = fetch_video_stats(yt, video_ids)

        rows = []
        for video_id in video_ids:
            meta    = pvid_map[video_id]
            pvid    = meta["pvid"]
            bare_id = extract_video_id(video_id)
            print(f"  {pvid} ({bare_id})")

            base  = stats.get(bare_id, {"views": None, "likes": None})
            anal  = fetch_analytics(yt_anal, bare_id, report_date)
            curve = fetch_retention_curve(yt_anal, bare_id, report_date)

            rows.append({
                "id":                  f"{pvid}-{report_date}",
                "pvid":                pvid,
                "protocol_code":       meta["protocol_code"],
                "volume_code":         meta["volume_code"],
                "report_date":         report_date.isoformat(),
                "views":               base["views"],
                "likes":               base["likes"],
                "avg_view_duration":   anal["avg_view_duration"],
                "avg_view_percentage": anal["avg_view_percentage"],
                "retention_curve":     json.dumps(curve) if curve else None,
            })

        upsert_rows(supabase, rows)

    print("\n✓ YouTube ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Fetch stats for a specific date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=1,
                        help="Fetch stats for last N days (default: 1 = yesterday)")
    args = parser.parse_args()

    if args.date:
        dates = [date.fromisoformat(args.date)]
    else:
        today = date.today()
        dates = [today - timedelta(days=i) for i in range(1, args.days + 1)]

    run(dates)