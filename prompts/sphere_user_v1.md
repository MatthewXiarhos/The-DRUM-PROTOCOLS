---
persona: sphere
scope: user
version: v1
model: claude-sonnet-4-6
active_from: 2026-04-02
description: Sphere user-turn prompt template — data payload and output schema. Dynamic sections marked with {{INJECT:*}}.
---

Current date: {{INJECT:TODAY}}
Outcome scores available (1-second survey): {{INJECT:TOTAL_N_SHORT}}
Contextual responses available (1-minute survey): {{INJECT:TOTAL_N_FULL}}

=== SERIES SUMMARY ===
{{INJECT:SERIES_SUMMARY}}

=== PROTOCOL DATA (1-second survey — outcome scores) ===
{{INJECT:PROTOCOL_DATA}}

=== QUALITATIVE CONTEXT (1-minute survey — per protocol) ===
Each entry shows: listener_type breakdown | change_rating distribution | settle_time mode | activity mode | music_opinion mode | rhythm_opinion mode
Protocols with no 1-minute responses are omitted.
{{INJECT:FULL_SURVEY_CONTEXT}}

=== YOUTUBE REACH & RETENTION (most recent available data per protocol) ===
Each entry shows: views (cumulative) | avg_view_duration (seconds) | avg_view_percentage (% of video watched)
NULL values indicate data not yet available (YouTube Analytics has a 2-3 day lag).
Use this data as a reach and engagement signal — how many people found each protocol,
and whether they stayed. A high outcome score with low views = undiscovered. A high
view count with low retention = listener drop-off before entrainment. Where retention
data is available, note whether it aligns or conflicts with outcome scores.
Protocols with no YouTube data are omitted.
{{INJECT:YOUTUBE_CONTEXT}}

=== YOUR TASK ===
Generate Sphere commentary in the following JSON structure.
Return ONLY valid JSON, no markdown fences, no preamble.

Where qualitative context exists for a protocol, incorporate it — listener type mix,
how listeners describe the change, how quickly they settled, and their opinions on the
music and rhythm are meaningful signals alongside the outcome scores.

Where YouTube data exists, reference reach and retention signals in your commentary —
particularly where they add or complicate the picture from outcome scores alone.

FORMATTING RULES for overview, cross_series, anomalies, and development_signals:
These fields are displayed directly in the UI. Use the following structure:

1. Start with one clear summary sentence on its own line.
2. Follow with bullet points for the main findings, each starting with • on a new line.
3. Keep each bullet under 25 words.
4. Use plain language — these are read by researchers and curious non-technical visitors alike.
5. Do not use markdown, asterisks, or HTML tags — plain text and • bullets only.

Example format for cross_series:
"The three series show distinct efficacy profiles that reflect their design intent.\n• HEALING leads in listener consistency, with the lowest variance across entry points.\n• THRIVING shows the highest mean scores, particularly in Focus & Clarity protocols.\n• TRANSFORMING carries the most spread — expected given its ascending-arc design."

{
  "overview": "One summary sentence.\n• Key finding one\n• Key finding two\n• Key finding three (if warranted)",
  "cross_series": "One summary sentence.\n• HEALING finding\n• THRIVING finding\n• TRANSFORMING finding",
  "anomalies": "One summary sentence.\n• Anomaly one with protocol code\n• Anomaly two with protocol code (if present)\n• Note on what the anomaly suggests",
  "development_signals": "One summary sentence.\n• Signal one — cite protocol code and suggested action\n• Signal two — cite protocol code and suggested action\n• Signal three — cite protocol code and suggested action (if warranted)\n• Signal four (if warranted)",
  "by_protocol": {
    "H-OS-L1": "1-2 sentence technical reading — use statistical terminology, cite n, mean, std, bimodality, confidence tier. Where qualitative data exists, reference it briefly. Where YouTube data exists, note views and retention.",
    "... (include all protocols)"
  },
  "by_protocol_plain": {
    "H-OS-L1": "1-2 sentence plain-language version — no jargon, no statistics terms, written for a curious non-technical listener. Focus on what the data actually means for the experience: how reliably it seems to help, whether results vary a lot between people, whether there are any surprises. Warm but honest tone.",
    "... (include all protocols)"
  }
}
