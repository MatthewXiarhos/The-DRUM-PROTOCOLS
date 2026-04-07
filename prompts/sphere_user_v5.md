---
persona: sphere
scope: user
version: v5
model: nebius-pool
active_from: 2026-04-07
description: >
  Sphere user-turn prompt template — data payload and output schema.
  v5 changes from v3: (1) corrected AutoGluon ensemble description —
  WeightedEnsemble_L3, stacked 3 levels, includes NeuralNetTorch;
  (2) by_protocol trimmed from 4-6 sentences to 3-4 sentences to manage
  synthesizer output token budget as the library grows across volumes;
  (3) volume-awareness note added — Sphere frames commentary correctly
  when a run covers protocols from multiple volumes; (4) by_protocol_volumes
  field added to output schema — per-volume commentary keyed by
  [protocol_code][volume_code], populated only when multiple volumes have
  data, empty object when single volume; (5) output schema placeholder
  updated from hardcoded 30-protocol list to "all protocols in this run".
---

Current date: {{INJECT:TODAY}}
Outcome scores available (1-second survey): {{INJECT:TOTAL_N_SHORT}}
Contextual responses available (1-minute survey): {{INJECT:TOTAL_N_FULL}}

NOTE ON SCOPE: The protocol data below covers one or more volumes of The Drum
Protocols library. Each volume contains the same 30 protocol designs (identical
Hz parameters, duration, and movement arc) recorded with a different groove style.
Per-volume response breakdowns are shown in the PROTOCOL DATA section where
multiple volumes have data. When only one volume has responses, frame findings as
volume-specific. When multiple volumes have data for the same protocol, note
whether response patterns differ across volumes — this is a primary research
question for the multi-volume library.

=== SERIES SUMMARY ===
{{INJECT:SERIES_SUMMARY}}

=== PROTOCOL DATA (1-second survey — outcome scores) ===
Aggregate stats shown at protocol level. Where multiple volumes have data,
per-volume breakdowns appear as indented lines below each protocol entry.
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

=== EMBEDDING CONTEXT (cosine similarity drift and semantic clustering) ===
This section contains two signals derived from past Sphere run embeddings stored in
Supabase. CROSS-RUN DRIFT shows how much Sphere's commentary on each protocol has
shifted between earliest and most recent runs — high drift (similarity < 0.85) may
indicate evolving data patterns or genuine analytical instability. CROSS-PROTOCOL
SEMANTIC CLUSTERS shows which protocols Sphere groups together analytically, regardless
of their series or entry category. Use this to identify whether Sphere's framing is
consistent and whether any protocols are analytically isolated.
{{INJECT:EMBEDDING_CONTEXT}}

=== AUTOGLUON ENSEMBLE ANALYSIS ===
This section contains the output of an AutoGluon ensemble ML model (WeightedEnsemble_L3,
stacked 3 levels: LightGBM + CatBoost + XGBoost + RandomForest + NeuralNetTorch) trained
on all available listener outcome data. This is the primary quantitative signal layer for
your analysis. Unlike the descriptive statistics above, AutoGluon output is predictive
— it tells you which variables are causally driving outcome scores, not just what the
distributions look like.

Key sections to reason from:
- FEATURE IMPORTANCE: which protocol design parameters and listener context features
  most strongly predict outcome_score. Features with zero importance currently lack
  signal due to uniform test data — they will activate as real listener data grows.
- PREDICTED OUTCOME SCORES BY ZONE: AutoGluon ensemble predictions per zone, not
  observed means. These are the model's best estimate of expected efficacy.
- CROSS-BUCKET ROUTING ANOMALIES: protocols the model recommends outside their
  design zone — these represent efficacy signals crossing categorical boundaries and
  warrant direct analytical attention.
- ZONE EFFICACY RANKING: which zones the model predicts will produce the highest
  listener outcomes, based on protocol design features and engagement signals.

Where AutoGluon findings converge with observed outcome score patterns, treat this as
stronger signal. Where they diverge, flag the tension explicitly — it is analytically
meaningful.
{{INJECT:AUTOGLUON_CONTEXT}}

=== YOUR TASK ===
Generate Sphere commentary in the following JSON structure.
Return ONLY valid JSON, no markdown fences, no preamble.

Where qualitative context exists for a protocol, incorporate it — listener type mix,
how listeners describe the change, how quickly they settled, and their opinions on the
music and rhythm are meaningful signals alongside the outcome scores.

Where YouTube data exists, reference reach and retention signals in your commentary —
particularly where they add or complicate the picture from outcome scores alone.

Where AutoGluon findings are relevant to a protocol, incorporate them — particularly
feature importance rankings, predicted scores, and any cross-bucket routing that
affects that protocol.

Where multiple volumes have data for the same protocol, note whether patterns differ
across volumes in both by_protocol (technical) and by_protocol_volumes entries.
Where only one volume has data, by_protocol_volumes may be omitted or left as {}.

FORMATTING RULES for overview, cross_series, anomalies, development_signals, and
embedding_analysis:
These fields are displayed directly in the UI. Use the following structure:

1. Start with one clear summary sentence on its own line.
2. Follow with bullet points for the main findings, each starting with • on a new line.
3. Keep each bullet under 25 words.
4. Use plain language — these are read by researchers and curious non-technical visitors alike.
5. Do not use markdown, asterisks, or HTML tags — plain text and • bullets only.

Example format for cross_series:
"The three series show distinct efficacy profiles that reflect their design intent.\n• HEALING leads in listener consistency, with the lowest variance across entry points.\n• THRIVING shows the highest mean scores, particularly in Focus & Clarity protocols.\n• TRANSFORMING carries the most spread — expected given its ascending-arc design."

OUTPUT LENGTH GUIDANCE:
- by_protocol entries: 3-4 sentences. Cite n, mean, std, confidence tier. Reference
  qualitative data, YouTube signal, and AutoGluon delta where material.
- by_protocol_plain entries: 1-2 sentences in plain language for a non-technical
  listener. No statistics terminology. Warm but honest tone.
- by_protocol_volumes entries: 2-3 sentences per volume. Focus on what differs
  between volumes — response pattern, n, mean delta, confidence tier. Only include
  where multiple volumes have data.
- Keep all entries concise — signal density over length. This applies regardless of
  how many protocols are in the payload.

{
  "overview": "One summary sentence.\n• Key finding one\n• Key finding two\n• Key finding three (if warranted)",
  "cross_series": "One summary sentence.\n• HEALING finding\n• THRIVING finding\n• TRANSFORMING finding",
  "anomalies": "One summary sentence.\n• Anomaly one with protocol code\n• Anomaly two with protocol code (if present)\n• Note on what the anomaly suggests",
  "development_signals": "One summary sentence.\n• Signal one — cite protocol code and suggested action\n• Signal two — cite protocol code and suggested action\n• Signal three — cite protocol code and suggested action (if warranted)\n• Signal four (if warranted)",
  "embedding_analysis": "One summary sentence.\n• CROSS-RUN DRIFT: finding or 'Insufficient run history for drift analysis'\n• CROSS-PROTOCOL SEMANTIC CLUSTERS: finding\n• Confidence note on embedding data quantity",
  "by_protocol": {
    "H-OS-L1": "3-4 sentences — aggregate across all volumes. Cite n, mean, std, confidence tier. Reference qualitative data, YouTube signal, AutoGluon predicted score delta where material. Flag bimodality or cross-volume divergence.",
    "... (one entry per protocol_code in this run)"
  },
  "by_protocol_plain": {
    "H-OS-L1": "1-2 sentences in plain language. No jargon. What does the data mean for the listening experience across all volumes?",
    "... (one entry per protocol_code in this run)"
  },
  "by_protocol_volumes": {
    "H-OS-L1": {
      "VOL001": "2-3 sentences on this protocol in this specific volume. What is distinctive about this volume's response pattern? How does it compare to other volumes if data is available?",
      "VOL002": "2-3 sentences on this protocol in VOL002."
    },
    "... (only include protocols where multiple volumes have data — omit entirely if single volume)"
  }
}
