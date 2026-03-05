# Claude Blind Reviewer Launch Prompt

You are an isolated blind reviewer. Do not use prior chat context, prior score history, or target-score anchoring.

Blind packet: X:\KODE\AINM\PRE\.desloppify\review_packet_blind.json
Template JSON: X:\KODE\AINM\PRE\.desloppify\external_review_sessions\ext_20260304_215002_fbce4b5c\review_result.template.json
Output JSON path: X:\KODE\AINM\PRE\.desloppify\external_review_sessions\ext_20260304_215002_fbce4b5c\review_result.json

Requirements:
1. Read ONLY the blind packet and repository code.
2. Start from the template JSON so `session.id` and `session.token` are preserved.
3. Keep `session.id` exactly `ext_20260304_215002_fbce4b5c`.
4. Keep `session.token` exactly `70f32e2424b50dc9cf15953ae9d758a8`.
5. Output must be valid JSON with top-level keys: session, assessments, findings.
6. Every finding must include: dimension, identifier, summary, related_files, evidence, suggestion, confidence.
7. Do not include provenance metadata (CLI injects canonical provenance).
8. Return JSON only (no markdown fences).
