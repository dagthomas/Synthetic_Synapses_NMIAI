# External Blind Review Session

Session id: ext_20260304_215002_fbce4b5c
Session token: 70f32e2424b50dc9cf15953ae9d758a8
Blind packet: X:\KODE\AINM\PRE\.desloppify\review_packet_blind.json
Template output: X:\KODE\AINM\PRE\.desloppify\external_review_sessions\ext_20260304_215002_fbce4b5c\review_result.template.json
Claude launch prompt: X:\KODE\AINM\PRE\.desloppify\external_review_sessions\ext_20260304_215002_fbce4b5c\claude_launch_prompt.md
Expected reviewer output: X:\KODE\AINM\PRE\.desloppify\external_review_sessions\ext_20260304_215002_fbce4b5c\review_result.json

Happy path:
1. Open the Claude launch prompt file and paste it into a context-isolated subagent task.
2. Reviewer writes JSON output to the expected reviewer output path.
3. Submit with the printed --external-submit command.

Reviewer output requirements:
1. Return JSON with top-level keys: session, assessments, findings.
2. session.id must be `ext_20260304_215002_fbce4b5c`.
3. session.token must be `70f32e2424b50dc9cf15953ae9d758a8`.
4. Include findings with required schema fields (dimension/identifier/summary/related_files/evidence/suggestion/confidence).
5. Use the blind packet only (no score targets or prior context).
