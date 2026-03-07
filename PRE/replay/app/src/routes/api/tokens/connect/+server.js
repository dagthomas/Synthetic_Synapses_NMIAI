import { query } from '$lib/db.server.js';
import { createCleanup, createSendEvent } from '$lib/sse.server.js';
import WebSocket from 'ws';

function parseToken(url) {
	try {
		const tokenMatch = url.match(/[?&]token=([^&]+)/);
		if (!tokenMatch) return null;
		const parts = tokenMatch[1].split('.');
		if (parts.length < 2) return null;
		const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
		return JSON.parse(Buffer.from(payload, 'base64').toString());
	} catch { return null; }
}

export async function POST({ request }) {
	const { ws_url, token_id } = await request.json();

	if (!ws_url?.trim()) {
		return new Response(JSON.stringify({ error: 'ws_url required' }), { status: 400 });
	}

	const parsed = parseToken(ws_url);
	const difficulty = parsed?.difficulty || null;
	const map_seed = parsed?.map_seed || null;

	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };
	let ws = null;
	let sessionId = null;

	const stream = new ReadableStream({
		async start(controller) {
			const cleanup = createCleanup(ctx, controller);
			const sendEvent = createSendEvent(ctx, controller, encoder);

			ctx.safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: 10 minutes exceeded' });
				if (ws) ws.close();
				cleanup();
			}, 600000);

			// Keep SSE alive
			ctx.heartbeatInterval = setInterval(() => {
				sendEvent('heartbeat', { t: Date.now() });
			}, 15000);

			// Create session in DB
			try {
				const rows = await query(
					`INSERT INTO ws_sessions (token_id, ws_url, difficulty, map_seed, status)
					 VALUES ($1, $2, $3, $4, 'connecting') RETURNING id`,
					[token_id || null, ws_url.trim(), difficulty, map_seed]
				);
				sessionId = rows[0].id;
				sendEvent('session', { session_id: sessionId, difficulty, map_seed });
			} catch (e) {
				sendEvent('error', { message: `DB error: ${e.message}` });
				cleanup();
				return;
			}

			let seq = 0;
			let roundCount = 0;
			let numBots = 0;
			let finalScore = null;

			// Batch insert buffer
			const msgBatch = [];
			let flushTimer = null;

			async function flushMessages() {
				if (msgBatch.length === 0) return;
				const batch = msgBatch.splice(0);
				const values = [];
				const placeholders = [];
				let i = 1;
				for (const m of batch) {
					placeholders.push(`($${i}, $${i+1}, $${i+2}, $${i+3}, $${i+4})`);
					values.push(sessionId, m.seq, m.msg_type, m.round_num, JSON.stringify(m.raw));
					i += 5;
				}
				try {
					await query(
						`INSERT INTO ws_messages (session_id, seq, msg_type, round_num, raw)
						 VALUES ${placeholders.join(', ')}`,
						values
					);
				} catch (e) {
					sendEvent('db_error', { message: e.message });
				}
			}

			function queueMessage(msg_type, round_num, raw) {
				msgBatch.push({ seq: seq++, msg_type, round_num, raw });
				if (!flushTimer) {
					flushTimer = setTimeout(async () => {
						flushTimer = null;
						await flushMessages();
					}, 500);
				}
			}

			// Connect to game WebSocket
			try {
				ws = new WebSocket(ws_url.trim());
			} catch (e) {
				sendEvent('error', { message: `WS connect failed: ${e.message}` });
				await query(`UPDATE ws_sessions SET status='error', ended_at=NOW() WHERE id=$1`, [sessionId]);
				cleanup();
				return;
			}

			ws.on('open', async () => {
				sendEvent('connected', { message: 'WebSocket connected' });
				await query(`UPDATE ws_sessions SET status='connected' WHERE id=$1`, [sessionId]);
			});

			ws.on('message', async (data) => {
				let msg;
				try { msg = JSON.parse(data.toString()); } catch {
					sendEvent('raw', { text: data.toString() });
					return;
				}

				const msgType = msg.type || 'game_state';
				const roundNum = msg.round ?? null;

				// Store in DB
				queueMessage(msgType, roundNum, msg);

				// Stream to frontend
				sendEvent('ws_message', { seq: seq - 1, msg_type: msgType, round: roundNum, data: msg });

				if (msgType === 'game_over') {
					finalScore = msg.score ?? null;
					sendEvent('game_over', { score: finalScore, rounds_used: msg.rounds_used });

					// Flush remaining
					await flushMessages();
					await query(
						`UPDATE ws_sessions SET status='finished', ended_at=NOW(),
						 final_score=$1, rounds_received=$2 WHERE id=$3`,
						[finalScore, roundCount, sessionId]
					);
					if (ws) ws.close();
					cleanup();
					return;
				}

				if (msgType === 'game_state' || roundNum !== null) {
					roundCount++;
					numBots = msg.bots?.length || numBots;

					// Send wait actions to keep the game alive
					const waitActions = [];
					for (let b = 0; b < numBots; b++) {
						waitActions.push({ bot: b, action: 'wait' });
					}
					try {
						ws.send(JSON.stringify({ actions: waitActions }));
					} catch { /* ws may be closing */ }
				}
			});

			ws.on('error', async (err) => {
				sendEvent('error', { message: `WS error: ${err.message}` });
				await flushMessages();
				await query(`UPDATE ws_sessions SET status='error', ended_at=NOW(), rounds_received=$1 WHERE id=$2`,
					[roundCount, sessionId]);
				cleanup();
			});

			ws.on('close', async (code, reason) => {
				sendEvent('ws_closed', { code, reason: reason?.toString() || '' });
				await flushMessages();
				if (!finalScore) {
					await query(
						`UPDATE ws_sessions SET status='closed', ended_at=NOW(), rounds_received=$1 WHERE id=$2`,
						[roundCount, sessionId]
					);
				}
				cleanup();
			});
		},
		cancel() {
			if (ws && ws.readyState === WebSocket.OPEN) ws.close();
			createCleanup(ctx, null)();
		}
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
		},
	});
}
