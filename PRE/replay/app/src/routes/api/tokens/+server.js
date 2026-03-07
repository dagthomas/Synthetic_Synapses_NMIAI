import { json } from '@sveltejs/kit';
import { query } from '$lib/db.server.js';

function parseToken(url) {
	try {
		const tokenMatch = url.match(/[?&]token=([^&]+)/);
		if (!tokenMatch) return null;
		const token = tokenMatch[1];
		const parts = token.split('.');
		if (parts.length < 2) return null;
		const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
		const decoded = JSON.parse(Buffer.from(payload, 'base64').toString());
		return { token_raw: token, ...decoded };
	} catch {
		return null;
	}
}

export async function GET() {
	const rows = await query('SELECT * FROM tokens ORDER BY created_at DESC');
	return json(rows);
}

export async function POST({ request }) {
	const { ws_url, label } = await request.json();
	if (!ws_url?.trim()) return json({ error: 'ws_url required' }, { status: 400 });

	const parsed = parseToken(ws_url.trim());
	const difficulty = parsed?.difficulty || null;
	const map_seed = parsed?.map_seed || null;
	const token_raw = parsed?.token_raw || null;

	try {
		const rows = await query(
			`INSERT INTO tokens (ws_url, difficulty, map_seed, token_raw, label)
			 VALUES ($1, $2, $3, $4, $5)
			 ON CONFLICT (ws_url) DO UPDATE SET label = COALESCE(EXCLUDED.label, tokens.label), created_at = NOW()
			 RETURNING *`,
			[ws_url.trim(), difficulty, map_seed, token_raw, label || null]
		);
		return json(rows[0], { status: 201 });
	} catch (e) {
		return json({ error: e.message }, { status: 500 });
	}
}

export async function DELETE({ request }) {
	const { id } = await request.json();
	if (!id) return json({ error: 'id required' }, { status: 400 });
	await query('DELETE FROM tokens WHERE id = $1', [id]);
	return json({ ok: true });
}
