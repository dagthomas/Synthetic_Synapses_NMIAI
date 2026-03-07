/**
 * GET /api/captures?difficulty=nightmare&date=2026-03-07
 *
 * Returns the best capture file for a given difficulty + date.
 * "Best" = most orders. If no date given, uses today (UTC).
 * Returns capture data + metadata (filename, order count).
 */
import { resolve } from 'path';
import { readdir, readFile } from 'fs/promises';
import { GPU_DIR } from '$lib/paths.server.js';

const CAPTURES_DIR = resolve(GPU_DIR, 'captures');

function parseCaptureFile(filename) {
	const m = filename.match(/^(\w+?)_(\d{4}-\d{2}-\d{2})(?:_(\d{6})_(\d+)orders)?\.json$/);
	if (!m) return null;
	return { difficulty: m[1], date: m[2], time: m[3] || null, filename };
}

export async function GET({ url }) {
	const difficulty = url.searchParams.get('difficulty') || 'nightmare';
	const date = url.searchParams.get('date') || new Date().toISOString().slice(0, 10);

	let files;
	try {
		files = await readdir(CAPTURES_DIR);
	} catch {
		return json({ error: 'No captures directory' }, 404);
	}

	// Find all captures matching difficulty + date
	const matches = [];
	for (const fname of files.filter(f => f.endsWith('.json'))) {
		const parsed = parseCaptureFile(fname);
		if (!parsed || parsed.difficulty !== difficulty || parsed.date !== date) continue;

		try {
			const raw = await readFile(resolve(CAPTURES_DIR, fname), 'utf-8');
			const data = JSON.parse(raw);
			const orders = data.orders || [];
			const total = data.total_orders_discovered || orders.length;
			matches.push({ filename: fname, total, data });
		} catch { /* skip */ }
	}

	if (matches.length === 0) {
		return new Response(JSON.stringify({ found: false, difficulty, date }), {
			status: 200,
			headers: { 'Content-Type': 'application/json' },
		});
	}

	// Pick best (most orders)
	matches.sort((a, b) => b.total - a.total);
	const best = matches[0];

	return new Response(JSON.stringify({
		found: true,
		difficulty,
		date,
		filename: best.filename,
		orders: best.total,
		num_bots: best.data.num_bots || 0,
		capture: best.data,
	}), {
		headers: { 'Content-Type': 'application/json' },
	});
}
