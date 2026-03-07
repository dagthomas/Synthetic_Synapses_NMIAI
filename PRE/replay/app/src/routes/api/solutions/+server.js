/**
 * GET /api/solutions?date=2026-03-07
 *
 * Returns all GPU solutions from the database, grouped by difficulty.
 * If no date given, returns ALL dates. If date given, filters to that date.
 */
import { query } from '$lib/db.server.js';

export async function GET({ url }) {
	const date = url.searchParams.get('date'); // optional filter

	try {
		let rows;
		if (date) {
			rows = await query(
				`SELECT difficulty, date, score, map_seed, num_bots, num_rounds,
				        capture_hash, optimizations_run, created_at, updated_at
				 FROM gpu_solutions
				 WHERE date = $1
				 ORDER BY difficulty, date DESC`,
				[date]
			);
		} else {
			rows = await query(
				`SELECT difficulty, date, score, map_seed, num_bots, num_rounds,
				        capture_hash, optimizations_run, created_at, updated_at
				 FROM gpu_solutions
				 ORDER BY date DESC, difficulty`
			);
		}

		// Group by difficulty
		const byDifficulty = {};
		for (const row of rows) {
			const d = row.difficulty;
			if (!byDifficulty[d]) byDifficulty[d] = [];
			byDifficulty[d].push({
				difficulty: row.difficulty,
				date: row.date,
				score: row.score,
				seed: row.map_seed || 0,
				num_bots: row.num_bots,
				num_rounds: row.num_rounds,
				capture_hash: row.capture_hash,
				optimizations_run: row.optimizations_run || 0,
				created_at: row.created_at,
				updated_at: row.updated_at,
			});
		}

		// Sort difficulties by priority
		const order = ['nightmare', 'expert', 'hard', 'medium', 'easy'];
		const sorted = {};
		for (const d of order) {
			if (byDifficulty[d]) sorted[d] = byDifficulty[d];
		}
		for (const [d, entries] of Object.entries(byDifficulty)) {
			if (!sorted[d]) sorted[d] = entries;
		}

		return new Response(JSON.stringify({ byDifficulty: sorted, total: rows.length }), {
			headers: { 'Content-Type': 'application/json' },
		});
	} catch (e) {
		return new Response(JSON.stringify({ error: e.message, byDifficulty: {}, total: 0 }), {
			status: 200,
			headers: { 'Content-Type': 'application/json' },
		});
	}
}
