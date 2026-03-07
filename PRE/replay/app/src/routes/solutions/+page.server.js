import { query } from '$lib/db.server.js';

export async function load() {
	try {
		const rows = await query(
			`SELECT difficulty, date, score, map_seed, num_bots, num_rounds,
			        capture_hash, optimizations_run, created_at, updated_at
			 FROM gpu_solutions
			 ORDER BY date DESC, difficulty`
		);

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
				created_at: row.created_at ? new Date(row.created_at).toISOString() : null,
				updated_at: row.updated_at ? new Date(row.updated_at).toISOString() : null,
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

		return { byDifficulty: sorted, total: rows.length };
	} catch (e) {
		return { byDifficulty: {}, total: 0, error: e.message };
	}
}
