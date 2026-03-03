import { query } from '$lib/db.server.js';

export async function load() {
	const runs = await query(`
		SELECT id, seed, difficulty, grid_width, grid_height, bot_count,
		       final_score, items_delivered, orders_completed, created_at,
		       run_type
		FROM runs
		ORDER BY created_at DESC
		LIMIT 200
	`);

	const stats = await query(`
		SELECT difficulty,
		       COUNT(*)::int as count,
		       MAX(final_score) as max_score,
		       ROUND(AVG(final_score), 1)::float as avg_score,
		       MIN(final_score) as min_score
		FROM runs
		GROUP BY difficulty
		ORDER BY difficulty
	`);

	return { runs, stats };
}
