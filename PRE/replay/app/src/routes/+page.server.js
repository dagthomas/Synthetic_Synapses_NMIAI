import { query } from '$lib/db.server.js';

const PAGE_SIZE = 50;

export async function load({ url }) {
	const page = Math.max(1, parseInt(url.searchParams.get('page') || '1'));
	const offset = (page - 1) * PAGE_SIZE;

	const [runs, countResult, stats, typeStats] = await Promise.all([
		query(`
			SELECT id, seed, difficulty, grid_width, grid_height, bot_count,
			       final_score, items_delivered, orders_completed, created_at,
			       run_type
			FROM runs
			ORDER BY created_at DESC
			LIMIT $1 OFFSET $2
		`, [PAGE_SIZE, offset]),

		query(`SELECT COUNT(*)::int as total FROM runs`),

		query(`
			SELECT difficulty,
			       COUNT(*)::int as count,
			       MAX(final_score) as max_score,
			       ROUND(AVG(final_score), 1)::float as avg_score,
			       MIN(final_score) as min_score
			FROM runs
			GROUP BY difficulty
			ORDER BY difficulty
		`),

		query(`
			SELECT difficulty, run_type,
			       COUNT(*)::int as count,
			       MAX(final_score) as max_score,
			       ROUND(AVG(final_score), 1)::float as avg_score
			FROM runs
			GROUP BY difficulty, run_type
			ORDER BY difficulty, run_type
		`)
	]);

	const total = countResult[0]?.total || 0;

	return {
		runs,
		stats,
		typeStats,
		page,
		pageSize: PAGE_SIZE,
		totalRuns: total,
		totalPages: Math.max(1, Math.ceil(total / PAGE_SIZE))
	};
}
