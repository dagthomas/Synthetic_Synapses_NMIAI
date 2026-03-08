import { query } from '$lib/db.server.js';

export async function load() {
	// Today's date in YYYY-MM-DD
	const today = new Date().toISOString().slice(0, 10);

	// Best run per difficulty for today (highest final_score)
	const runs = await query(`
		SELECT DISTINCT ON (difficulty)
			id, difficulty, seed, bot_count, final_score, items_delivered,
			orders_completed, run_type, created_at
		FROM runs
		WHERE created_at::date = $1::date
		ORDER BY difficulty, final_score DESC, id DESC
	`, [today]);

	// Extract all unique orders from best runs' round data (separate rounds table)
	const runOrdersMap = {};
	for (const run of runs) {
		const orderRows = await query(`
			SELECT order_id, items_required FROM (
				SELECT DISTINCT ON (ord->>'id')
					ord->>'id' AS order_id,
					ord->'items_required' AS items_required,
					r.round_number AS first_round
				FROM rounds r,
					jsonb_array_elements(r.orders) AS ord
				WHERE r.run_id = $1
				ORDER BY ord->>'id', r.round_number
			) sub
			ORDER BY first_round, order_id
		`, [run.id]);
		runOrdersMap[run.difficulty] = orderRows.map(r => ({
			id: r.order_id,
			items: r.items_required || [],
		}));
	}

	// Captures from DB for today
	const captures = await query(`
		SELECT difficulty, date, num_orders, capture_data, created_at, updated_at
		FROM captures
		WHERE date = $1
	`, [today]);

	// GPU solutions for today
	const solutions = await query(`
		SELECT difficulty, score, num_bots, date
		FROM gpu_solutions
		WHERE date = $1
	`, [today]);

	// Order sequences
	const orderSeqs = await query(`
		SELECT difficulty, map_seed, total_orders, date
		FROM order_sequences
		WHERE date = $1
	`, [today]);

	// Build combined view per difficulty
	const difficultyOrder = ['nightmare', 'expert', 'hard', 'medium', 'easy'];
	const byDifficulty = {};

	for (const diff of difficultyOrder) {
		const run = runs.find(r => r.difficulty === diff);
		const cap = captures.find(c => c.difficulty === diff);
		const sol = solutions.find(s => s.difficulty === diff);
		const seq = orderSeqs.find(o => o.difficulty === diff);

		if (!run && !cap && !sol) continue;

		// Prefer run orders (most complete — extracted from game rounds), fall back to capture orders
		const runOrders = runOrdersMap[diff] || [];
		const captureOrderList = (cap?.capture_data?.orders || []).map((o, i) => ({
			id: o.id || `order_${i}`,
			items: o.items_required || []
		}));
		const allOrders = runOrders.length > captureOrderList.length ? runOrders : captureOrderList;

		byDifficulty[diff] = {
			difficulty: diff,
			date: today,
			bestRun: run ? {
				id: run.id,
				score: run.final_score,
				itemsDelivered: run.items_delivered,
				ordersCompleted: run.orders_completed,
				runType: run.run_type,
				botCount: run.bot_count,
				seed: run.seed,
				createdAt: run.created_at
			} : null,
			captureOrders: allOrders,
			totalOrders: Math.max(seq?.total_orders || 0, allOrders.length),
			gpuScore: sol?.score || null,
		};
	}

	return { today, byDifficulty };
}
