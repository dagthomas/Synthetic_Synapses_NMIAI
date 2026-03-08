import { queryOne, query } from '$lib/db.server.js';

const EMPTY = {
	defaultCapture: null,
	defaultDifficulty: null,
	defaultFileName: null,
	defaultNumOrders: 0,
};

export async function load() {
	const today = new Date().toISOString().slice(0, 10);

	try {
		// Find today's best capture across all difficulties (most orders)
		const best = await queryOne(`
			SELECT difficulty, date, num_orders, capture_data
			FROM captures
			WHERE date = $1
			ORDER BY num_orders DESC
			LIMIT 1
		`, [today]);

		if (best?.capture_data) {
			const capture = best.capture_data;

			// Enrich capture orders from runs table (which has all orders seen during the game)
			const bestRun = await queryOne(`
				SELECT id FROM runs
				WHERE difficulty = $1 AND created_at::date = $2::date
				ORDER BY final_score DESC, id DESC
				LIMIT 1
			`, [best.difficulty, today]);

			if (bestRun) {
				const runOrders = await query(`
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
				`, [bestRun.id]);

				if (runOrders.length > (capture.orders?.length || 0)) {
					capture.orders = runOrders.map(r => ({
						id: r.order_id,
						items_required: r.items_required || [],
					}));
				}
			}

			const numOrders = capture.orders?.length || best.num_orders;
			return {
				defaultCapture: capture,
				defaultDifficulty: best.difficulty,
				defaultFileName: `${best.difficulty}_${best.date}_${numOrders}orders (DB)`,
				defaultNumOrders: numOrders,
			};
		}
	} catch {
		// DB unavailable — fall through to empty defaults
	}

	return EMPTY;
}
