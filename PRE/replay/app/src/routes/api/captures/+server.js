/**
 * GET /api/captures?difficulty=nightmare&date=2026-03-07
 *
 * Returns the best capture for a given difficulty + date.
 * "Best" = most orders. Reads from DB (same source as /captures page).
 * If no date given, uses today (UTC).
 */
import { queryOne } from '$lib/db.server.js';

export async function GET({ url }) {
	const difficulty = url.searchParams.get('difficulty') || 'nightmare';
	const date = url.searchParams.get('date') || new Date().toISOString().slice(0, 10);

	try {
		const row = await queryOne(`
			SELECT difficulty, date, num_orders, capture_data
			FROM captures
			WHERE difficulty = $1 AND date = $2
		`, [difficulty, date]);

		if (row?.capture_data) {
			const data = row.capture_data;
			return Response.json({
				found: true,
				difficulty,
				date,
				filename: `${difficulty}_${date}_${row.num_orders}orders (DB)`,
				orders: row.num_orders,
				num_bots: data.num_bots || 0,
				capture: data,
			});
		}
	} catch {
		// DB unavailable
	}

	return Response.json({ found: false, difficulty, date });
}
