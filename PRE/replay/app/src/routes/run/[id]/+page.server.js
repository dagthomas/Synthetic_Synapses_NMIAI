import { query, queryOne } from '$lib/db.server.js';
import { error } from '@sveltejs/kit';

export async function load({ params }) {
	const run = await queryOne(`SELECT * FROM runs WHERE id = $1`, [params.id]);
	if (!run) throw error(404, 'Run not found');

	const rounds = await query(
		`SELECT round_number, bots, orders, actions, score, events
		 FROM rounds WHERE run_id = $1 ORDER BY round_number`,
		[params.id]
	);

	if (rounds.length === 0) throw error(404, 'Run has no round data');

	return { run, rounds };
}
