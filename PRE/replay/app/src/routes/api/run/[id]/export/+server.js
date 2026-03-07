import { query, queryOne } from '$lib/db.server.js';
import { error } from '@sveltejs/kit';

export async function GET({ params }) {
	const run = await queryOne(`SELECT * FROM runs WHERE id = $1`, [params.id]);
	if (!run) throw error(404, 'Run not found');

	const rounds = await query(
		`SELECT round_number, bots, orders, actions, score, events
		 FROM rounds WHERE run_id = $1 ORDER BY round_number`,
		[params.id]
	);

	const filename = `run-${run.id}-${run.difficulty}-seed${run.seed}.json`;

	return new Response(JSON.stringify({ run, rounds }, null, 2), {
		headers: {
			'Content-Type': 'application/json',
			'Content-Disposition': `attachment; filename="${filename}"`,
		},
	});
}
