import { query } from '$lib/db.server.js';

export async function load() {
	const tokens = await query('SELECT * FROM tokens ORDER BY created_at DESC');
	return { tokens };
}
