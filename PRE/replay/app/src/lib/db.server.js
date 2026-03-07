import pg from 'pg';

const pool = new pg.Pool({
	connectionString: process.env.DATABASE_URL || 'postgres://grocery:grocery123@localhost:5433/grocery_bot',
	max: 10,
});

export async function query(text, params) {
	const res = await pool.query(text, params);
	return res.rows;
}

export async function queryOne(text, params) {
	const res = await pool.query(text, params);
	return res.rows[0] || null;
}
