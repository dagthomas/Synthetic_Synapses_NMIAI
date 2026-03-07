import { readFileSync, rmSync, existsSync } from 'fs';
import { resolve } from 'path';
import { GPU_DIR } from '$lib/paths.server.js';

const SOLUTIONS_DIR = resolve(GPU_DIR, 'solutions');

const DIFFICULTIES = ['easy', 'medium', 'hard', 'expert', 'nightmare'];

function loadMeta(difficulty) {
	try {
		const path = resolve(SOLUTIONS_DIR, difficulty, 'meta.json');
		const data = readFileSync(path, 'utf-8');
		return JSON.parse(data);
	} catch {
		return null;
	}
}

function countOrders(difficulty) {
	try {
		const path = resolve(SOLUTIONS_DIR, difficulty, 'capture.json');
		const data = JSON.parse(readFileSync(path, 'utf-8'));
		return (data.orders || []).length;
	} catch {
		return 0;
	}
}

export async function GET() {
	const solutions = {};
	for (const d of DIFFICULTIES) {
		const meta = loadMeta(d);
		const orders = countOrders(d);
		solutions[d] = meta ? { ...meta, orders } : (orders > 0 ? { orders } : null);
	}

	return new Response(JSON.stringify(solutions), {
		headers: { 'Content-Type': 'application/json' },
	});
}

export async function DELETE({ request }) {
	const { difficulty } = await request.json();

	const diffs = difficulty === 'all' ? DIFFICULTIES :
		DIFFICULTIES.includes(difficulty) ? [difficulty] : [];

	if (diffs.length === 0) {
		return new Response(JSON.stringify({ error: 'Invalid difficulty' }), { status: 400 });
	}

	const cleared = [];
	for (const d of diffs) {
		const dir = resolve(SOLUTIONS_DIR, d);
		if (existsSync(dir)) {
			for (const file of ['best.json', 'capture.json', 'meta.json', 'route_table.json']) {
				const p = resolve(dir, file);
				if (existsSync(p)) {
					rmSync(p);
				}
			}
			cleared.push(d);
		}
	}

	return new Response(JSON.stringify({ cleared }), {
		headers: { 'Content-Type': 'application/json' },
	});
}
