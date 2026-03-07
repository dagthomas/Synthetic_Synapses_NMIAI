import { resolve } from 'path';
import { readdir, readFile } from 'fs/promises';
import { GPU_DIR } from '$lib/paths.server.js';

const CAPTURES_DIR = resolve(GPU_DIR, 'captures');

// Extract date from filename like "expert_2026-03-05.json" or "expert_2026-03-06_095709_2orders.json"
function parseCaptureFile(filename) {
	const m = filename.match(/^(\w+?)_(\d{4}-\d{2}-\d{2})(?:_(\d{6})_(\d+)orders)?\.json$/);
	if (!m) return null;
	return {
		difficulty: m[1],
		date: m[2],
		time: m[3] || null,
		filename
	};
}

export async function load() {
	let files;
	try {
		files = await readdir(CAPTURES_DIR);
	} catch {
		return { days: {} };
	}

	const jsonFiles = files.filter(f => f.endsWith('.json'));

	// Group by difficulty+date, find best (most orders) per group
	const groups = {}; // key: "difficulty|date" -> { files: [...], best: null }

	for (const fname of jsonFiles) {
		const parsed = parseCaptureFile(fname);
		if (!parsed) continue; // skip files like captured_orders_expert.json

		const key = `${parsed.difficulty}|${parsed.date}`;
		if (!groups[key]) {
			groups[key] = { difficulty: parsed.difficulty, date: parsed.date, fileCount: 0, best: null };
		}
		groups[key].fileCount++;

		// Read file to count orders
		try {
			const raw = await readFile(resolve(CAPTURES_DIR, fname), 'utf-8');
			const data = JSON.parse(raw);
			const orders = data.orders || [];
			const total = data.total_orders_discovered || orders.length;
			const capturedAt = data.captured_at || null;

			if (!groups[key].best || total > groups[key].best.total) {
				groups[key].best = {
					filename: fname,
					total,
					capturedAt,
					orders: orders.map((o, i) => ({
						id: o.id || `order_${i}`,
						items: o.items_required || []
					}))
				};
			}
		} catch {
			// skip unreadable files
		}
	}

	// Sort by date descending, group by difficulty
	const entries = Object.values(groups).sort((a, b) => b.date.localeCompare(a.date));

	// Group by difficulty
	const byDifficulty = {};
	const difficultyOrder = ['nightmare', 'expert', 'hard', 'medium', 'easy'];
	for (const entry of entries) {
		if (!byDifficulty[entry.difficulty]) {
			byDifficulty[entry.difficulty] = [];
		}
		byDifficulty[entry.difficulty].push(entry);
	}

	// Sort difficulties by priority
	const sorted = {};
	for (const diff of difficultyOrder) {
		if (byDifficulty[diff]) sorted[diff] = byDifficulty[diff];
	}
	// Add any remaining
	for (const [diff, entries] of Object.entries(byDifficulty)) {
		if (!sorted[diff]) sorted[diff] = entries;
	}

	return { byDifficulty: sorted };
}
