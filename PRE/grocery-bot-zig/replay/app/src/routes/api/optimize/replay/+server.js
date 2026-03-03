import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readFileSync, readdirSync, statSync } from 'fs';
import { query } from '$lib/db.server.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

export async function POST({ request }) {
	const { url, difficulty } = await request.json();

	if (!url) {
		return new Response(JSON.stringify({ error: 'Missing WebSocket URL' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	let pollInterval = null;
	let botProcess = null;
	let closed = false;
	let safetyTimeout = null;

	function cleanupShared(controller) {
		if (closed) return;
		closed = true;
		if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
		if (safetyTimeout) { clearTimeout(safetyTimeout); safetyTimeout = null; }
		if (botProcess && !botProcess.killed) {
			try { botProcess.kill(); } catch (e) {}
		}
		try { controller?.close(); } catch (e) {}
	}

	const stream = new ReadableStream({
		start(controller) {
			let logFile = null;
			let lineCount = 0;
			let initSent = false;
			let lastState = null;
			let gameWidth = 0;
			let gameHeight = 0;
			let botCount = 0;
			let detectedDiff = difficulty || 'auto';
			const startTime = Date.now();
			const MAX_RUNTIME = 180000;

			function cleanup() { cleanupShared(controller); }

			safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: replay took too long' });
				cleanup();
			}, MAX_RUNTIME);

			function sendEvent(type, data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
				} catch (e) {}
			}

			function findLogFile() {
				try {
					const files = readdirSync(GPU_DIR)
						.filter(f => f.startsWith('game_log_') && f.endsWith('.jsonl'))
						.map(f => ({ name: f, mtime: statSync(resolve(GPU_DIR, f)).mtimeMs }))
						.filter(f => f.mtime >= startTime)
						.sort((a, b) => b.mtime - a.mtime);
					if (files.length > 0) return resolve(GPU_DIR, files[0].name);
				} catch (e) {}
				return null;
			}

			function pollLogFile() {
				if (!logFile) {
					logFile = findLogFile();
					if (!logFile) return;
				}

				try {
					const content = readFileSync(logFile, 'utf-8');
					const lines = content.split('\n').filter(l => l.trim());

					while (lineCount < lines.length) {
						const line = lines[lineCount];
						lineCount++;

						if ((lineCount - 1) % 2 === 0) {
							let state;
							try { state = JSON.parse(line); } catch (e) { continue; }

							if (state.type === 'game_over') {
								sendEvent('game_over', { score: state.score });
								continue;
							}
							if (state.round === undefined) continue;

							lastState = state;
							const w = state.grid?.width || state.width || 0;
							const h = state.grid?.height || state.height || 0;
							gameWidth = w;
							gameHeight = h;
							botCount = state.bots?.length || 0;

							// Detect difficulty from bots
							if (state.round === 0) {
								if (botCount <= 1) detectedDiff = 'easy';
								else if (botCount <= 3) detectedDiff = 'medium';
								else if (botCount <= 5) detectedDiff = 'hard';
								else detectedDiff = 'expert';
							}

							if (!initSent && state.round === 0) {
								let walls = [];
								let shelves = [];

								if (state.grid?.walls) {
									const shelfPositions = new Set();
									for (const item of (state.items || [])) {
										shelfPositions.add(`${item.position[0]},${item.position[1]}`);
									}
									for (const w of state.grid.walls) walls.push(w);
									shelves = [...shelfPositions].map(s => {
										const [x, y] = s.split(',').map(Number);
										return [x, y];
									});
								} else if (Array.isArray(state.grid)) {
									for (let y = 0; y < h; y++) {
										for (let x = 0; x < w; x++) {
											const cell = state.grid[y]?.[x];
											if (cell === 'wall') walls.push([x, y]);
											else if (cell === 'shelf') shelves.push([x, y]);
										}
									}
								}

								sendEvent('init', {
									width: w, height: h, walls, shelves,
									items: state.items || [], drop_off: state.drop_off,
									spawn: [w - 2, h - 2], bot_count: botCount,
									max_rounds: state.max_rounds || 300,
								});
								initSent = true;
							}

							sendEvent('round', {
								round: state.round,
								bots: state.bots || [],
								orders: state.orders || [],
								score: state.score || 0,
							});
						} else {
							try {
								const resp = JSON.parse(line);
								if (resp.actions) {
									sendEvent('actions', {
										round: lastState?.round || 0,
										actions: resp.actions,
									});
								}
							} catch (e) {}
						}
					}
				} catch (e) {}
			}

			sendEvent('status', { message: `Replaying best solution...` });

			const args = ['replay_solution.py', url];
			if (difficulty && difficulty !== 'auto') {
				args.push('--difficulty', difficulty);
			}

			botProcess = spawn('python', args, {
				cwd: GPU_DIR,
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let stderrBuffer = '';
			botProcess.stderr.on('data', (data) => {
				stderrBuffer += data.toString();
				const lines = stderrBuffer.split('\n');
				stderrBuffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.trim()) continue;
					sendEvent('log', { text: line.trim() });

					const match = line.match(/R(\d+)\/(\d+) Score:(\d+)/);
					if (match) {
						sendEvent('progress', {
							round: parseInt(match[1]),
							max_rounds: parseInt(match[2]),
							score: parseInt(match[3]),
						});
					}

					const gameOver = line.match(/GAME_OVER Score:(\d+)/);
					if (gameOver) {
						sendEvent('final_score', { score: parseInt(gameOver[1]) });
					}
				}
			});

			botProcess.stdout.on('data', () => {});

			botProcess.on('close', async (code) => {
				pollLogFile();
				sendEvent('done', { code, message: `Replay finished (exit ${code})` });

				// DB import handled by replay_solution.py via import_logs.py

				cleanup();
			});

			botProcess.on('error', (err) => {
				sendEvent('error', { message: `Failed to spawn: ${err.message}` });
				cleanup();
			});

			pollInterval = setInterval(pollLogFile, 150);
		},
		cancel() {
			cleanupShared(null);
		}
	});

	return new Response(stream, {
		headers: {
			'Content-Type': 'text/event-stream',
			'Cache-Control': 'no-cache',
			'Connection': 'keep-alive',
		},
	});
}
