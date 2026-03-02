import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readFileSync, readdirSync, statSync } from 'fs';
import { query } from '$lib/db.server.js';

// Resolve BOT_DIR from this file's location (6 levels up from src/routes/api/run-live/)
const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

function getBotCommand(difficulty, solver) {
	if (solver === 'gpu') {
		return {
			cmd: 'python',
			args: ['capture_and_solve_stream.py'],  // ws_url added by caller, then difficulty
			cwd: GPU_DIR,
			logDir: GPU_DIR,
			streaming: true,  // reads stdout JSON lines instead of polling log file
			extraArgs: [difficulty],  // appended after ws_url
		};
	}
	if (solver === 'python') {
		return {
			cmd: 'python',
			args: ['live_solver.py'],  // ws_url added by caller
			cwd: GPU_DIR,
			logDir: GPU_DIR,
		};
	}
	// Zig: try difficulty-specific executable first, fall back to generic
	const specific = resolve(BOT_DIR, 'zig-out', 'bin', `grocery-bot-${difficulty}.exe`);
	let botPath;
	try {
		statSync(specific);
		botPath = specific;
	} catch {
		botPath = resolve(BOT_DIR, 'zig-out', 'bin', 'grocery-bot.exe');
	}
	return {
		cmd: botPath,
		args: [],  // ws_url added by caller
		cwd: BOT_DIR,
		logDir: BOT_DIR,
	};
}

export async function POST({ request }) {
	const { url, difficulty, solver } = await request.json();

	if (!url || !url.startsWith('wss://')) {
		return new Response(JSON.stringify({ error: 'Invalid WebSocket URL' }), { status: 400 });
	}

	const encoder = new TextEncoder();

	// Shared state accessible from both start() and cancel()
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
		try { controller.close(); } catch (e) {}
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
			const startTime = Date.now();
			const MAX_RUNTIME = solver === 'gpu' ? 600000 : 180000; // GPU: 10min, others: 3min

			function cleanup() { cleanupShared(controller); }

			safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: game took too long' });
				cleanup();
			}, MAX_RUNTIME);

			function sendEvent(type, data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
				} catch (e) {}
			}

			const botInfo = getBotCommand(difficulty || 'auto', solver || 'zig');

			function findLogFile() {
				try {
					const searchDir = botInfo.logDir;
					const files = readdirSync(searchDir)
						.filter(f => f.startsWith('game_log_') && f.endsWith('.jsonl'))
						.map(f => ({ name: f, mtime: statSync(resolve(searchDir, f)).mtimeMs }))
						.filter(f => f.mtime >= startTime) // Only files created AFTER we started
						.sort((a, b) => b.mtime - a.mtime);
					if (files.length > 0) {
						return resolve(searchDir, files[0].name);
					}
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

						// Even indices (0,2,4...) = server state, odd = our response
						if ((lineCount - 1) % 2 === 0) {
							let state;
							try { state = JSON.parse(line); } catch (e) { continue; }

							if (state.type === 'game_over') {
								sendEvent('game_over', { score: state.score });
								continue;
							}

							if (state.round === undefined) continue;

							lastState = state;

							// Production server format: grid.width, grid.height, grid.walls
							const w = state.grid?.width || state.width || 0;
							const h = state.grid?.height || state.height || 0;
							gameWidth = w;
							gameHeight = h;
							botCount = state.bots?.length || 0;

							// First round: send init data
							if (!initSent && state.round === 0) {
								// Walls come from grid.walls (production) or need to be parsed from 2D grid
								let walls = [];
								let shelves = [];

								if (state.grid?.walls) {
									// Production format: grid.walls is array of [x,y]
									const wallSet = new Set(state.grid.walls.map(w => `${w[0]},${w[1]}`));
									// Items are on shelves; shelf = item position
									const shelfPositions = new Set();
									for (const item of (state.items || [])) {
										const pos = item.position;
										shelfPositions.add(`${pos[0]},${pos[1]}`);
									}
									// Separate walls from shelves
									for (const w of state.grid.walls) {
										walls.push(w);
									}
									shelves = [...shelfPositions].map(s => {
										const [x, y] = s.split(',').map(Number);
										return [x, y];
									});
								} else if (Array.isArray(state.grid)) {
									// Sim server format: 2D array
									for (let y = 0; y < h; y++) {
										for (let x = 0; x < w; x++) {
											const cell = state.grid[y]?.[x];
											if (cell === 'wall') walls.push([x, y]);
											else if (cell === 'shelf') shelves.push([x, y]);
										}
									}
								}

								sendEvent('init', {
									width: w,
									height: h,
									walls,
									shelves,
									items: state.items || [],
									drop_off: state.drop_off,
									spawn: [w - 2, h - 2],
									bot_count: botCount,
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
							// Our action response
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

			const solverLabels = { zig: 'Zig', python: 'Python', gpu: 'GPU' };
			const solverLabel = solverLabels[solver || 'zig'] || 'Zig';
			sendEvent('status', { message: `Starting ${solverLabel} bot: ${botInfo.cmd}` });
			sendEvent('status', { message: `CWD: ${botInfo.cwd}` });

			// Verify bot executable/script exists before spawning
			if (solver === 'gpu') {
				const scriptPath = resolve(botInfo.cwd, 'capture_and_solve_stream.py');
				try {
					statSync(scriptPath);
				} catch {
					sendEvent('error', { message: `GPU solver not found: ${scriptPath}` });
					sendEvent('done', { code: -1, message: 'capture_and_solve_stream.py not found in grocery-bot-gpu/' });
					cleanup();
					return;
				}
			} else if (solver === 'python') {
				const scriptPath = resolve(botInfo.cwd, 'live_solver.py');
				try {
					statSync(scriptPath);
				} catch {
					sendEvent('error', { message: `Python solver not found: ${scriptPath}` });
					sendEvent('done', { code: -1, message: 'live_solver.py not found in grocery-bot-gpu/' });
					cleanup();
					return;
				}
			} else {
				try {
					statSync(botInfo.cmd);
				} catch {
					sendEvent('error', { message: `Bot executable not found: ${botInfo.cmd}` });
					sendEvent('done', { code: -1, message: 'Bot not found. Run SvelteKit locally (npm run dev), not in Docker.' });
					cleanup();
					return;
				}
			}

			const spawnArgs = [...botInfo.args, url, ...(botInfo.extraArgs || [])];
			botProcess = spawn(botInfo.cmd, spawnArgs, {
				cwd: botInfo.cwd,
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

			// GPU solver streams JSON lines on stdout; others use log file polling
			let stdoutBuffer = '';
			botProcess.stdout.on('data', (data) => {
				if (!botInfo.streaming) return;
				stdoutBuffer += data.toString();
				const lines = stdoutBuffer.split('\n');
				stdoutBuffer = lines.pop() || '';
				for (const line of lines) {
					if (!line.trim()) continue;
					try {
						const event = JSON.parse(line);
						// Forward GPU solver events to SSE with gpu_ prefix to avoid collision
						sendEvent('gpu_event', event);

						// Map key GPU events to standard dashboard events
						if (event.type === 'progress') {
							sendEvent('progress', {
								round: event.round,
								max_rounds: event.max_rounds || 300,
								score: event.score || 0,
							});
						} else if (event.type === 'solver_result') {
							sendEvent('final_score', { score: event.score });
						} else if (event.type === 'done') {
							sendEvent('final_score', { score: event.solver_score || event.score || 0 });
						}
					} catch (e) {}
				}
			});

			botProcess.on('close', async (code) => {
				if (!botInfo.streaming) pollLogFile();
				sendEvent('done', { code, message: `Bot exited with code ${code}` });

				// Insert to DB if we have valid data
				if (lastState && difficulty && gameWidth > 0) {
					try {
						const result = await query(
							`INSERT INTO runs (seed, difficulty, grid_width, grid_height, bot_count,
								final_score, items_delivered, orders_completed)
							VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id`,
							[0, difficulty, gameWidth, gameHeight, botCount,
							 lastState.score || 0, 0, 0]
						);
						sendEvent('db', { run_id: result[0]?.id, message: 'Saved to database' });
					} catch (e) {
						sendEvent('db_error', { message: e.message });
					}
				}

				cleanup();
			});

			botProcess.on('error', (err) => {
				sendEvent('error', { message: `Failed to spawn bot: ${err.message}` });
				cleanup();
			});

			if (!botInfo.streaming) {
				pollInterval = setInterval(pollLogFile, 150);
			}
		},
		cancel() {
			// Client disconnected — kill the bot process
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
