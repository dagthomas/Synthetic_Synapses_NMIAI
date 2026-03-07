import { spawn, spawnSync } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync, statSync, readFileSync, writeFileSync, existsSync, copyFileSync } from 'fs';
import { createCleanup, createSendEvent } from '$lib/sse.server.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');
const ZIG_BOT_DIR = BOT_DIR;

const GPU_PARAMS = {
	easy:   { coldTime: 20,  warmTime: 15, coldOrd: 1, coldRef: 5,  warmRef: 2, maxStates: null,   plateau: 3 },
	medium: { coldTime: 25,  warmTime: 20, coldOrd: 1, coldRef: 5,  warmRef: 3, maxStates: null,   plateau: 3 },
	hard:   { coldTime: 45,  warmTime: 25, coldOrd: 3, coldRef: 5,  warmRef: 3, maxStates: 50000,  plateau: 8 },
	expert: { coldTime: 45,  warmTime: 25, coldOrd: 1, coldRef: 3,  warmRef: 2, maxStates: 50000,  plateau: 8 },
};

// Deep mode: when order discovery stalls, use bigger search
const DEEP_PARAMS = { maxStates: 100000, orderings: 3, refineIters: 8 };

export async function POST({ request }) {
	const {
		url,
		timeBudget = 280,       // seconds total (~288s token window, leave margin)
		postOptimizeTime = 30,  // post-opt for initial live game
		gpuOptimizeTime = 20,   // time per offline GPU optimize pass (keep short to allow more iterations)
	} = await request.json();

	if (!url) {
		return new Response(JSON.stringify({ error: 'Missing WebSocket URL' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };

	const stream = new ReadableStream({
		start(controller) {
			const pipelineStart = Date.now();
			const MAX_RUNTIME = (timeBudget + 300) * 1000;

			const cleanup = createCleanup(ctx, controller);
			const sendEvent = createSendEvent(ctx, controller, encoder);

			ctx.safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Pipeline safety timeout reached' });
				cleanup();
			}, MAX_RUNTIME);

			ctx.heartbeatInterval = setInterval(() => {
				if (ctx.closed) return;
				try { controller.enqueue(encoder.encode(': ping\n\n')); } catch (e) { /* stream closed by client */ }
			}, 10000);

			function elapsedSecs() {
				return (Date.now() - pipelineStart) / 1000;
			}

			function remaining() {
				return Math.max(0, timeBudget - elapsedSecs());
			}

			// ── Capture orders from game log (synchronous) ───────────────
			function captureOrders(difficulty, iter) {
				return new Promise((res) => {
					const logPath = findNewestLog(GPU_DIR,pipelineStart);
					if (!logPath || !difficulty) { res(0); return; }

					sendEvent('log', { text: `[capture] Extracting orders from ${logPath.split(/[\\/]/).pop()}...`, _iter: iter });

					const proc = spawn('python', [
						'capture_from_game_log.py', logPath, difficulty,
					], { cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'] });

					let captured = 0;
					proc.stderr.on('data', (d) => {
						const t = d.toString().trim();
						if (t) {
							sendEvent('log', { text: `[capture] ${t}`, _iter: iter });
							const m = t.match(/(\d+) orders/);
							if (m) captured = parseInt(m[1]);
						}
					});
					proc.on('close', () => res(captured));
					proc.on('error', () => res(0));
				});
			}

			// ── Detect difficulty from game log ──────────────────────────
			function detectDifficultyFromLog(logPath) {
				try {
					const content = readFileSync(logPath, 'utf-8');
					const lines = content.split('\n').filter(l => l.trim());
					for (const line of lines) {
						try {
							const state = JSON.parse(line);
							if (state.type === 'game_state' || state.bots) {
								const botCount = (state.bots || []).length;
								if (botCount === 1) return 'easy';
								if (botCount === 3) return 'medium';
								if (botCount === 5) return 'hard';
								if (botCount >= 10) return 'expert';
								const w = state.grid?.width || 0;
								if (w <= 12) return 'easy';
								if (w <= 16) return 'medium';
								if (w <= 22) return 'hard';
								return 'expert';
							}
						} catch (e) { /* ignore malformed JSON line */ }
					}
				} catch (e) { /* log file may not exist yet */ }
				return null;
			}

			// ── Find newest game log in a directory ─────────────────────
			function findNewestLog(dir, afterMs = 0) {
				try {
					const files = readdirSync(dir)
						.filter(f => f.startsWith('game_log_') && f.endsWith('.jsonl'))
						.map(f => ({ name: f, mtime: statSync(resolve(dir, f)).mtimeMs }))
						.filter(f => f.mtime >= afterMs)
						.sort((a, b) => b.mtime - a.mtime);
					return files.length > 0 ? resolve(dir, files[0].name) : null;
				} catch (e) { return null; }
			}

			// ── Parse difficulty from JWT in WebSocket URL ──────────────
			function parseDifficultyFromUrl(wsUrl) {
				try {
					const m = wsUrl.match(/[?&]token=([^&]+)/);
					if (!m) return null;
					const parts = m[1].split('.');
					if (parts.length < 2) return null;
					const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString());
					return payload.difficulty || null;
				} catch (e) { return null; }
			}

			// ── Count orders from DB ────────────────────────────────────
			function getOrderCount(diff) {
				if (!diff) return 0;
				try {
					const result = spawnSync('python', ['-u', 'db_query.py', 'order_count', diff], { cwd: GPU_DIR, timeout: 5000 });
					const data = JSON.parse(result.stdout.toString());
					return data.count || 0;
				} catch (e) { return 0; }
			}

			// ── Check if a GPU solution exists in DB ────────────────────
			function solutionExists(diff) {
				if (!diff) return false;
				try {
					const result = spawnSync('python', ['-u', 'db_query.py', 'solution_exists', diff], { cwd: GPU_DIR, timeout: 5000 });
					const data = JSON.parse(result.stdout.toString());
					return data.exists || false;
				} catch (e) { return false; }
			}

			// ── Phase: Zig bot play (fallback for iter 0) ───────────────
			function runZigBot(iter) {
				return new Promise((resolvePhase) => {
					sendEvent('iter_start', {
						iter, phase: 'zig_bot',
						elapsed: elapsedSecs(), remaining: remaining(),
					});

					const exe = resolve(ZIG_BOT_DIR, 'zig-out', 'bin', 'grocery-bot.exe');
					if (!existsSync(exe)) {
						sendEvent('error', { message: `Zig bot exe not found: ${exe}` });
						resolvePhase({ score: 0, difficulty: null });
						return;
					}

					const spawnMs = Date.now();
					ctx.process = spawn(exe, [url], {
						cwd: ZIG_BOT_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let score = 0, difficulty = null;

					ctx.process.stderr.on('data', (data) => {
						for (const line of data.toString().split('\n')) {
							if (!line.trim()) continue;
							sendEvent('log', { text: `[zig] ${line.trim()}`, _iter: iter });

							const roundMatch = line.match(/R(\d+)\/(\d+)\s+Score:(\d+)/);
							if (roundMatch) {
								const rnd = parseInt(roundMatch[1]);
								const maxR = parseInt(roundMatch[2]);
								score = parseInt(roundMatch[3]);
								if (rnd % 10 === 0 || rnd === maxR) {
									sendEvent('round', { round: rnd, score, _iter: iter });
								}
							}
							const gameOver = line.match(/GAME_OVER\s+Score:(\d+)/);
							if (gameOver) score = parseInt(gameOver[1]);
							const dimMatch = line.match(/width=(\d+)\s+height=(\d+)\s+bots=(\d+)/);
							if (dimMatch) {
								const bots = parseInt(dimMatch[3]);
								if (bots === 1) difficulty = 'easy';
								else if (bots === 3) difficulty = 'medium';
								else if (bots === 5) difficulty = 'hard';
								else if (bots >= 10) difficulty = 'expert';
							}
						}
					});

					ctx.process.stdout.on('data', () => {});

					ctx.process.on('close', (code) => {
						sendEvent('log', { text: `[zig] Process exited with code ${code}`, _iter: iter });

						const logPath = findNewestLog(ZIG_BOT_DIR, spawnMs);
						if (logPath) {
							if (!difficulty) {
								difficulty = detectDifficultyFromLog(logPath);
							}

							try {
								const logContent = readFileSync(logPath, 'utf8');
								const firstLine = logContent.split('\n')[0];
								if (firstLine) {
									const state = JSON.parse(firstLine);
									if (state.type === 'game_state' && state.grid) {
										const w = state.grid.width, h = state.grid.height;
										const walls = state.grid.walls || [];
										const shelfSet = new Set();
										const items = (state.items || []).map(it => ({
											id: it.id, type: it.type, position: it.position
										}));
										for (const it of items) {
											shelfSet.add(`${it.position[0]},${it.position[1]}`);
										}
										const shelves = [...shelfSet].map(s => {
											const [x, y] = s.split(',').map(Number);
											return [x, y];
										});
										sendEvent('init', {
											width: w, height: h, walls, shelves,
											drop_off: state.drop_off || [1, h - 2],
											spawn: state.bots?.[0]?.position || [w - 2, h - 2],
											items,
											num_bots: state.bots ? state.bots.length : 1,
											difficulty,
										});
									}
								}
							} catch (e) {
								sendEvent('log', { text: `[zig] Failed to parse log for grid: ${e.message}`, _iter: iter });
							}

							const destPath = resolve(GPU_DIR, logPath.split(/[\\/]/).pop());
							try {
								copyFileSync(logPath, destPath);
								sendEvent('log', { text: `[zig] Copied log to ${destPath.split(/[\\/]/).pop()}`, _iter: iter });
							} catch (e) {
								sendEvent('log', { text: `[zig] Failed to copy log: ${e.message}`, _iter: iter });
							}
						} else {
							sendEvent('log', { text: '[zig] No game log found', _iter: iter });
						}

						resolvePhase({ score, difficulty, logPath });
					});
					ctx.process.on('error', (err) => {
						sendEvent('error', { message: `Zig bot failed: ${err.message}` });
						resolvePhase({ score: 0, difficulty: null, logPath: null });
					});
				});
			}

			// ── Phase: Live play (iteration 0) ──────────────────────────
			function runLivePlay(iter, postOpt, opts = {}) {
				return new Promise((resolvePhase) => {
					sendEvent('iter_start', {
						iter, phase: 'live_play',
						elapsed: elapsedSecs(), remaining: remaining(),
						post_optimize_time: postOpt,
					});

					const args = [
						'-u', 'live_gpu_stream.py', url,
						'--save', '--json-stream', '--record',
						'--pipeline-mode',
						'--post-optimize-time', String(postOpt),
					];
					if (opts.preloadCapture) args.push('--preload-capture');

					ctx.process = spawn('python', args, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let buf = '';
					let score = 0, gameScore = 0, difficulty = null, orders = 0;

					ctx.process.stdout.on('data', (data) => {
						buf += data.toString();
						const lines = buf.split('\n');
						buf = lines.pop() || '';
						for (const line of lines) {
							if (!line.trim()) continue;
							try {
								const evt = JSON.parse(line);
								const type = evt.type; delete evt.type;
								evt._iter = iter;
								evt._elapsed = elapsedSecs();
								evt._remaining = remaining();
								if (type === 'game_over') gameScore = evt.score || 0;
								if (type === 'pipeline_done') {
									score = evt.final_score || evt.plan_score || gameScore;
									difficulty = evt.difficulty;
								}
								if (type === 'init') difficulty = evt.difficulty;
								if (type === 'round' && evt.orders) orders = evt.orders.length;
								sendEvent(type, evt);
							} catch (e) { /* ignore malformed JSON output */ }
						}
					});

					ctx.process.stderr.on('data', (data) => {
						for (const line of data.toString().trim().split('\n')) {
							if (line.trim()) sendEvent('log', { text: line.trim(), _iter: iter });
						}
					});

					ctx.process.on('close', (code) => {
						resolvePhase({ score, gameScore, difficulty, orders, exitCode: code });
					});
					ctx.process.on('error', (err) => {
						sendEvent('error', { message: `Live play failed: ${err.message}` });
						resolvePhase({ score: 0, gameScore: 0, difficulty: null, orders: 0, exitCode: 1 });
					});
				});
			}

			// ── Phase: Offline GPU optimize ──────────────────────────────
			function runGpuOptimize(difficulty, iter, maxTime, opts = {}) {
				return new Promise((resolvePhase) => {
					sendEvent('optimize_phase_start', {
						_iter: iter, difficulty, max_time: maxTime,
						warm_only: !!opts.warmOnly,
						speed_bonus: opts.speedBonus || 0,
						_elapsed: elapsedSecs(), _remaining: remaining(),
					});

					const args = [
						'-u', 'optimize_and_save.py', difficulty,
						'--max-time', String(Math.floor(maxTime)),
					];
					if (opts.warmOnly) args.push('--warm-only');
					if (opts.orderings) args.push('--orderings', String(opts.orderings));
					if (opts.refineIters) args.push('--refine-iters', String(opts.refineIters));
					if (opts.maxStates) args.push('--max-states', String(opts.maxStates));
					if (opts.speedBonus) args.push('--speed-bonus', String(opts.speedBonus));

					ctx.process = spawn('python', args, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let buf = '';
					let score = 0, prevScore = 0;

					ctx.process.stdout.on('data', (data) => {
						buf += data.toString();
						const lines = buf.split('\n');
						buf = lines.pop() || '';
						for (const line of lines) {
							if (!line.trim()) continue;
							try {
								const evt = JSON.parse(line);
								evt._iter = iter;
								evt._elapsed = elapsedSecs();
								evt._remaining = remaining();
								if (evt.type === 'optimize_done') {
									score = evt.score || 0;
									prevScore = evt.prev_score || 0;
								}
								sendEvent(evt.type, evt);
							} catch (e) { /* ignore malformed JSON output */ }
						}
					});

					ctx.process.stderr.on('data', (data) => {
						for (const line of data.toString().trim().split('\n')) {
							if (line.trim()) sendEvent('log', { text: `[gpu] ${line.trim()}`, _iter: iter });
						}
					});

					ctx.process.on('close', () => {
						resolvePhase({ score, prevScore });
					});
					ctx.process.on('error', (err) => {
						sendEvent('error', { message: `GPU optimize failed: ${err.message}` });
						resolvePhase({ score: 0, prevScore: 0 });
					});
				});
			}

			// ── Phase: Replay optimized solution ─────────────────────────
			function runReplay(difficulty, iter) {
				return new Promise((resolvePhase) => {
					sendEvent('replay_phase_start', {
						_iter: iter, difficulty,
						_elapsed: elapsedSecs(), _remaining: remaining(),
					});

					const replayArgs = ['replay_solution.py', url, '--difficulty', difficulty];

					const replayStartMs = Date.now();
					ctx.process = spawn('python', replayArgs, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let score = 0;
					let logPath = null;
					let stderrBuf = '';

					let pollLogFile = null;
					let lineCount = 0;
					let initSent = false;
					let lastState = null;

					const pollInterval = setInterval(() => {
						if (!pollLogFile) {
							pollLogFile = findNewestLog(GPU_DIR,replayStartMs);
							if (!pollLogFile) return;
						}
						try {
							const content = readFileSync(pollLogFile, 'utf-8');
							const lines = content.split('\n').filter(l => l.trim());
							while (lineCount < lines.length) {
								const line = lines[lineCount];
								lineCount++;
								if ((lineCount - 1) % 2 === 0) {
									let state;
									try { state = JSON.parse(line); } catch (e) { continue; }
									if (state.type === 'game_over') {
										score = state.score || score;
										sendEvent('game_over', { score: state.score, _iter: iter });
										continue;
									}
									if (state.round === undefined) continue;
									lastState = state;
									const w = state.grid?.width || state.width || 0;
									const h = state.grid?.height || state.height || 0;
									const botCount = state.bots?.length || 0;
									if (!initSent && state.round === 0) {
										let walls = [], shelves = [];
										if (state.grid?.walls) {
											const shelfPos = new Set();
											for (const item of (state.items || []))
												shelfPos.add(`${item.position[0]},${item.position[1]}`);
											for (const w of state.grid.walls) walls.push(w);
											shelves = [...shelfPos].map(s => {
												const [x, y] = s.split(',').map(Number);
												return [x, y];
											});
										}
										sendEvent('init', {
											width: w, height: h, walls, shelves,
											items: state.items || [], drop_off: state.drop_off,
											spawn: [w - 2, h - 2], bot_count: botCount,
											max_rounds: state.max_rounds || 300,
											difficulty, _iter: iter,
										});
										initSent = true;
									}
									sendEvent('round', {
										round: state.round, bots: state.bots || [],
										orders: state.orders || [], score: state.score || 0,
										_iter: iter,
									});
								}
							}
						} catch (e) { /* log file may not be readable yet */ }
					}, 200);

					ctx.process.stderr.on('data', (data) => {
						stderrBuf += data.toString();
						const lines = stderrBuf.split('\n');
						stderrBuf = lines.pop() || '';
						for (const line of lines) {
							if (!line.trim()) continue;
							sendEvent('log', { text: `[replay] ${line.trim()}`, _iter: iter });
							const gameOver = line.match(/GAME_OVER Score:(\d+)/);
							if (gameOver) score = parseInt(gameOver[1]);
							const logMatch = line.match(/Log saved: (.+)/);
							if (logMatch) logPath = logMatch[1].trim();
						}
					});

					ctx.process.stdout.on('data', () => {});

					ctx.process.on('close', (code) => {
						clearInterval(pollInterval);
						if (pollLogFile) {
							try {
								const content = readFileSync(pollLogFile, 'utf-8');
								const lines = content.split('\n').filter(l => l.trim());
								while (lineCount < lines.length) {
									const line = lines[lineCount]; lineCount++;
									if ((lineCount - 1) % 2 === 0) {
										try {
											const state = JSON.parse(line);
											if (state.type === 'game_over') score = state.score || score;
										} catch (e) { /* ignore malformed JSON line */ }
									}
								}
							} catch (e) { /* log file may have been removed */ }
						}
						sendEvent('replay_phase_done', {
							_iter: iter, score, exit_code: code,
							_elapsed: elapsedSecs(), _remaining: remaining(),
						});
						resolvePhase({ score, logPath });
					});
					ctx.process.on('error', (err) => {
						clearInterval(pollInterval);
						sendEvent('error', { message: `Replay failed: ${err.message}` });
						resolvePhase({ score: 0, logPath: null });
					});
				});
			}

			// ── Phase: Export DP plan for Zig replay ─────────────────────
			function exportDpPlan(difficulty) {
				return new Promise((res) => {
					const proc = spawn('python', [
						'-u', 'export_plan_for_zig.py', difficulty,
					], { cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'] });

					proc.stderr.on('data', (d) => {
						const t = d.toString().trim();
						if (t) sendEvent('log', { text: `[export] ${t}` });
					});
					proc.on('close', (code) => res(code === 0));
					proc.on('error', () => res(false));
				});
			}

			// ── Phase: Zig bot with DP replay ────────────────────────────
			// Much faster and more reliable than Python replay on slow connections.
			// Zig sends cached DP responses when synced, falls back to reactive strategy when desynced.
			function runZigDpReplay(difficulty, iter) {
				return new Promise((resolvePhase) => {
					sendEvent('replay_phase_start', {
						_iter: iter, difficulty,
						_elapsed: elapsedSecs(), _remaining: remaining(),
					});

					const exe = resolve(ZIG_BOT_DIR, 'zig-out', 'bin', 'grocery-bot.exe');
					if (!existsSync(exe)) {
						sendEvent('log', { text: `[zig-replay] Missing exe, falling back to Python replay`, _iter: iter });
						resolvePhase({ score: 0, logPath: null, fallback: true });
						return;
					}

					// Export dp_plan and capture from DB to temp files
					const tmpDir = resolve(GPU_DIR, '.tmp');
					try { if (!existsSync(tmpDir)) { const { mkdirSync } = require('fs'); mkdirSync(tmpDir, { recursive: true }); } } catch (e) {}
					const dpPlan = resolve(tmpDir, `dp_plan_${difficulty}.json`);
					const captureJson = resolve(tmpDir, `capture_${difficulty}.json`);

					const dpResult = spawnSync('python', ['-u', 'db_query.py', 'export_dp_plan', difficulty, dpPlan], { cwd: GPU_DIR, timeout: 10000 });
					let hasDpPlan = false;
					try { hasDpPlan = JSON.parse(dpResult.stdout.toString()).ok; } catch (e) {}

					if (!hasDpPlan) {
						sendEvent('log', { text: `[zig-replay] No dp_plan in DB, falling back to Python replay`, _iter: iter });
						resolvePhase({ score: 0, logPath: null, fallback: true });
						return;
					}

					const capResult = spawnSync('python', ['-u', 'db_query.py', 'export_capture', difficulty, captureJson], { cwd: GPU_DIR, timeout: 10000 });
					let hasCapture = false;
					try { hasCapture = JSON.parse(capResult.stdout.toString()).ok; } catch (e) {}

					const args = [exe, url, '--dp-plan', dpPlan];
					if (hasCapture) {
						args.push('--precomputed', captureJson);
					}

					const spawnMs = Date.now();
					ctx.process = spawn(args[0], args.slice(1), {
						cwd: ZIG_BOT_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let score = 0;
					let initSent = false;

					ctx.process.stderr.on('data', (data) => {
						for (const line of data.toString().split('\n')) {
							if (!line.trim()) continue;
							sendEvent('log', { text: `[zig-replay] ${line.trim()}`, _iter: iter });

							const roundMatch = line.match(/R(\d+)\/(\d+)\s+Score:(\d+)/);
							if (roundMatch) {
								const rnd = parseInt(roundMatch[1]);
								score = parseInt(roundMatch[3]);
								if (rnd % 10 === 0 || rnd === parseInt(roundMatch[2])) {
									sendEvent('round', { round: rnd, score, _iter: iter });
								}
							}
							const gameOver = line.match(/GAME_OVER\s+Score:(\d+)/);
							if (gameOver) score = parseInt(gameOver[1]);

							// Grid init from Zig output
							const dimMatch = line.match(/width=(\d+)\s+height=(\d+)\s+bots=(\d+)/);
							if (dimMatch && !initSent) {
								sendEvent('init', {
									width: parseInt(dimMatch[1]),
									height: parseInt(dimMatch[2]),
									num_bots: parseInt(dimMatch[3]),
									difficulty,
									_iter: iter,
								});
								initSent = true;
							}
						}
					});

					ctx.process.stdout.on('data', () => {});

					ctx.process.on('close', (code) => {
						// Find game log for order capture
						const logPath = findNewestLog(ZIG_BOT_DIR, spawnMs);
						if (logPath) {
							// Copy log to GPU_DIR for captureOrders
							const destPath = resolve(GPU_DIR, logPath.split(/[\\/]/).pop());
							try { copyFileSync(logPath, destPath); } catch (e) { /* ok */ }
						}

						sendEvent('replay_phase_done', {
							_iter: iter, score, exit_code: code,
							_elapsed: elapsedSecs(), _remaining: remaining(),
						});
						resolvePhase({ score, logPath, fallback: false });
					});
					ctx.process.on('error', (err) => {
						sendEvent('error', { message: `Zig DP replay failed: ${err.message}` });
						resolvePhase({ score: 0, logPath: null, fallback: true });
					});
				});
			}

			// ── Smart replay: Zig DP replay (fast) with Python fallback ──
			async function smartReplay(difficulty, iter) {
				// Try Zig DP replay first (needs dp_plan.json)
				const dpPlan = resolve(GPU_DIR, 'solutions', difficulty, 'dp_plan.json');
				if (existsSync(dpPlan)) {
					const result = await runZigDpReplay(difficulty, iter);
					if (!result.fallback) return result;
				}
				// Fallback to Python replay
				return await runReplay(difficulty, iter);
			}

			// ── Import game log to PostgreSQL (best-effort, fire-and-forget)
			function importLogToDb(logPath, runType, iter) {
				if (!logPath) return;
				const importScript = resolve(BOT_DIR, 'replay', 'import_logs.py');
				try {
					spawn('python', [importScript, logPath, '--run-type', runType], { stdio: 'ignore' })
						.on('error', () => {});
					sendEvent('log', { text: `[db] Importing as ${runType}`, _iter: iter });
				} catch (e) { /* best-effort */ }
			}

			// ── Main pipeline loop ───────────────────────────────────────
			(async () => {
				sendEvent('pipeline_start', {
					time_budget: timeBudget,
					post_optimize_time: postOptimizeTime,
					gpu_optimize_time: gpuOptimizeTime,
				});

				// Keep existing solutions for replay-first strategy.
				// Solutions persist across keys (same day = same seed).
				const solDir = resolve(GPU_DIR, "solutions");

				// Orders are stored in PostgreSQL (date-keyed), no flat file seeding needed.

				let bestScore = 0;
				let difficulty = parseDifficultyFromUrl(url);
				let iterCount = 0;
				let orderCount = getOrderCount(difficulty);
				let staleOrderIters = 0;

				sendEvent('log', {
					text: `[pipeline] Difficulty: ${difficulty || 'unknown'}, known orders: ${orderCount}, solution: ${solutionExists(difficulty) ? 'yes' : 'no'}`,
				});

				// ── ITERATION 0: Initial discovery ──────────────────────
				// Strategy: replay existing solution if available (fastest
				// way to discover orders), else optimize from known orders,
				// else fall back to Zig bot.
				if (solutionExists(difficulty)) {
					// Replay existing solution via Zig — discovers orders from game
					sendEvent('iter_start', {
						iter: 0, phase: 'replay_discover',
						elapsed: elapsedSecs(), remaining: remaining(),
					});

					// Export dp_plan.json for Zig replay
					await exportDpPlan(difficulty);

					const replayResult = await smartReplay(difficulty, 0);
					if (replayResult.score > bestScore) bestScore = replayResult.score;
					if (replayResult.logPath) importLogToDb(replayResult.logPath, 'replay', 0);

					const captured = await captureOrders(difficulty, 0);
					const newCount = getOrderCount(difficulty);

					sendEvent('orders_update', {
						count: newCount, new_orders: newCount - orderCount, _iter: 0,
					});
					orderCount = newCount;

					sendEvent('iter_done', {
						iter: 0, phase: 'replay_discover',
						score: replayResult.score, game_score: replayResult.score,
						difficulty, captured_orders: captured, orders_total: orderCount,
						elapsed: elapsedSecs(), remaining: remaining(),
					});
					sendEvent('iter_summary', {
						iter: 0, score: replayResult.score, best_score: bestScore,
						iterations_done: 1, remaining: remaining(), orders: orderCount,
					});
					iterCount = 1;

				} else if (orderCount > 0) {
					// Have orders but no solution — optimize first, then replay
					sendEvent('iter_start', {
						iter: 0, phase: 'initial_optimize',
						elapsed: elapsedSecs(), remaining: remaining(),
					});

					const p = GPU_PARAMS[difficulty] || GPU_PARAMS.easy;
					const optResult = await runGpuOptimize(difficulty, 0, p.coldTime, {
						orderings: p.coldOrd, refineIters: p.coldRef,
						maxStates: p.maxStates, speedBonus: 50,
					});
					if (optResult.score > bestScore) bestScore = optResult.score;

					let replayScore = 0;
					if (!ctx.closed && remaining() > 6) {
						await exportDpPlan(difficulty);
						const replayResult = await smartReplay(difficulty, 0);
						replayScore = replayResult.score;
						if (replayScore > bestScore) bestScore = replayScore;
						if (replayResult.logPath) importLogToDb(replayResult.logPath, 'replay', 0);
						await captureOrders(difficulty, 0);
						const newCount = getOrderCount(difficulty);
						sendEvent('orders_update', {
							count: newCount, new_orders: newCount - orderCount, _iter: 0,
						});
						orderCount = newCount;
					}

					sendEvent('iter_done', {
						iter: 0, phase: 'initial_optimize',
						score: Math.max(optResult.score, replayScore),
						opt_score: optResult.score, game_score: replayScore,
						difficulty, orders_total: orderCount,
						elapsed: elapsedSecs(), remaining: remaining(),
					});
					sendEvent('iter_summary', {
						iter: 0, score: bestScore, best_score: bestScore,
						iterations_done: 1, remaining: remaining(), orders: orderCount,
					});
					iterCount = 1;

				} else {
					// No orders, no solution — fall back to Zig bot
					const result = await runZigBot(0);
					difficulty = result.difficulty || difficulty;
					if (result.score > bestScore) bestScore = result.score;
					if (result.logPath) importLogToDb(result.logPath, 'zig', 0);

					const captured = await captureOrders(difficulty, 0);
					orderCount = getOrderCount(difficulty);

					sendEvent('iter_done', {
						iter: 0, phase: 'zig_bot',
						score: result.score, game_score: result.score,
						difficulty, captured_orders: captured, orders_total: orderCount,
						elapsed: elapsedSecs(), remaining: remaining(),
					});
					sendEvent('iter_summary', {
						iter: 0, score: result.score, best_score: bestScore,
						iterations_done: 1, remaining: remaining(), orders: orderCount,
					});
					iterCount = 1;
				}

				// ── ITERATIONS 1+: Optimize -> Replay -> Capture ────────
				// Each cycle discovers orders. When discovery stalls for 2+
				// iterations, switch to deep mode (more states/orderings).
				const params = GPU_PARAMS[difficulty] || GPU_PARAMS.easy;
				const minIterBudget = difficulty === 'expert' ? 30 : difficulty === 'hard' ? 30 : 20;
				let lastImprovementIter = 0;
				let replayFailed = false;

				while (!ctx.closed && remaining() > minIterBudget && difficulty) {
					const i = iterCount;
					const timeLeft = remaining();

					if (timeLeft < minIterBudget) {
						sendEvent('iter_skip', { iter: i, reason: 'insufficient_time', remaining: timeLeft });
						break;
					}

					// Plateau detection
					if (i - lastImprovementIter > params.plateau) {
						sendEvent('iter_skip', {
							iter: i, reason: 'score_plateau',
							remaining: timeLeft, best_score: bestScore,
							stale_iters: i - lastImprovementIter,
						});
						break;
					}

					const prevOrderCount = orderCount;
					const isDeep = staleOrderIters >= 2;
					const isCold = (i === 1);

					if (isDeep && staleOrderIters === 2) {
						sendEvent('mode_change', {
							mode: 'deep', reason: 'no_new_orders',
							stale_iters: staleOrderIters, _iter: i,
						});
					}

					sendEvent('iter_start', {
						iter: i, phase: isDeep ? 'deep_training' : 'optimize_replay',
						elapsed: elapsedSecs(), remaining: timeLeft,
					});

					// GPU optimize with adaptive params
					const baseTime = isCold ? params.coldTime : (isDeep ? Math.min(60, timeLeft - 10) : params.warmTime);
					const gpuTime = Math.min(baseTime, Math.max(10, timeLeft - 8));
					const speedBonus = 50 * Math.pow(0.7, i - 1);

					const optResult = await runGpuOptimize(difficulty, i, gpuTime, {
						warmOnly: !isCold,
						orderings: isDeep ? DEEP_PARAMS.orderings : (isCold ? params.coldOrd : 1),
						refineIters: isDeep ? DEEP_PARAMS.refineIters : (isCold ? params.coldRef : params.warmRef),
						maxStates: isDeep ? DEEP_PARAMS.maxStates : params.maxStates,
						speedBonus,
					});

					if (ctx.closed || remaining() < 6) {
						if (optResult.score > bestScore) bestScore = optResult.score;
						sendEvent('iter_done', {
							iter: i, phase: 'optimize_only',
							score: optResult.score, game_score: 0,
							difficulty, orders_total: orderCount,
							elapsed: elapsedSecs(), remaining: remaining(),
						});
						iterCount++;
						break;
					}

					// Replay + capture
					let replayResult = { score: 0, logPath: null };
					let captured = 0;
					if (!replayFailed) {
						await exportDpPlan(difficulty);
						replayResult = await smartReplay(difficulty, i);
						if (replayResult.logPath) importLogToDb(replayResult.logPath, 'replay', i);
						if (ctx.closed) break;

						if (optResult.score > 10 && replayResult.score < optResult.score * 0.3) {
							replayFailed = true;
							sendEvent('log', {
								text: `[pipeline] Replay failed (${replayResult.score} vs expected ${optResult.score}), skipping future replays`,
								_iter: i,
							});
						}

						captured = await captureOrders(difficulty, i);
					} else {
						sendEvent('log', { text: `[pipeline] Skipping replay (previous failed)`, _iter: i });
					}

					// Track order discovery
					const newCount = getOrderCount(difficulty);
					if (newCount > prevOrderCount) {
						staleOrderIters = 0;
						sendEvent('orders_update', {
							count: newCount, new_orders: newCount - prevOrderCount, _iter: i,
						});
					} else {
						staleOrderIters++;
					}
					orderCount = newCount;

					const iterScore = Math.max(optResult.score, replayResult.score);
					if (iterScore > bestScore) {
						bestScore = iterScore;
						lastImprovementIter = i;
					}

					sendEvent('iter_done', {
						iter: i, phase: isDeep ? 'deep_training' : 'optimize_replay',
						score: iterScore, game_score: replayResult.score,
						opt_score: optResult.score,
						difficulty, captured_orders: captured,
						orders_total: orderCount, is_deep: isDeep,
						replay_skipped: replayFailed && replayResult.score === 0,
						elapsed: elapsedSecs(), remaining: remaining(),
					});
					sendEvent('iter_summary', {
						iter: i, score: iterScore, best_score: bestScore,
						iterations_done: i + 1, remaining: remaining(),
						orders: orderCount,
					});

					iterCount++;
				}

				// Final replay if time remains
				if (!ctx.closed && remaining() > 5 && difficulty && !replayFailed) {
					sendEvent('log', { text: `[pipeline] Final replay with ${remaining().toFixed(0)}s remaining` });
					await exportDpPlan(difficulty);
					const finalReplay = await smartReplay(difficulty, iterCount);
					if (finalReplay.score > bestScore) bestScore = finalReplay.score;
					await captureOrders(difficulty, iterCount);
					orderCount = getOrderCount(difficulty);
				}

				sendEvent('pipeline_complete', {
					best_score: bestScore,
					iterations: iterCount,
					total_elapsed: elapsedSecs(),
					difficulty,
					orders: orderCount,
				});

				// Signal frontend to request new key for continuation
				sendEvent('need_new_key', {
					best_score: bestScore,
					orders: orderCount,
					difficulty,
				});

				cleanup();
			})();
		},
		cancel() {
			createCleanup(ctx, null)();
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
