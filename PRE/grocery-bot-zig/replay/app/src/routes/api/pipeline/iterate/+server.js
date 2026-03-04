import { spawn, spawnSync } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync, statSync, readFileSync, existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, 'grocery-bot-gpu');

export async function POST({ request }) {
	const {
		url,
		timeBudget = 280,       // seconds total (~288s token window, leave margin)
		postOptimizeTime = 30,  // post-opt for initial live game
		gpuOptimizeTime = 45,   // time per offline GPU optimize pass
	} = await request.json();

	if (!url) {
		return new Response(JSON.stringify({ error: 'Missing WebSocket URL' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	let currentProcess = null;
	let closed = false;
	let safetyTimeout = null;
	let heartbeatInterval = null;

	function cleanupShared(controller) {
		if (closed) return;
		closed = true;
		if (safetyTimeout) { clearTimeout(safetyTimeout); safetyTimeout = null; }
		if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }
		if (currentProcess && !currentProcess.killed) {
			try { currentProcess.kill(); } catch (e) {}
		}
		try { controller?.close(); } catch (e) {}
	}

	const stream = new ReadableStream({
		start(controller) {
			const pipelineStart = Date.now();
			const MAX_RUNTIME = (timeBudget + 300) * 1000;

			function cleanup() { cleanupShared(controller); }

			safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Pipeline safety timeout reached' });
				cleanup();
			}, MAX_RUNTIME);

			heartbeatInterval = setInterval(() => {
				if (closed) return;
				try { controller.enqueue(encoder.encode(': ping\n\n')); } catch (e) {}
			}, 10000);

			function sendEvent(type, data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
				} catch (e) {}
			}

			function elapsedSecs() {
				return (Date.now() - pipelineStart) / 1000;
			}

			function remaining() {
				return Math.max(0, timeBudget - elapsedSecs());
			}

			// ── Find latest game log ─────────────────────────────────────
			function findLatestLog(afterMs = 0) {
				try {
					const files = readdirSync(GPU_DIR)
						.filter(f => f.startsWith('game_log_') && f.endsWith('.jsonl'))
						.map(f => ({ name: f, mtime: statSync(resolve(GPU_DIR, f)).mtimeMs }))
						.filter(f => f.mtime >= afterMs)
						.sort((a, b) => b.mtime - a.mtime);
					return files.length > 0 ? resolve(GPU_DIR, files[0].name) : null;
				} catch (e) { return null; }
			}

			// ── Capture orders from game log (synchronous) ───────────────
			function captureOrders(difficulty, iter) {
				return new Promise((res) => {
					const logPath = findLatestLog(pipelineStart);
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

			// ── Phase: Live play (iteration 0) ──────────────────────────
			function runLivePlay(iter, postOpt) {
				return new Promise((resolvePhase) => {
					sendEvent('iter_start', {
						iter, phase: 'live_play',
						elapsed: elapsedSecs(), remaining: remaining(),
						post_optimize_time: postOpt,
					});

					const args = [
						'-u', 'live_gpu_stream.py', url,
						'--save', '--json-stream', '--record',
						'--post-optimize-time', String(postOpt),
					];

					currentProcess = spawn('python', args, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let buf = '';
					let score = 0, gameScore = 0, difficulty = null, orders = 0;

					currentProcess.stdout.on('data', (data) => {
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
							} catch (e) {}
						}
					});

					currentProcess.stderr.on('data', (data) => {
						for (const line of data.toString().trim().split('\n')) {
							if (line.trim()) sendEvent('log', { text: line.trim(), _iter: iter });
						}
					});

					currentProcess.on('close', (code) => {
						resolvePhase({ score, gameScore, difficulty, orders, exitCode: code });
					});
					currentProcess.on('error', (err) => {
						sendEvent('error', { message: `Live play failed: ${err.message}` });
						resolvePhase({ score: 0, gameScore: 0, difficulty: null, orders: 0, exitCode: 1 });
					});
				});
			}

			// ── Phase: Offline GPU optimize ──────────────────────────────
			function runGpuOptimize(difficulty, iter, maxTime) {
				return new Promise((resolvePhase) => {
					sendEvent('optimize_phase_start', {
						_iter: iter, difficulty, max_time: maxTime,
						_elapsed: elapsedSecs(), _remaining: remaining(),
					});

					const args = [
						'-u', 'optimize_and_save.py', difficulty,
						'--max-time', String(maxTime),
					];

					currentProcess = spawn('python', args, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let buf = '';
					let score = 0, prevScore = 0;

					currentProcess.stdout.on('data', (data) => {
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
							} catch (e) {}
						}
					});

					currentProcess.stderr.on('data', (data) => {
						for (const line of data.toString().trim().split('\n')) {
							if (line.trim()) sendEvent('log', { text: `[gpu] ${line.trim()}`, _iter: iter });
						}
					});

					currentProcess.on('close', () => {
						resolvePhase({ score, prevScore });
					});
					currentProcess.on('error', (err) => {
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
					currentProcess = spawn('python', replayArgs, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let score = 0;
					let logPath = null;
					let stderrBuf = '';

					// Poll game log for grid events (like the optimize/replay endpoint)
					let pollLogFile = null;
					let lineCount = 0;
					let initSent = false;
					let lastState = null;

					const pollInterval = setInterval(() => {
						if (!pollLogFile) {
							pollLogFile = findLatestLog(replayStartMs);
							if (!pollLogFile) return;
						}
						try {
							const content = readFileSync(pollLogFile, 'utf-8');
							const lines = content.split('\n').filter(l => l.trim());
							while (lineCount < lines.length) {
								const line = lines[lineCount];
								lineCount++;
								// Even lines = game state, odd lines = bot response
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
						} catch (e) {}
					}, 200);

					currentProcess.stderr.on('data', (data) => {
						stderrBuf += data.toString();
						const lines = stderrBuf.split('\n');
						stderrBuf = lines.pop() || '';
						for (const line of lines) {
							if (!line.trim()) continue;
							sendEvent('log', { text: `[replay] ${line.trim()}`, _iter: iter });
							// Parse score from stderr
							const gameOver = line.match(/GAME_OVER Score:(\d+)/);
							if (gameOver) score = parseInt(gameOver[1]);
							const logMatch = line.match(/Log saved: (.+)/);
							if (logMatch) logPath = logMatch[1].trim();
						}
					});

					currentProcess.stdout.on('data', () => {});

					currentProcess.on('close', (code) => {
						clearInterval(pollInterval);
						// Final poll
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
										} catch (e) {}
									}
								}
							} catch (e) {}
						}
						sendEvent('replay_phase_done', {
							_iter: iter, score, exit_code: code,
							_elapsed: elapsedSecs(), _remaining: remaining(),
						});
						resolvePhase({ score, logPath });
					});
					currentProcess.on('error', (err) => {
						clearInterval(pollInterval);
						sendEvent('error', { message: `Replay failed: ${err.message}` });
						resolvePhase({ score: 0, logPath: null });
					});
				});
			}

			// ── Main pipeline loop ───────────────────────────────────────
			(async () => {
				sendEvent('pipeline_start', {
					time_budget: timeBudget,
					post_optimize_time: postOptimizeTime,
					gpu_optimize_time: gpuOptimizeTime,
				});

				let bestScore = 0;
				let difficulty = null;
				let iterCount = 0;

				// ── ITERATION 0: Live play with GPU ──────────────────────
				{
					const postOpt = Math.min(postOptimizeTime, remaining() * 0.3);
					const result = await runLivePlay(0, Math.floor(postOpt));
					difficulty = result.difficulty;
					if (result.score > bestScore) bestScore = result.score;

					// Capture orders from game log
					const captured = await captureOrders(difficulty, 0);

					sendEvent('iter_done', {
						iter: 0, phase: 'live',
						score: result.score, game_score: result.gameScore,
						difficulty, orders: result.orders,
						captured_orders: captured,
						elapsed: elapsedSecs(), remaining: remaining(),
					});
					sendEvent('iter_summary', {
						iter: 0, score: result.score, best_score: bestScore,
						iterations_done: 1, remaining: remaining(),
					});
					iterCount = 1;
				}

				// ── ITERATIONS 1+: GPU optimize → replay → capture ───────
				while (!closed && remaining() > 45 && difficulty) {
					const i = iterCount;
					const timeLeft = remaining();

					// Need: GPU optimize (~30-60s) + replay (~120s) + capture (~5s)
					// At minimum ~45s for a quick optimize + replay
					if (timeLeft < 45) {
						sendEvent('iter_skip', { iter: i, reason: 'insufficient_time', remaining: timeLeft });
						break;
					}

					sendEvent('iter_start', {
						iter: i, phase: 'optimize_replay',
						elapsed: elapsedSecs(), remaining: timeLeft,
					});

					// GPU optimize time: use configured amount, but leave enough for replay
					// Replay takes ~120s for a full game, minimum ~30s for partial
					const gpuTime = Math.min(gpuOptimizeTime, Math.max(10, timeLeft - 40));

					// Phase A: Offline GPU optimize
					const optResult = await runGpuOptimize(difficulty, i, gpuTime);

					if (closed || remaining() < 30) {
						sendEvent('iter_done', {
							iter: i, phase: 'optimize_only',
							score: optResult.score, game_score: 0,
							difficulty, elapsed: elapsedSecs(), remaining: remaining(),
						});
						if (optResult.score > bestScore) bestScore = optResult.score;
						iterCount++;
						break;
					}

					// Phase B: Replay optimized solution with same URL
					const replayResult = await runReplay(difficulty, i);

					// Phase C: Capture new orders from replay
					const captured = await captureOrders(difficulty, i);

					const iterScore = Math.max(optResult.score, replayResult.score);
					if (iterScore > bestScore) bestScore = iterScore;

					sendEvent('iter_done', {
						iter: i, phase: 'optimize_replay',
						score: iterScore, game_score: replayResult.score,
						opt_score: optResult.score,
						difficulty, captured_orders: captured,
						elapsed: elapsedSecs(), remaining: remaining(),
					});
					sendEvent('iter_summary', {
						iter: i, score: iterScore, best_score: bestScore,
						iterations_done: i + 1, remaining: remaining(),
					});

					iterCount++;
					// Brief pause
					await new Promise(r => setTimeout(r, 1000));
				}

				sendEvent('pipeline_complete', {
					best_score: bestScore,
					iterations: iterCount,
					total_elapsed: elapsedSecs(),
					difficulty,
				});

				cleanup();
			})();
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
