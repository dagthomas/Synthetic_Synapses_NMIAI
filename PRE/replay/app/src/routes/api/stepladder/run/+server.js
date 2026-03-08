import { spawn, spawnSync } from 'child_process';
import { resolve } from 'path';
import { existsSync, readFileSync, copyFileSync, readdirSync, statSync } from 'fs';
import { createCleanup, createSendEvent } from '$lib/sse.server.js';
import { ZIG_BOT_DIR, GPU_DIR, B200_DIR, PYTHON } from '$lib/paths.server.js';

/**
 * Stepladder single-iteration endpoint.
 *
 * Each call runs ONE iteration of the stepladder loop:
 *   1. Replay or Zig capture (uses token)
 *   2. GPU optimize (offline, no token needed)
 *   3. Deep training (offline, configurable budget)
 *
 * The frontend drives the loop by providing a fresh token for each iteration.
 */
export async function POST({ request }) {
	const {
		url,                          // WSS url with token (or 'offline')
		difficulty,                   // easy|medium|hard|expert|nightmare
		iteration = 0,                // current iteration number
		phase = 'auto',               // auto|capture|replay|optimize|deep
		captureBot = 'auto',          // auto|zig|nightmare|python
		gpu = 'auto',                 // auto|b200|5090
		deepBudget = 300,             // seconds for deep training
		maxStates = null,             // override max states
		offlineCapture = null,        // pre-loaded capture data (offline mode)
	} = await request.json();

	if ((!url && !offlineCapture) || !difficulty) {
		return new Response(JSON.stringify({ error: 'Missing url/capture or difficulty' }), { status: 400 });
	}

	const isOffline = url === 'offline' && offlineCapture;

	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };

	const stream = new ReadableStream({
		start(controller) {
			const t0 = Date.now();
			const MAX_RUNTIME = (deepBudget + 600) * 1000;

			const cleanup = createCleanup(ctx, controller);
			const sendEvent = createSendEvent(ctx, controller, encoder);

			ctx.safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Safety timeout reached' });
				cleanup();
			}, MAX_RUNTIME);

			ctx.heartbeatInterval = setInterval(() => {
				if (ctx.closed) return;
				try { controller.enqueue(encoder.encode(': ping\n\n')); } catch (e) {}
			}, 10000);

			function elapsed() { return (Date.now() - t0) / 1000; }

			// ── Helpers ────────────────────────────────────────────────

			function getOrderCount() {
				try {
					const r = spawnSync(PYTHON, ['-u', 'db_query.py', 'order_count', difficulty], { cwd: GPU_DIR, timeout: 5000 });
					return JSON.parse(r.stdout.toString()).count || 0;
				} catch { return 0; }
			}

			function solutionExists() {
				try {
					const r = spawnSync(PYTHON, ['-u', 'db_query.py', 'solution_exists', difficulty], { cwd: GPU_DIR, timeout: 5000 });
					return JSON.parse(r.stdout.toString()).exists || false;
				} catch { return false; }
			}

			function getSolutionScore() {
				try {
					const r = spawnSync(PYTHON, ['-u', 'db_query.py', 'solution_score', difficulty], { cwd: GPU_DIR, timeout: 5000 });
					return JSON.parse(r.stdout.toString()).score || 0;
				} catch { return 0; }
			}

			function findNewestLog(dir, afterMs = 0) {
				try {
					const files = readdirSync(dir)
						.filter(f => f.startsWith('game_log_') && f.endsWith('.jsonl'))
						.map(f => ({ name: f, mtime: statSync(resolve(dir, f)).mtimeMs }))
						.filter(f => f.mtime >= afterMs)
						.sort((a, b) => b.mtime - a.mtime);
					return files.length > 0 ? resolve(dir, files[0].name) : null;
				} catch { return null; }
			}

			// ── Phase: Capture ────────────────────────────────────────
			function runCapture() {
				return new Promise((res) => {
					sendEvent('phase_start', { phase: 'capture', elapsed: elapsed() });

					// Determine which bot to use
					const zigExe = resolve(ZIG_BOT_DIR, 'zig-out', 'bin', 'grocery-bot.exe');
					let bot = captureBot;
					if (bot === 'auto') {
						bot = (difficulty === 'nightmare') ? 'nightmare' : 'zig';
					}
					// Zig doesn't support nightmare
					if (bot === 'zig' && (difficulty === 'nightmare' || !existsSync(zigExe))) {
						bot = 'nightmare';
					}

					if (bot === 'zig') {
						sendEvent('log', { text: '[capture] Using Zig bot' });
						const spawnMs = Date.now();
						ctx.process = spawn(zigExe, [url], {
							cwd: ZIG_BOT_DIR, stdio: ['pipe', 'pipe', 'pipe'],
						});

						let score = 0;
						ctx.process.stderr.on('data', (data) => {
							for (const line of data.toString().split('\n')) {
								if (!line.trim()) continue;
								const rm = line.match(/R(\d+)\/(\d+)\s+Score:(\d+)/);
								if (rm) {
									score = parseInt(rm[3]);
									if (parseInt(rm[1]) % 20 === 0 || rm[1] === rm[2]) {
										sendEvent('round', { round: parseInt(rm[1]), max: parseInt(rm[2]), score });
									}
								}
								const gm = line.match(/GAME_OVER\s+Score:(\d+)/);
								if (gm) score = parseInt(gm[1]);
								sendEvent('log', { text: `[zig] ${line.trim()}` });
							}
						});
						ctx.process.stdout.on('data', () => {});

						ctx.process.on('close', (code) => {
							const logPath = findNewestLog(ZIG_BOT_DIR, spawnMs);
							if (logPath) {
								const dest = resolve(GPU_DIR, logPath.split(/[\\/]/).pop());
								try { copyFileSync(logPath, dest); } catch {}
							}
							sendEvent('phase_done', { phase: 'capture', score, elapsed: elapsed() });
							res({ score, logPath });
						});
						ctx.process.on('error', (err) => {
							sendEvent('error', { message: `Zig capture failed: ${err.message}` });
							res({ score: 0 });
						});
					} else if (bot === 'nightmare') {
						// NightmareBot — PIBT+Hungarian, outputs to stderr
						sendEvent('log', { text: `[capture] Using NightmareBot (${difficulty})` });
						const spawnMs = Date.now();
						const args = ['-u', 'nightmare_bot.py', url, '-v'];
						ctx.process = spawn(PYTHON, args, {
							cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
						});

						let score = 0;
						ctx.process.stderr.on('data', (data) => {
							for (const line of data.toString().split('\n')) {
								if (!line.trim()) continue;
								const rm = line.match(/R(\d+)\/(\d+)\s+Score:(\d+)/);
								if (rm) {
									score = parseInt(rm[3]);
									if (parseInt(rm[1]) % 20 === 0 || rm[1] === rm[2]) {
										sendEvent('round', { round: parseInt(rm[1]), max: parseInt(rm[2]), score });
									}
								}
								const gm = line.match(/GAME_OVER\s+Score:(\d+)/);
								if (gm) score = parseInt(gm[1]);
								sendEvent('log', { text: `[nightmare] ${line.trim()}` });
							}
						});
						ctx.process.stdout.on('data', () => {});

						ctx.process.on('close', () => {
							const logPath = findNewestLog(GPU_DIR, spawnMs);
							sendEvent('phase_done', { phase: 'capture', score, elapsed: elapsed() });
							res({ score, logPath });
						});
						ctx.process.on('error', (err) => {
							sendEvent('error', { message: `NightmareBot capture failed: ${err.message}` });
							res({ score: 0 });
						});
					} else {
						// Python fallback — live_gpu_stream.py --cpu
						sendEvent('log', { text: `[capture] Using live_gpu_stream.py --cpu (${difficulty})` });
						const spawnMs = Date.now();
						const args = [
							'-u', 'live_gpu_stream.py', url,
							'--cpu', '--save', '--json-stream', '--record',
							'--pipeline-mode', '--preload-capture',
						];
						ctx.process = spawn(PYTHON, args, {
							cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
						});

						let buf = '';
						let score = 0;
						ctx.process.stdout.on('data', (data) => {
							buf += data.toString();
							const lines = buf.split('\n');
							buf = lines.pop() || '';
							for (const line of lines) {
								if (!line.trim()) continue;
								try {
									const evt = JSON.parse(line);
									const type = evt.type; delete evt.type;
									if (type === 'game_over') score = evt.score || 0;
									if (type === 'pipeline_done') score = evt.final_score || evt.plan_score || score;
									if (type === 'round') {
										if (evt.round % 20 === 0) {
											sendEvent('round', { round: evt.round, score: evt.score || 0 });
										}
									}
									sendEvent(type, evt);
								} catch {}
							}
						});
						ctx.process.stderr.on('data', (data) => {
							for (const line of data.toString().split('\n')) {
								if (line.trim()) sendEvent('log', { text: `[live] ${line.trim()}` });
							}
						});
						ctx.process.on('close', () => {
							sendEvent('phase_done', { phase: 'capture', score, elapsed: elapsed() });
							res({ score });
						});
						ctx.process.on('error', (err) => {
							sendEvent('error', { message: `Python capture failed: ${err.message}` });
							res({ score: 0 });
						});
					}
				});
			}

			// ── Phase: Replay existing solution ────────────────────────
			function runReplay() {
				return new Promise((res) => {
					sendEvent('phase_start', { phase: 'replay', elapsed: elapsed() });

					// Export dp_plan for Zig replay
					spawnSync(PYTHON, ['-u', 'db_query.py', 'export_dp_plan', difficulty,
						resolve(GPU_DIR, '.tmp', `dp_plan_${difficulty}.json`)], { cwd: GPU_DIR, timeout: 10000 });

					const exe = resolve(ZIG_BOT_DIR, 'zig-out', 'bin', 'grocery-bot.exe');
					const dpPlan = resolve(GPU_DIR, '.tmp', `dp_plan_${difficulty}.json`);
					const useDp = existsSync(exe) && existsSync(dpPlan) && difficulty !== 'nightmare';

					if (useDp) {
						const spawnMs = Date.now();
						const args = [url, '--dp-plan', dpPlan];
						ctx.process = spawn(exe, args, { cwd: ZIG_BOT_DIR, stdio: ['pipe', 'pipe', 'pipe'] });

						let score = 0;
						ctx.process.stderr.on('data', (data) => {
							for (const line of data.toString().split('\n')) {
								if (!line.trim()) continue;
								const rm = line.match(/R(\d+)\/(\d+)\s+Score:(\d+)/);
								if (rm) {
									score = parseInt(rm[3]);
									if (parseInt(rm[1]) % 20 === 0 || rm[1] === rm[2]) {
										sendEvent('round', { round: parseInt(rm[1]), max: parseInt(rm[2]), score });
									}
								}
								const gm = line.match(/GAME_OVER\s+Score:(\d+)/);
								if (gm) score = parseInt(gm[1]);
								sendEvent('log', { text: `[zig-replay] ${line.trim()}` });
							}
						});
						ctx.process.stdout.on('data', () => {});

						ctx.process.on('close', () => {
							const logPath = findNewestLog(ZIG_BOT_DIR, spawnMs);
							if (logPath) {
								const dest = resolve(GPU_DIR, logPath.split(/[\\/]/).pop());
								try { copyFileSync(logPath, dest); } catch {}
							}
							sendEvent('phase_done', { phase: 'replay', score, elapsed: elapsed() });
							res({ score, logPath });
						});
						ctx.process.on('error', () => res({ score: 0 }));
					} else {
						// Python replay fallback
						ctx.process = spawn(PYTHON, [
							'replay_solution.py', url, '--difficulty', difficulty
						], { cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'] });

						let score = 0;
						ctx.process.stderr.on('data', (data) => {
							for (const line of data.toString().split('\n')) {
								if (!line.trim()) continue;
								const gm = line.match(/GAME_OVER Score:(\d+)/);
								if (gm) score = parseInt(gm[1]);
								sendEvent('log', { text: `[replay] ${line.trim()}` });
							}
						});
						ctx.process.stdout.on('data', () => {});
						ctx.process.on('close', () => {
							sendEvent('phase_done', { phase: 'replay', score, elapsed: elapsed() });
							res({ score });
						});
						ctx.process.on('error', () => res({ score: 0 }));
					}
				});
			}

			// ── Phase: Capture orders from game log ────────────────────
			function captureOrders() {
				return new Promise((res) => {
					const logPath = findNewestLog(GPU_DIR, t0);
					if (!logPath) { res(0); return; }
					sendEvent('phase_start', { phase: 'capture_orders', elapsed: elapsed() });

					const proc = spawn(PYTHON, ['capture_from_game_log.py', logPath, difficulty], {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});
					let captured = 0;
					proc.stderr.on('data', (d) => {
						const t = d.toString().trim();
						if (t) {
							sendEvent('log', { text: `[capture] ${t}` });
							const m = t.match(/(\d+) orders/);
							if (m) captured = parseInt(m[1]);
						}
					});
					proc.on('close', () => {
						sendEvent('phase_done', { phase: 'capture_orders', captured, elapsed: elapsed() });
						res(captured);
					});
					proc.on('error', () => res(0));
				});
			}

			// ── Phase: GPU optimize ────────────────────────────────────
			function runGpuOptimize(maxTime) {
				return new Promise((res) => {
					sendEvent('phase_start', { phase: 'optimize', elapsed: elapsed(), budget: maxTime });

					const args = ['-u', 'optimize_and_save.py', difficulty, '--max-time', String(Math.floor(maxTime))];
					if (maxStates) args.push('--max-states', String(maxStates));

					ctx.process = spawn(PYTHON, args, {
						cwd: GPU_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let buf = '';
					let score = 0;
					ctx.process.stdout.on('data', (data) => {
						buf += data.toString();
						const lines = buf.split('\n');
						buf = lines.pop() || '';
						for (const line of lines) {
							if (!line.trim()) continue;
							try {
								const evt = JSON.parse(line);
								if (evt.type === 'optimize_done') score = evt.score || 0;
								sendEvent(evt.type, evt);
							} catch {}
						}
					});
					ctx.process.stderr.on('data', (data) => {
						for (const line of data.toString().trim().split('\n')) {
							if (line.trim()) sendEvent('log', { text: `[gpu] ${line.trim()}` });
						}
					});
					ctx.process.on('close', () => {
						sendEvent('phase_done', { phase: 'optimize', score, elapsed: elapsed() });
						res({ score });
					});
					ctx.process.on('error', (err) => {
						sendEvent('error', { message: `GPU optimize failed: ${err.message}` });
						res({ score: 0 });
					});
				});
			}

			// ── Phase: Deep training ───────────────────────────────────
			function runDeepTraining(budget) {
				return new Promise((res) => {
					sendEvent('phase_start', { phase: 'deep_train', elapsed: elapsed(), budget });

					const args = ['-u', 'deep_optimize.py', difficulty, '--budget', String(Math.floor(budget))];
					if (maxStates) args.push('--max-states', String(maxStates));
					args.push('--gpu', gpu);

					ctx.process = spawn(PYTHON, args, {
						cwd: B200_DIR, stdio: ['pipe', 'pipe', 'pipe'],
					});

					let score = 0;
					ctx.process.stderr.on('data', (data) => {
						for (const line of data.toString().split('\n')) {
							if (!line.trim()) continue;
							sendEvent('log', { text: `[deep] ${line.trim()}` });
							const sm = line.match(/NEW BEST (\d+)/);
							if (sm) {
								score = parseInt(sm[1]);
								sendEvent('deep_improvement', { score, elapsed: elapsed() });
							}
							const fm = line.match(/Final score: (\d+)/);
							if (fm) score = parseInt(fm[1]);
							// Phase progress
							const pm = line.match(/Phase (\d): (\w+) \((\d+)s\)/);
							if (pm) {
								sendEvent('deep_phase', { phase: parseInt(pm[1]), name: pm[2], budget: parseInt(pm[3]) });
							}
						}
					});
					ctx.process.stdout.on('data', () => {});

					ctx.process.on('close', () => {
						sendEvent('phase_done', { phase: 'deep_train', score, elapsed: elapsed() });
						res({ score });
					});
					ctx.process.on('error', (err) => {
						sendEvent('error', { message: `Deep training failed: ${err.message}` });
						res({ score: 0 });
					});
				});
			}

			// ── Load capture data from DB (for remote GPU) ────────────
			function loadCaptureData() {
				try {
					const r = spawnSync(PYTHON,
						['-u', 'db_query.py', 'export_capture_json', difficulty],
						{ cwd: GPU_DIR, timeout: 10000 });
					return JSON.parse(r.stdout.toString());
				} catch { return null; }
			}

			// ── Main iteration ─────────────────────────────────────────
			(async () => {
				if (isOffline) {
					// ── Offline mode: save capture, then optimize ────
					const offDiff = offlineCapture.difficulty || difficulty;
					sendEvent('iter_start', {
						iteration, difficulty: offDiff, gpu,
						orders: offlineCapture.orders?.length || 0,
						has_solution: false, prev_score: 0,
						deep_budget: deepBudget, offline: true,
					});

					// Save capture to file for optimize_and_save.py
					const { writeFileSync, mkdirSync } = await import('fs');
					const capturePath = resolve(GPU_DIR, 'captures', `offline_${offDiff}_temp.json`);
					try { mkdirSync(resolve(GPU_DIR, 'captures'), { recursive: true }); } catch {}
					writeFileSync(capturePath, JSON.stringify(offlineCapture));
					sendEvent('log', { text: `Saved capture to ${capturePath}` });

					// Save capture to DB via solution_store
					const saveResult = spawnSync(PYTHON, [
						'-u', '-c',
						`import json, sys; sys.path.insert(0, '.'); from solution_store import save_capture; d=json.load(open(r'${capturePath.replace(/\\/g, '/')}')); save_capture('${offDiff}', d); print('OK')`
					], { cwd: GPU_DIR, timeout: 10000 });
					if (saveResult.stdout?.toString().includes('OK')) {
						sendEvent('log', { text: 'Capture saved to DB' });
					} else {
						sendEvent('log', { text: `DB save: ${saveResult.stderr?.toString().trim() || 'unknown'}` });
					}

					if (ctx.closed) { cleanup(); return; }

					// Run GPU optimize
					let optScore = 0, deepScore = 0;
					const optResult = await runGpuOptimize(Math.min(120, deepBudget));
					optScore = optResult.score;

					if (!ctx.closed && deepBudget > 120) {
						const dr = await runDeepTraining(deepBudget);
						deepScore = dr.score;
					}

					sendEvent('iter_done', {
						iteration, game_score: 0,
						opt_score: optScore, deep_score: deepScore,
						best_score: Math.max(optScore, deepScore),
						orders: offlineCapture.orders?.length || 0,
						new_orders: 0, elapsed: elapsed(),
					});
					cleanup();
					return;
				}

				// ── Live mode ─────────────────────────────────────────
				const ordersBefore = getOrderCount();
				const hasSolution = solutionExists();
				const prevScore = getSolutionScore();

				sendEvent('iter_start', {
					iteration, difficulty, gpu,
					orders: ordersBefore,
					has_solution: hasSolution,
					prev_score: prevScore,
					deep_budget: deepBudget,
				});

				// Step 1: Capture or Replay (uses token)
				let gameScore = 0;
				if (hasSolution && iteration > 0) {
					const r = await runReplay();
					gameScore = r.score;
				} else {
					const r = await runCapture();
					gameScore = r.score;
				}

				if (ctx.closed) { cleanup(); return; }

				// Step 2: Capture orders from game log
				await captureOrders();
				const ordersAfter = getOrderCount();
				const newOrders = ordersAfter - ordersBefore;

				sendEvent('orders_update', {
					before: ordersBefore, after: ordersAfter, new_orders: newOrders,
				});

				// Emit capture data for remote GPU mode
				const captureJson = loadCaptureData();
				if (captureJson) {
					sendEvent('capture_data', { data: captureJson });
				}

				if (ctx.closed) { cleanup(); return; }

				// Step 3+4: Local GPU optimize + deep (skip if deepBudget=0, i.e. remote mode)
				let optScore = 0, deepScore = 0;
				if (deepBudget > 0) {
					const optResult = await runGpuOptimize(Math.min(60, 240 - elapsed()));
					optScore = optResult.score;

					if (!ctx.closed && deepBudget > 30) {
						const dr = await runDeepTraining(deepBudget);
						deepScore = dr.score;
					}
				}

				const bestScore = Math.max(gameScore, optScore, deepScore, prevScore);
				const finalOrders = getOrderCount();

				sendEvent('iter_done', {
					iteration,
					game_score: gameScore,
					opt_score: optScore,
					deep_score: deepScore,
					best_score: bestScore,
					orders: finalOrders,
					new_orders: finalOrders - ordersBefore,
					elapsed: elapsed(),
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
