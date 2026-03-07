import { spawn } from 'child_process';
import { resolve } from 'path';
import { createCleanup, createSendEvent } from '$lib/sse.server.js';
import { GPU_DIR } from '$lib/paths.server.js';

export async function POST({ request }) {
	const { difficulty, time = 120, workers = 12 } = await request.json();

	if (!difficulty || !['easy', 'medium', 'hard', 'expert'].includes(difficulty)) {
		return new Response(JSON.stringify({ error: 'Invalid difficulty' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };

	const stream = new ReadableStream({
		start(controller) {
			const cleanup = createCleanup(ctx, controller);
			const sendEvent = createSendEvent(ctx, controller, encoder);

			const MAX_RUNTIME = (time + 30) * 1000;
			ctx.safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: optimization took too long' });
				cleanup();
			}, MAX_RUNTIME);

			sendEvent('status', { message: `Starting optimizer: ${difficulty}, ${time}s, ${workers} workers` });

			ctx.process = spawn('python', [
				'learn_from_capture.py', difficulty,
				'--time', String(time),
				'--workers', String(workers),
			], {
				cwd: GPU_DIR,
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let stderrBuffer = '';
			ctx.process.stderr.on('data', (data) => {
				stderrBuffer += data.toString();
				const lines = stderrBuffer.split('\n');
				stderrBuffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.trim()) continue;
					sendEvent('log', { text: line.trim() });

					// Parse worker results
					const workerMatch = line.match(/W(\d+) mab=(\d+): planner=(\d+) final=(\d+)/);
					if (workerMatch) {
						sendEvent('worker_result', {
							worker: parseInt(workerMatch[1]),
							mab: parseInt(workerMatch[2]),
							planner: parseInt(workerMatch[3]),
							final: parseInt(workerMatch[4]),
						});
					}

					// Parse best result
					const bestMatch = line.match(/BEST: W(\d+) mab=(\d+) score=(\d+)/);
					if (bestMatch) {
						sendEvent('best', {
							worker: parseInt(bestMatch[1]),
							mab: parseInt(bestMatch[2]),
							score: parseInt(bestMatch[3]),
						});
					}

					// Parse improvement
					const improveMatch = line.match(/IMPROVED: (\d+) -> (\d+) \(\+(\d+)\)/);
					if (improveMatch) {
						sendEvent('improved', {
							old_score: parseInt(improveMatch[1]),
							new_score: parseInt(improveMatch[2]),
							delta: parseInt(improveMatch[3]),
						});
					}

					// Parse learn done
					const doneMatch = line.match(/LEARN_DONE score=(\d+) prev=(\d+)/);
					if (doneMatch) {
						sendEvent('learn_done', {
							score: parseInt(doneMatch[1]),
							prev_score: parseInt(doneMatch[2]),
						});
					}
				}
			});

			ctx.process.stdout.on('data', () => {});

			ctx.process.on('close', (code) => {
				sendEvent('done', { code, message: `Optimizer finished (exit ${code})` });
				cleanup();
			});

			ctx.process.on('error', (err) => {
				sendEvent('error', { message: `Failed to spawn optimizer: ${err.message}` });
				cleanup();
			});
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
