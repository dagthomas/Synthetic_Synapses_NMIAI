import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

export async function POST({ request }) {
	const { difficulty, time = 120, workers = 12 } = await request.json();

	if (!difficulty || !['easy', 'medium', 'hard', 'expert'].includes(difficulty)) {
		return new Response(JSON.stringify({ error: 'Invalid difficulty' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	let process = null;
	let closed = false;
	let safetyTimeout = null;

	function cleanupShared(controller) {
		if (closed) return;
		closed = true;
		if (safetyTimeout) { clearTimeout(safetyTimeout); safetyTimeout = null; }
		if (process && !process.killed) {
			try { process.kill(); } catch (e) {}
		}
		try { controller?.close(); } catch (e) {}
	}

	const stream = new ReadableStream({
		start(controller) {
			function cleanup() { cleanupShared(controller); }

			const MAX_RUNTIME = (time + 30) * 1000;
			safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: optimization took too long' });
				cleanup();
			}, MAX_RUNTIME);

			function sendEvent(type, data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
				} catch (e) {}
			}

			sendEvent('status', { message: `Starting optimizer: ${difficulty}, ${time}s, ${workers} workers` });

			process = spawn('python', [
				'learn_from_capture.py', difficulty,
				'--time', String(time),
				'--workers', String(workers),
			], {
				cwd: GPU_DIR,
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let stderrBuffer = '';
			process.stderr.on('data', (data) => {
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

			process.stdout.on('data', () => {});

			process.on('close', (code) => {
				sendEvent('done', { code, message: `Optimizer finished (exit ${code})` });
				cleanup();
			});

			process.on('error', (err) => {
				sendEvent('error', { message: `Failed to spawn optimizer: ${err.message}` });
				cleanup();
			});
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
