import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createCleanup } from '$lib/sse.server.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

export async function POST({ request }) {
	const { difficulty, maxStates, refineIters } = await request.json();

	if (!difficulty || !['easy', 'medium', 'hard', 'expert'].includes(difficulty)) {
		return new Response(JSON.stringify({ error: 'Invalid difficulty' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };

	const stream = new ReadableStream({
		start(controller) {
			const cleanup = createCleanup(ctx, controller);

			ctx.safetyTimeout = setTimeout(() => {
				sendEvent({ type: 'error', msg: 'Timeout: Training took too long' });
				cleanup();
			}, 7200_000); // 2 hours for long training runs

			ctx.heartbeatInterval = setInterval(() => {
				if (ctx.closed) return;
				try { controller.enqueue(encoder.encode(': ping\n\n')); } catch (e) { /* stream closed */ }
			}, 10000);

			function sendEvent(data) {
				if (ctx.closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
				} catch (e) { /* stream closed by client */ }
			}

			// For expert: use solve_expert_3060.py (cracked seed, 3060-optimized)
			// For others: use gpu_multi_solve_stream.py (standard solver)
			let args;
			if (difficulty === 'expert') {
				args = ['solve_expert_3060.py'];
				if (maxStates) args.push('--max-states', String(maxStates));
				if (refineIters) args.push('--refine-iters', String(refineIters));
			} else {
				args = ['gpu_multi_solve_stream.py', difficulty];
			}

			sendEvent({ type: 'training_start', difficulty, script: args[0] });

			ctx.process = spawn('python', args, {
				cwd: GPU_DIR,
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let stdoutBuffer = '';
			ctx.process.stdout.on('data', (data) => {
				stdoutBuffer += data.toString();
				const lines = stdoutBuffer.split('\n');
				stdoutBuffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.trim()) continue;
					try {
						const parsed = JSON.parse(line);
						sendEvent(parsed);
					} catch (e) {
						sendEvent({ type: 'log', text: line.trim() });
					}
				}
			});

			ctx.process.stderr.on('data', (data) => {
				const text = data.toString();
				for (const line of text.split('\n')) {
					if (!line.trim()) continue;
					sendEvent({ type: 'stderr', text: line.trim() });
				}
			});

			ctx.process.on('close', (code) => {
				sendEvent({ type: 'process_done', code });
				cleanup();
			});

			ctx.process.on('error', (err) => {
				sendEvent({ type: 'error', msg: `Failed to spawn: ${err.message}` });
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
