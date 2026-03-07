import { spawn } from 'child_process';
import { resolve } from 'path';
import { createCleanup } from '$lib/sse.server.js';
import { GPU_DIR } from '$lib/paths.server.js';

export async function POST({ request }) {
	const { difficulty, seed } = await request.json();

	if (!difficulty || !['easy', 'medium', 'hard', 'expert'].includes(difficulty)) {
		return new Response(JSON.stringify({ error: 'Invalid difficulty' }), { status: 400 });
	}

	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };

	const stream = new ReadableStream({
		start(controller) {
			const cleanup = createCleanup(ctx, controller);

			ctx.safetyTimeout = setTimeout(() => {
				sendEvent({ type: 'error', msg: 'Timeout: GPU solve took too long' });
				cleanup();
			}, 3600_000); // 60 min for expert 10-bot sequential DP

			// Custom sendEvent: this endpoint passes pre-formed objects (not type+data)
			function sendEvent(data) {
				if (ctx.closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
				} catch (e) { /* stream closed by client */ }
			}

			const args = ['gpu_multi_solve_stream.py', difficulty];
			if (seed) args.push('--seed', String(seed));

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
				const text = data.toString().trim();
				if (text) sendEvent({ type: 'stderr', text });
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
