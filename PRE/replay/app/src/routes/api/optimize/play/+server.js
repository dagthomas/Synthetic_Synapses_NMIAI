import { spawn } from 'child_process';
import { resolve } from 'path';
import { createCleanup, createSendEvent } from '$lib/sse.server.js';
import { GPU_DIR, PYTHON } from '$lib/paths.server.js';

export async function POST({ request }) {
	const { url, difficulty } = await request.json();

	if (!url) {
		return new Response(JSON.stringify({ error: 'Missing WebSocket URL' }), { status: 400 });
	}

	const diff = difficulty || 'easy';
	const encoder = new TextEncoder();
	const ctx = { closed: false, safetyTimeout: null, heartbeatInterval: null, process: null };

	const stream = new ReadableStream({
		start(controller) {
			const MAX_RUNTIME = 900000; // 15 min: Zig capture (120s) + GPU solve (up to 600s)

			const cleanup = createCleanup(ctx, controller);
			const sendEvent = createSendEvent(ctx, controller, encoder);

			ctx.safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: pipeline took too long (15 min limit)' });
				cleanup();
			}, MAX_RUNTIME);

			// SSE heartbeat: send a comment every 15s to keep the connection alive
			// through proxies and browser idle timers
			ctx.heartbeatInterval = setInterval(() => {
				if (ctx.closed) return;
				try {
					controller.enqueue(encoder.encode(': ping\n\n'));
				} catch (e) { /* stream closed by client */ }
			}, 15000);

			sendEvent('status', { message: `Zig capture → GPU solve pipeline (${diff})` });

			ctx.process = spawn(PYTHON, ['-u', 'capture_and_solve_stream.py', url, diff, '--capture', 'zig', '--time', '300'], {
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
						const evt = JSON.parse(line);
						const type = evt.type;
						delete evt.type;
						sendEvent(type, evt);
					} catch (e) { /* ignore malformed JSON output */ }
				}
			});

			ctx.process.stderr.on('data', (data) => {
				const text = data.toString().trim();
				if (text) {
					for (const line of text.split('\n')) {
						const l = line.trim();
						if (!l || l.includes('Connecting to') || l.includes('Logging to')) continue;
						sendEvent('log', { text: l });
					}
				}
			});

			ctx.process.on('close', (code) => {
				cleanup();
			});

			ctx.process.on('error', (err) => {
				sendEvent('error', { message: `Failed to start: ${err.message}` });
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
