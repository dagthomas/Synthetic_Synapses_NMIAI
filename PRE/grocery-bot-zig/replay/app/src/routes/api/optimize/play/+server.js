import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

export async function POST({ request }) {
	const { url, difficulty } = await request.json();

	if (!url) {
		return new Response(JSON.stringify({ error: 'Missing WebSocket URL' }), { status: 400 });
	}

	const diff = difficulty || 'easy';
	const encoder = new TextEncoder();
	let botProcess = null;
	let closed = false;
	let safetyTimeout = null;
	let heartbeatInterval = null;

	function cleanupShared(controller) {
		if (closed) return;
		closed = true;
		if (safetyTimeout) { clearTimeout(safetyTimeout); safetyTimeout = null; }
		if (heartbeatInterval) { clearInterval(heartbeatInterval); heartbeatInterval = null; }
		if (botProcess && !botProcess.killed) {
			try { botProcess.kill(); } catch (e) {}
		}
		try { controller?.close(); } catch (e) {}
	}

	const stream = new ReadableStream({
		start(controller) {
			const MAX_RUNTIME = 900000; // 15 min: Zig capture (120s) + GPU solve (up to 600s)

			function cleanup() { cleanupShared(controller); }

			safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: pipeline took too long (15 min limit)' });
				cleanup();
			}, MAX_RUNTIME);

			function sendEvent(type, data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
				} catch (e) {}
			}

			// SSE heartbeat: send a comment every 15s to keep the connection alive
			// through proxies and browser idle timers
			heartbeatInterval = setInterval(() => {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(': ping\n\n'));
				} catch (e) {}
			}, 15000);

			sendEvent('status', { message: `Zig capture → GPU solve pipeline (${diff})` });

			botProcess = spawn('python', ['-u', 'capture_and_solve_stream.py', url, diff, '--capture', 'zig', '--time', '300'], {
				cwd: GPU_DIR,
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let stdoutBuffer = '';

			botProcess.stdout.on('data', (data) => {
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
					} catch (e) {}
				}
			});

			botProcess.stderr.on('data', (data) => {
				const text = data.toString().trim();
				if (text) {
					for (const line of text.split('\n')) {
						const l = line.trim();
						if (!l || l.includes('Connecting to') || l.includes('Logging to')) continue;
						sendEvent('log', { text: l });
					}
				}
			});

			botProcess.on('close', (code) => {
				cleanup();
			});

			botProcess.on('error', (err) => {
				sendEvent('error', { message: `Failed to start: ${err.message}` });
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
