import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { readdirSync, statSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

export async function POST({ request }) {
	const { url, postOptimizeTime = 1800 } = await request.json();

	if (!url) {
		return new Response(JSON.stringify({ error: 'Missing WebSocket URL' }), { status: 400 });
	}

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
			// game 120s + post-optimize + buffer
			const MAX_RUNTIME = (postOptimizeTime + 300) * 1000;

			function cleanup() { cleanupShared(controller); }

			safetyTimeout = setTimeout(() => {
				sendEvent('error', { message: 'Timeout: pipeline took too long' });
				cleanup();
			}, MAX_RUNTIME);

			function sendEvent(type, data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type, ...data })}\n\n`));
				} catch (e) {}
			}

			heartbeatInterval = setInterval(() => {
				if (closed) return;
				try { controller.enqueue(encoder.encode(': ping\n\n')); } catch (e) {}
			}, 15000);

			sendEvent('status', { message: `GPU pipeline starting (post-optimize: ${postOptimizeTime}s)` });

			const args = [
				'-u', 'live_gpu_stream.py', url,
				'--save',
				'--json-stream',
				'--post-optimize-time', String(postOptimizeTime),
			];

			botProcess = spawn('python', args, {
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
				if (!text) return;
				for (const line of text.split('\n')) {
					const l = line.trim();
					if (!l) continue;
					sendEvent('log', { text: l });
				}
			});

			botProcess.on('close', (code) => {
				sendEvent('status', { message: `Process exited (code ${code})` });

				// Import game log to PostgreSQL in background (non-blocking)
				// live_gpu_stream.py also does this internally, but catches edge cases
				try {
					const importScript = resolve(GPU_DIR, '..', 'grocery-bot-zig', 'replay', 'import_logs.py');
					const files = readdirSync(GPU_DIR)
						.filter(f => f.startsWith('game_log_') && f.endsWith('.jsonl'))
						.map(f => ({ name: f, mtime: statSync(resolve(GPU_DIR, f)).mtimeMs }))
						.sort((a, b) => b.mtime - a.mtime);
					if (files.length > 0) {
						const logFile = resolve(GPU_DIR, files[0].name);
						const importer = spawn('python', [importScript, logFile, '--run-type', 'live'], {
							stdio: 'ignore',
						});
						importer.on('error', () => {});
						sendEvent('db', { message: `Saving to DB: ${files[0].name}` });
					}
				} catch (e) {}

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
