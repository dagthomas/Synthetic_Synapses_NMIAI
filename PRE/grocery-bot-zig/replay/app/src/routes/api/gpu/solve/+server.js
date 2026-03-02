import { spawn } from 'child_process';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const BOT_DIR = resolve(__dirname, '..', '..', '..', '..', '..', '..', '..');
const GPU_DIR = resolve(BOT_DIR, '..', 'grocery-bot-gpu');

export async function POST({ request }) {
	const { difficulty, seed } = await request.json();

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

			safetyTimeout = setTimeout(() => {
				sendEvent({ type: 'error', msg: 'Timeout: GPU solve took too long' });
				cleanup();
			}, 300_000); // 5 min for multi-bot sequential DP

			function sendEvent(data) {
				if (closed) return;
				try {
					controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
				} catch (e) {}
			}

			const args = ['gpu_multi_solve_stream.py', difficulty];
			if (seed) args.push('--seed', String(seed));

			process = spawn('python', args, {
				cwd: GPU_DIR,
				stdio: ['pipe', 'pipe', 'pipe'],
			});

			let stdoutBuffer = '';
			process.stdout.on('data', (data) => {
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

			process.stderr.on('data', (data) => {
				const text = data.toString().trim();
				if (text) sendEvent({ type: 'stderr', text });
			});

			process.on('close', (code) => {
				sendEvent({ type: 'process_done', code });
				cleanup();
			});

			process.on('error', (err) => {
				sendEvent({ type: 'error', msg: `Failed to spawn: ${err.message}` });
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
