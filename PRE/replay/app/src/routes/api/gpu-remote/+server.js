/**
 * Proxy to remote GPU server (RunPod B200).
 * Forwards capture data to gpu_server.py running on the remote machine.
 * Accessed via SSH tunnel: localhost:5555 -> RunPod:5555
 *
 * Falls back to local GPU detection via nvidia-smi when remote is down.
 */
import { execSync } from 'child_process';

const GPU_URL = process.env.GPU_REMOTE_URL || 'http://localhost:5555';

function detectLocalGpu() {
	try {
		const out = execSync('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits', {
			timeout: 3000, encoding: 'utf-8',
		}).trim();
		if (!out) return null;
		const [name, mem] = out.split(',').map(s => s.trim());
		return {
			available: true,
			name,
			vram_gb: Math.round(parseInt(mem) / 1024 * 10) / 10,
		};
	} catch {
		return null;
	}
}

export async function GET() {
	// Try remote GPU server first
	try {
		const res = await fetch(GPU_URL, { signal: AbortSignal.timeout(5000) });
		const info = await res.json();
		return new Response(JSON.stringify({ ...info, url: GPU_URL, connected: true, source: 'remote' }), {
			headers: { 'Content-Type': 'application/json' },
		});
	} catch {
		// Remote down — try local GPU
	}

	const local = detectLocalGpu();
	if (local) {
		return new Response(JSON.stringify({
			...local, connected: true, source: 'local',
		}), {
			headers: { 'Content-Type': 'application/json' },
		});
	}

	return new Response(JSON.stringify({
		connected: false, url: GPU_URL, error: 'No remote or local GPU found',
	}), {
		status: 200,
		headers: { 'Content-Type': 'application/json' },
	});
}

export async function POST({ request }) {
	const body = await request.json();
	const endpoint = body.deep ? '/deep' : '/optimize';

	try {
		const res = await fetch(`${GPU_URL}${endpoint}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				capture: body.capture,
				params: body.params || {},
			}),
			signal: AbortSignal.timeout(body.params?.budget ? (body.params.budget + 60) * 1000 : 300000),
		});
		const result = await res.json();
		return new Response(JSON.stringify(result), {
			headers: { 'Content-Type': 'application/json' },
		});
	} catch (e) {
		return new Response(JSON.stringify({ error: e.message }), {
			status: 502,
			headers: { 'Content-Type': 'application/json' },
		});
	}
}
