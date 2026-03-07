/**
 * Proxy to remote GPU server (RunPod B200).
 * Forwards capture data to gpu_server.py running on the remote machine.
 * Accessed via SSH tunnel: localhost:5555 -> RunPod:5555
 */

const GPU_URL = process.env.GPU_REMOTE_URL || 'http://localhost:5555';

export async function GET() {
	try {
		const res = await fetch(GPU_URL, { signal: AbortSignal.timeout(5000) });
		const info = await res.json();
		return new Response(JSON.stringify({ ...info, url: GPU_URL, connected: true }), {
			headers: { 'Content-Type': 'application/json' },
		});
	} catch (e) {
		return new Response(JSON.stringify({
			connected: false, url: GPU_URL, error: e.message,
		}), {
			status: 200,
			headers: { 'Content-Type': 'application/json' },
		});
	}
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
