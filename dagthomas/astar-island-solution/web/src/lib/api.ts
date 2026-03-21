import { browser } from '$app/environment';
import { env as publicEnv } from '$env/dynamic/public';

// Server-side: internal Docker network (api container) or localhost fallback
// Browser-side: host-mapped port so the browser can reach the Go API
const GO_API_INTERNAL = publicEnv.PUBLIC_GO_API_INTERNAL || 'http://localhost:7091';
const GO_API_PUBLIC = publicEnv.PUBLIC_GO_API_PUBLIC || 'http://localhost:7091';

export const GO_API = browser ? GO_API_PUBLIC : GO_API_INTERNAL;

export async function fetchAPI<T>(path: string, fetchFn: typeof fetch = fetch): Promise<T> {
	const base = browser ? GO_API_PUBLIC : GO_API_INTERNAL;
	const res = await fetchFn(`${base}${path}`);
	if (!res.ok) {
		const body = await res.text();
		throw new Error(`API ${res.status}: ${body}`);
	}
	return res.json();
}

export async function postAPI<T>(path: string, body: unknown): Promise<T> {
	const base = browser ? GO_API_PUBLIC : GO_API_INTERNAL;
	const res = await fetch(`${base}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) {
		const text = await res.text();
		throw new Error(`API ${res.status}: ${text}`);
	}
	return res.json();
}

export async function deleteAPI(path: string): Promise<void> {
	const base = browser ? GO_API_PUBLIC : GO_API_INTERNAL;
	const res = await fetch(`${base}${path}`, { method: 'DELETE' });
	if (!res.ok) {
		const text = await res.text();
		throw new Error(`API ${res.status}: ${text}`);
	}
}

export function scoreColor(score: number): string {
	if (score >= 93) return 'var(--color-score-great)';
	if (score >= 90) return 'var(--color-score-good)';
	if (score >= 85) return 'var(--color-score-ok)';
	if (score >= 80) return 'var(--color-score-low)';
	return 'var(--color-score-bad)';
}

export function scoreClass(score: number): string {
	if (score >= 93) return 'text-score-great';
	if (score >= 90) return 'text-score-good';
	if (score >= 85) return 'text-score-ok';
	if (score >= 80) return 'text-score-low';
	return 'text-score-bad';
}

export function formatTime(iso: string): string {
	if (!iso) return '—';
	const d = new Date(iso);
	return d.toLocaleString('nb-NO', {
		month: 'short',
		day: 'numeric',
		hour: '2-digit',
		minute: '2-digit'
	});
}

export function formatDuration(seconds: number): string {
	if (seconds < 60) return `${seconds.toFixed(1)}s`;
	const m = Math.floor(seconds / 60);
	const s = seconds % 60;
	return `${m}m ${s.toFixed(0)}s`;
}
