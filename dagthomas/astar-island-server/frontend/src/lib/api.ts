import { browser } from '$app/environment';

const BASE = browser ? '' : 'http://localhost:8080';

function getToken(): string | null {
	if (!browser) return null;
	return localStorage.getItem('admin_token');
}

export function setToken(token: string) {
	if (browser) localStorage.setItem('admin_token', token);
}

export function clearToken() {
	if (browser) localStorage.removeItem('admin_token');
}

export function isLoggedIn(): boolean {
	return !!getToken();
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
	const token = getToken();
	const headers: Record<string, string> = {
		'Content-Type': 'application/json',
		...((options.headers as Record<string, string>) || {})
	};
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}

	const res = await fetch(`${BASE}${path}`, { ...options, headers });
	if (!res.ok) {
		const text = await res.text();
		throw new Error(`${res.status}: ${text}`);
	}
	return res.json();
}

// Auth
export async function login(name: string, password: string) {
	const data = await apiFetch<{ token: string; team_id: string; team_name: string; is_admin: boolean }>('/auth/login', {
		method: 'POST',
		body: JSON.stringify({ name, password })
	});
	setToken(data.token);
	return data;
}

// Admin API
export const admin = {
	dashboard: () => apiFetch<any>('/admin/api/dashboard'),
	rounds: () => apiFetch<any[]>('/admin/api/rounds'),
	roundDetail: (id: string) => apiFetch<any>(`/admin/api/rounds/${id}`),
	createRound: (regime: string) =>
		apiFetch<any>('/admin/api/rounds', { method: 'POST', body: JSON.stringify({ regime }) }),
	activateRound: (id: string) =>
		apiFetch<any>(`/admin/api/rounds/${id}/activate`, { method: 'POST' }),
	scoreRound: (id: string) =>
		apiFetch<any>(`/admin/api/rounds/${id}/score`, { method: 'POST' }),
	teams: () => apiFetch<any[]>('/admin/api/teams'),
	teamDetail: (id: string) => apiFetch<any>(`/admin/api/teams/${id}`),
	stats: () => apiFetch<any>('/admin/api/stats'),
	statsRounds: () => apiFetch<any>('/admin/api/stats/rounds'),
	statsTeams: () => apiFetch<any>('/admin/api/stats/teams'),
	statsPredictions: () => apiFetch<any>('/admin/api/stats/predictions'),
	statsQueries: () => apiFetch<any>('/admin/api/stats/queries'),
	statsParams: () => apiFetch<any>('/admin/api/stats/params')
};
