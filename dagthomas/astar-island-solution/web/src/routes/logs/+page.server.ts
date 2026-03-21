import { fetchAPI } from '$lib/api';

export async function load({ fetch }) {
	try {
		const entries = await fetchAPI<any[]>('/api/logs/autoloop?last=200', fetch);
		return { entries };
	} catch {
		return { entries: [] };
	}
}
