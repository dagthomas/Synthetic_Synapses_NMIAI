import { fetchAPI } from '$lib/api';
import type { AutoloopEntry, ProcessInfo } from '$lib/types';

export async function load({ fetch }) {
	try {
		const [entries, processes] = await Promise.all([
			fetchAPI<AutoloopEntry[]>('/api/logs/autoloop?last=200', fetch),
			fetchAPI<ProcessInfo[]>('/api/processes', fetch)
		]);
		return { entries, processes };
	} catch {
		return { entries: [] as AutoloopEntry[], processes: [] as ProcessInfo[] };
	}
}
