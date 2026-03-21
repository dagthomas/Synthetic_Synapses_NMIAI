import { fetchAPI } from '$lib/api';
import type { Metrics, AutoloopEntry } from '$lib/types';

export async function load({ fetch }) {
	try {
		const [metrics, autoloopEntries] = await Promise.all([
			fetchAPI<Metrics>('/api/metrics', fetch),
			fetchAPI<AutoloopEntry[]>('/api/logs/autoloop?last=500', fetch)
		]);
		return { metrics, autoloopEntries };
	} catch {
		return { metrics: null, autoloopEntries: [] as AutoloopEntry[] };
	}
}
