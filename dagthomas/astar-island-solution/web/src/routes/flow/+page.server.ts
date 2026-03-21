import { fetchAPI } from '$lib/api';
import type {
	ProcessInfo,
	DaemonAutoloop,
	DaemonParams,
	DaemonRoundScore,
	Metrics,
	Budget,
} from '$lib/types';

export async function load({ fetch }) {
	try {
		const [processes, autoloop, params, scores, status, metrics, budget] = await Promise.all([
			fetchAPI<ProcessInfo[]>('/api/processes', fetch),
			fetchAPI<DaemonAutoloop>('/api/daemon/autoloop', fetch),
			fetchAPI<DaemonParams>('/api/daemon/params', fetch),
			fetchAPI<DaemonRoundScore[]>('/api/daemon/scores', fetch),
			fetchAPI<string[]>('/api/daemon/status', fetch),
			fetchAPI<Metrics>('/api/metrics', fetch),
			fetchAPI<Budget>('/api/budget', fetch),
		]);
		return { processes, autoloop, params, scores, status, metrics, budget };
	} catch {
		return {
			processes: [],
			autoloop: null,
			params: null,
			scores: [],
			status: [],
			metrics: null,
			budget: null,
		};
	}
}
