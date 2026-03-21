import { fetchAPI } from '$lib/api';
import type { Budget, ProcessInfo } from '$lib/types';

export async function load({ fetch }) {
	let budget: Budget | null = null;
	let processes: ProcessInfo[] = [];
	let connected = false;

	try {
		[budget, processes] = await Promise.all([
			fetchAPI<Budget>('/api/budget', fetch),
			fetchAPI<ProcessInfo[]>('/api/processes', fetch)
		]);
		connected = true;
	} catch {
		// API offline
	}

	return { budget, processes, connected };
}
