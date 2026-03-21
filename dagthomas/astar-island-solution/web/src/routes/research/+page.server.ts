import { fetchAPI } from '$lib/api';
import type { ResearchEntry } from '$lib/types';
import type { ProcessInfo } from '$lib/types';

export async function load({ fetch }) {
	try {
		const [adk, gemini, multi, processes] = await Promise.all([
			fetchAPI<ResearchEntry[]>('/api/logs/adk?last=100', fetch),
			fetchAPI<ResearchEntry[]>('/api/logs/gemini?last=100', fetch),
			fetchAPI<ResearchEntry[]>('/api/logs/multi?last=100', fetch),
			fetchAPI<ProcessInfo[]>('/api/processes', fetch)
		]);
		return { adk, gemini, multi, processes };
	} catch {
		return {
			adk: [] as ResearchEntry[],
			gemini: [] as ResearchEntry[],
			multi: [] as ResearchEntry[],
			processes: [] as ProcessInfo[]
		};
	}
}
