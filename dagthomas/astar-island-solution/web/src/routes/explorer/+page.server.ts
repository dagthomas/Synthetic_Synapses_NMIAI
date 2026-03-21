import { fetchAPI } from '$lib/api';
import type { Round, RoundDetail } from '$lib/types';

export async function load({ fetch }) {
	try {
		const rounds = await fetchAPI<Round[]>('/api/rounds', fetch);
		const active = rounds.find((r) => r.status === 'active') ?? rounds[0];

		let detail: RoundDetail | null = null;
		if (active) {
			detail = await fetchAPI<RoundDetail>(`/api/rounds/${active.id}`, fetch);
		}

		return { rounds, detail, activeRoundId: active?.id ?? null };
	} catch {
		return { rounds: [] as Round[], detail: null, activeRoundId: null };
	}
}
