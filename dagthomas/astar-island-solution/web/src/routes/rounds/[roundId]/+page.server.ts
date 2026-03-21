import { fetchAPI } from '$lib/api';
import type { RoundDetail, MyRound, Analysis } from '$lib/types';

export async function load({ params, fetch }) {
	const { roundId } = params;

	try {
		const [detail, myRounds] = await Promise.all([
			fetchAPI<RoundDetail>(`/api/rounds/${roundId}`, fetch),
			fetchAPI<MyRound[]>('/api/my-rounds', fetch)
		]);

		const myRound = myRounds.find((r) => r.id === roundId) ?? null;

		// Try to get analysis for seed 0
		let analysis: Analysis | null = null;
		try {
			analysis = await fetchAPI<Analysis>(
				`/api/rounds/${roundId}/seeds/0/analysis`,
				fetch
			);
		} catch {
			// Analysis may not be available yet
		}

		return { detail, myRound, analysis };
	} catch {
		return { detail: null, myRound: null, analysis: null };
	}
}
