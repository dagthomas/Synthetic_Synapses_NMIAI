import { fetchAPI } from '$lib/api';
import type { Round, MyRound, LeaderboardEntry, Metrics } from '$lib/types';

export async function load({ fetch }) {
	try {
		const [rounds, myRounds, leaderboard, metrics] = await Promise.all([
			fetchAPI<Round[]>('/api/rounds', fetch),
			fetchAPI<MyRound[]>('/api/my-rounds', fetch),
			fetchAPI<LeaderboardEntry[]>('/api/leaderboard', fetch),
			fetchAPI<Metrics>('/api/metrics', fetch)
		]);

		return { rounds, myRounds, leaderboard, metrics };
	} catch {
		return {
			rounds: [] as Round[],
			myRounds: [] as MyRound[],
			leaderboard: [] as LeaderboardEntry[],
			metrics: null
		};
	}
}
