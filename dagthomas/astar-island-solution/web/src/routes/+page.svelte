<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import ScoreCard from '$lib/components/data/ScoreCard.svelte';
	import GaugeBar from '$lib/components/charts/GaugeBar.svelte';
	import Sparkline from '$lib/components/charts/Sparkline.svelte';
	import LeaderboardTable from '$lib/components/data/LeaderboardTable.svelte';
	import RoundTable from '$lib/components/data/RoundTable.svelte';
	import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
	import { scoreColor } from '$lib/api';
	import type { Round, MyRound, LeaderboardEntry, Metrics } from '$lib/types';

	interface PageData {
		rounds: Round[];
		myRounds: MyRound[];
		leaderboard: LeaderboardEntry[];
		metrics: Metrics | null;
	}
	let { data }: { data: PageData } = $props();

	let activeRound = $derived(data.rounds.find((r) => r.status === 'active') ?? data.rounds[0]);
	let bestMyRound = $derived(
		data.myRounds.reduce(
			(best, r) =>
				r.round_score != null && (best == null || r.round_score > (best.round_score ?? 0))
					? r
					: best,
			null as typeof data.myRounds[0] | null
		)
	);
	let scoreHistory = $derived(
		data.myRounds
			.filter((r) => r.round_score != null)
			.map((r) => r.round_score!)
			.reverse()
	);
</script>

<div class="space-y-4">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-lg text-neon-cyan neon-text tracking-wider font-bold">DASHBOARD</h1>
			<p class="text-xs text-cyber-muted">Astar Island Competition Monitor</p>
		</div>
		{#if activeRound}
			<div class="text-right glass glass-border-glow px-4 py-2">
				<div class="text-xs text-cyber-muted">Active Round</div>
				<div class="text-neon-gold font-bold neon-text-gold text-lg">R{activeRound.round_number}</div>
			</div>
		{/if}
	</div>

	<!-- Score cards row -->
	<div class="grid grid-cols-4 gap-3">
		<ScoreCard
			label="Best Score"
			score={data.metrics?.best_score ?? null}
			subtitle="All-time best"
		/>
		<ScoreCard
			label="Latest Score"
			score={bestMyRound?.round_score ?? null}
			subtitle={bestMyRound ? `R${bestMyRound.round_number}` : ''}
		/>
		<ScoreCard
			label="Best Rank"
			score={bestMyRound?.rank ?? null}
			subtitle={bestMyRound?.rank != null ? `of ${data.leaderboard.length}` : ''}
		/>
		<div class="glass glass-border-glow glass-animated p-3 relative overflow-hidden">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Experiments</div>
			<div class="text-2xl font-bold text-neon-cyan" style="text-shadow: 0 0 12px rgba(0,255,240,0.4), 0 0 24px rgba(0,255,240,0.15)">
				{(data.metrics?.autoloop_count ?? 0).toLocaleString()}
			</div>
			<div class="text-xs text-cyber-muted mt-0.5">autoloop runs</div>
			<!-- Corner accents -->
			<div class="absolute top-0 left-0 w-2 h-2 border-t border-l border-neon-cyan/30"></div>
			<div class="absolute top-0 right-0 w-2 h-2 border-t border-r border-neon-cyan/30"></div>
			<div class="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-neon-cyan/30"></div>
			<div class="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-neon-cyan/30"></div>
		</div>
	</div>

	<!-- Main grid -->
	<div class="grid grid-cols-3 gap-4">
		<!-- My Rounds -->
		<div class="col-span-2">
			<GlassPanel>
				<div class="flex items-center justify-between mb-3">
					<h2 class="text-xs text-neon-cyan uppercase tracking-wider">My Rounds</h2>
					{#if scoreHistory.length > 1}
						<Sparkline data={scoreHistory} width={100} height={24} />
					{/if}
				</div>
				<RoundTable rounds={data.myRounds} />
			</GlassPanel>
		</div>

		<!-- Sidebar column -->
		<div class="space-y-4">
			<!-- Budget -->
			{#if data.metrics}
				<GlassPanel>
					<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Query Budget</h2>
					<GaugeBar
						value={data.metrics.queries_used}
						max={data.metrics.queries_max}
						label="Queries Used"
					/>
					<div class="mt-3 grid grid-cols-2 gap-2 text-xs text-cyber-muted">
						<div>
							ADK:
							<span class="text-neon-orange">{data.metrics.adk_count}</span>
						</div>
						<div>
							Gemini:
							<span class="text-neon-magenta">{data.metrics.gemini_count}</span>
						</div>
						<div>
							Multi:
							<span class="text-neon-gold">{data.metrics.multi_count}</span>
						</div>
						<div>
							Autoloop:
							<span class="text-neon-cyan">{data.metrics.autoloop_count}</span>
						</div>
					</div>
				</GlassPanel>
			{/if}

			<!-- Leaderboard -->
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Leaderboard</h2>
				<LeaderboardTable entries={data.leaderboard.slice(0, 10)} />
			</GlassPanel>
		</div>
	</div>
</div>
