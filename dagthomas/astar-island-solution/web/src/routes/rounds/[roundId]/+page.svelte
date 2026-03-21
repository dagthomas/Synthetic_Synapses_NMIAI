<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import ScoreCard from '$lib/components/data/ScoreCard.svelte';
	import SeedSelector from '$lib/components/data/SeedSelector.svelte';
	import BarChart from '$lib/components/charts/BarChart.svelte';
	import Heatmap from '$lib/components/charts/Heatmap.svelte';
	import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
	import { fetchAPI, scoreColor, formatTime } from '$lib/api';
	import type { Analysis, RoundDetail, MyRound } from '$lib/types';

	interface PageData {
		detail: RoundDetail | null;
		myRound: MyRound | null;
		analysis: Analysis | null;
	}
	let { data }: { data: PageData } = $props();
	let selectedSeed = $state(0);
	let analysis = $state<Analysis | null>(data.analysis);

	async function loadAnalysis(idx: number) {
		selectedSeed = idx;
		if (!data.detail) return;
		try {
			analysis = await fetchAPI<Analysis>(
				`/api/rounds/${data.detail.id}/seeds/${idx}/analysis`
			);
		} catch {
			analysis = null;
		}
	}

	let seedScoreItems = $derived(
		(data.myRound?.seed_scores ?? []).map((s: number, i: number) => ({
			label: `Seed ${i}`,
			value: s
		}))
	);
</script>

<div class="space-y-4">
	<!-- Header -->
	<div class="flex items-center gap-4">
		<a href="/rounds" class="text-cyber-muted hover:text-neon-cyan text-sm">&larr; Rounds</a>
		{#if data.detail}
			<h1 class="text-lg text-neon-cyan neon-text tracking-wider">
				ROUND {data.detail.round_number}
			</h1>
			<StatusBadge state={data.detail.status} />
		{/if}
	</div>

	{#if !data.detail}
		<div class="text-cyber-muted text-center py-8">Round not found</div>
	{:else}
		<!-- Score cards -->
		<div class="grid grid-cols-4 gap-3">
			<ScoreCard label="Score" score={data.myRound?.round_score ?? null} />
			<ScoreCard label="Rank" score={data.myRound?.rank ?? null} />
			<div class="glass p-3 text-center">
				<div class="text-[10px] uppercase tracking-wider text-cyber-muted mb-1">Map Size</div>
				<div class="text-xl text-cyber-fg">
					{data.detail.map_width}x{data.detail.map_height}
				</div>
			</div>
			<div class="glass p-3 text-center">
				<div class="text-[10px] uppercase tracking-wider text-cyber-muted mb-1">Seeds</div>
				<div class="text-xl text-cyber-fg">
					{data.detail.initial_states?.length ?? 0}
				</div>
			</div>
		</div>

		<!-- Seed scores -->
		{#if seedScoreItems.length}
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Seed Scores</h2>
				<BarChart items={seedScoreItems} maxValue={100} />
			</GlassPanel>
		{/if}

		<!-- Analysis section -->
		<GlassPanel>
			<div class="flex items-center justify-between mb-3">
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider">Analysis</h2>
				<SeedSelector
					count={data.detail.initial_states?.length ?? 5}
					selected={selectedSeed}
					onselect={loadAnalysis}
				/>
			</div>

			{#if analysis}
				<div class="grid grid-cols-2 gap-4">
					<!-- Cell Scores Heatmap -->
					<div>
						<h3 class="text-[10px] text-cyber-muted uppercase mb-2">Cell Scores</h3>
						{#if analysis.cell_scores}
							<Heatmap data={analysis.cell_scores} size={360} />
						{:else}
							<div class="text-cyber-muted text-center py-4">No cell scores</div>
						{/if}
					</div>

					<!-- Score summary -->
					<div class="space-y-3">
						<div class="glass p-4 text-center">
							<div class="text-[10px] text-cyber-muted uppercase mb-1">Seed Score</div>
							<div class="text-3xl font-bold" style="color: {scoreColor(analysis.score)}">
								{analysis.score.toFixed(2)}
							</div>
						</div>

						<div class="text-[11px] text-cyber-muted space-y-1">
							<div>Round: {analysis.round_id}</div>
							<div>Seed index: {analysis.seed_index}</div>
							{#if analysis.ground_truth}
								<div>
									Grid: {analysis.ground_truth.length}x{analysis.ground_truth[0]?.length ?? 0}
								</div>
							{/if}
						</div>
					</div>
				</div>
			{:else}
				<div class="text-cyber-muted text-center py-8">
					Analysis not available for this seed
				</div>
			{/if}
		</GlassPanel>
	{/if}
</div>
