<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import Sparkline from '$lib/components/charts/Sparkline.svelte';
	import GaugeBar from '$lib/components/charts/GaugeBar.svelte';
	import type { Metrics, AutoloopEntry } from '$lib/types';

	let { data }: { data: { metrics: Metrics | null; autoloopEntries: AutoloopEntry[] } } = $props();

	let acceptedScores = $derived(
		(data.autoloopEntries ?? [])
			.filter((e: AutoloopEntry) => e.accepted)
			.map((e: AutoloopEntry) => {
				const vals = Object.values(e.scores_quick || {});
				return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
			})
	);

	let allScores = $derived(
		(data.autoloopEntries ?? []).map((e: AutoloopEntry) => {
			const vals = Object.values(e.scores_quick || {});
			return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
		})
	);

	let elapsedTimes = $derived((data.autoloopEntries ?? []).map((e: AutoloopEntry) => e.elapsed));

	// Parameter frequency heatmap — count how often each param name appears
	let paramCounts = $derived.by(() => {
		const counts: Record<string, number> = {};
		for (const e of (data.autoloopEntries ?? []) as AutoloopEntry[]) {
			for (const key of Object.keys(e.params || {})) {
				counts[key] = (counts[key] || 0) + 1;
			}
		}
		return Object.entries(counts)
			.sort((a, b) => b[1] - a[1])
			.slice(0, 15);
	});
</script>

<div class="space-y-4">
	<h1 class="text-lg text-neon-cyan neon-text tracking-wider">METRICS</h1>

	{#if !data.metrics}
		<div class="text-cyber-muted text-center py-8">No metrics available. Start the Go API.</div>
	{:else}
		<!-- Gauges -->
		<div class="grid grid-cols-2 gap-4">
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Query Budget</h2>
				<GaugeBar
					value={data.metrics.queries_used}
					max={data.metrics.queries_max}
					label="Queries"
				/>
			</GlassPanel>
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Best Score</h2>
				<div class="text-4xl font-bold text-neon-gold text-center">
					{data.metrics.best_score.toFixed(2)}
				</div>
			</GlassPanel>
		</div>

		<!-- Charts -->
		<div class="grid grid-cols-2 gap-4">
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Score Trend (Accepted)</h2>
				{#if acceptedScores.length > 1}
					<Sparkline data={acceptedScores} width={400} height={80} color="var(--color-neon-gold)" />
				{:else}
					<div class="text-cyber-muted text-center py-4">Not enough data</div>
				{/if}
			</GlassPanel>
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">All Scores</h2>
				{#if allScores.length > 1}
					<Sparkline data={allScores} width={400} height={80} color="var(--color-neon-cyan)" />
				{:else}
					<div class="text-cyber-muted text-center py-4">Not enough data</div>
				{/if}
			</GlassPanel>
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Elapsed Time</h2>
				{#if elapsedTimes.length > 1}
					<Sparkline
						data={elapsedTimes}
						width={400}
						height={80}
						color="var(--color-neon-orange)"
					/>
				{:else}
					<div class="text-cyber-muted text-center py-4">Not enough data</div>
				{/if}
			</GlassPanel>
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Top Parameters</h2>
				{#if paramCounts.length}
					<div class="space-y-1">
						{#each paramCounts as [name, count]}
							{@const pct = (count / (data.autoloopEntries?.length || 1)) * 100}
							<div class="flex items-center gap-2 text-[10px]">
								<span class="w-28 truncate text-cyber-muted">{name}</span>
								<div class="flex-1 h-2 bg-cyber-surface rounded overflow-hidden">
									<div
										class="h-full rounded bg-neon-magenta/60"
										style="width: {pct}%"
									></div>
								</div>
								<span class="w-8 text-right text-cyber-muted">{count}</span>
							</div>
						{/each}
					</div>
				{:else}
					<div class="text-cyber-muted text-center py-4">No parameters yet</div>
				{/if}
			</GlassPanel>
		</div>
	{/if}
</div>
