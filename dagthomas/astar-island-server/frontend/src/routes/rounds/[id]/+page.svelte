<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { admin } from '$lib/api';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import GridViewer from '$lib/components/GridViewer.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';

	let detail = $state<any>(null);
	let loading = $state(true);
	let activeSeed = $state(0);

	onMount(async () => {
		try {
			detail = await admin.roundDetail(page.params.id!);
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});

	function scoreColor(s: number | null): string {
		if (s == null) return 'text-cyber-muted/40';
		if (s >= 80) return 'text-score-great';
		if (s >= 60) return 'text-score-good';
		if (s >= 40) return 'text-score-ok';
		if (s >= 20) return 'text-score-low';
		return 'text-score-bad';
	}
</script>

{#if loading}
	<p class="text-cyber-muted animate-pulse-glow">Loading...</p>
{:else if detail}
	<div class="flex items-center gap-3 mb-6">
		<div class="w-1 h-6 bg-neon-cyan rounded-full animate-pulse-glow"></div>
		<h1 class="text-2xl font-bold text-neon-cyan neon-text tracking-wider uppercase">PLANET #{detail.round_number}</h1>
		<StatusBadge status={detail.status} />
	</div>

	<!-- Planetary Physics -->
	<div class="glass glass-glow p-4 mb-6">
		<h2 class="text-sm font-semibold mb-3 text-neon-gold neon-text-gold tracking-wider uppercase">Planetary Physics (26)</h2>
		<div class="grid grid-cols-4 gap-2 text-sm">
			{#each Object.entries(detail.hidden_params) as [key, value]}
				<div class="flex justify-between bg-cyber-surface/80 border border-cyber-border/30 rounded px-3 py-1.5">
					<span class="text-cyber-muted text-xs">{key}</span>
					<span class="font-mono text-neon-cyan text-xs">{typeof value === 'number' ? value.toFixed(3) : value}</span>
				</div>
			{/each}
		</div>
	</div>

	<!-- Seed tabs + grid viewer -->
	<div class="glass glass-glow p-4 mb-6">
		<div class="flex gap-2 mb-4">
			{#each detail.seeds as seed, i}
				<Button
					onclick={() => (activeSeed = i)}
					variant={activeSeed === i ? 'default' : 'ghost'}
					size="sm"
					class={activeSeed === i
						? 'bg-neon-cyan/15 border border-neon-cyan/40 text-neon-cyan shadow-[0_0_10px_rgba(0,255,240,0.1)]'
						: 'text-cyber-muted hover:text-neon-cyan hover:bg-cyber-panel border border-transparent'}
				>
					Scan Zone {seed.seed_index} ({seed.settlement_count} nodes)
				</Button>
			{/each}
		</div>

		{#if detail.seeds[activeSeed]?.initial_grid}
			<GridViewer grid={detail.seeds[activeSeed].initial_grid} />
		{/if}
	</div>

	<!-- Corporation Rankings -->
	{#if detail.team_scores.length > 0}
		<div class="glass glass-glow p-4">
			<h2 class="text-sm font-semibold mb-3 text-neon-magenta tracking-wider uppercase">Corporation Rankings</h2>
			<Table.Root>
				<Table.Header>
					<Table.Row class="border-b border-cyber-border bg-cyber-surface/50 hover:bg-cyber-surface/50">
						<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Corporation</Table.Head>
						{#each { length: 5 } as _, i}
							<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase text-center">Zone {i}</Table.Head>
						{/each}
						<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase text-center">Avg</Table.Head>
						<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase text-center">Queries</Table.Head>
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each detail.team_scores as ts}
						<Table.Row class="border-b border-cyber-border/50 hover:bg-neon-cyan/5 transition-colors">
							<Table.Cell class="text-cyber-fg font-medium">{ts.team_name}</Table.Cell>
							{#each ts.seed_scores as s}
								<Table.Cell class="text-center font-mono {scoreColor(s)}">
									{s != null ? s.toFixed(1) : '-'}
								</Table.Cell>
							{/each}
							<Table.Cell class="text-center font-bold text-neon-gold neon-text-gold">
								{ts.average_score != null ? ts.average_score.toFixed(1) : '-'}
							</Table.Cell>
							<Table.Cell class="text-center text-cyber-muted">{ts.queries_used}/50</Table.Cell>
						</Table.Row>
					{/each}
				</Table.Body>
			</Table.Root>
		</div>
	{/if}
{/if}
