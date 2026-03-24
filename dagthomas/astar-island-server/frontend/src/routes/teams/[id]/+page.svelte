<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { admin } from '$lib/api';
	import CyberChart from '$lib/components/CyberChart.svelte';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import * as Table from '$lib/components/ui/table';
	import { CYBER, scoreColor } from '$lib/chart-theme';

	let team = $state<any>(null);
	let loading = $state(true);

	onMount(async () => {
		try {
			const id = $page.params.id!;
			team = await admin.teamDetail(id);
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});

	let scoreHistoryData = $derived.by(() => {
		if (!team?.rounds?.length) return null;
		const completed = team.rounds.filter((r: any) => r.avg_score != null);
		return {
			labels: completed.map((r: any) => `R${r.round_number}`),
			datasets: [{
				label: 'Avg Score',
				data: completed.map((r: any) => r.avg_score),
				borderColor: CYBER.cyan,
				backgroundColor: 'rgba(0,255,240,0.1)',
				fill: true,
				tension: 0.3,
				pointRadius: 5,
				pointBackgroundColor: completed.map((r: any) => scoreColor(r.avg_score ?? 0))
			}]
		};
	});

	let queryHistoryData = $derived.by(() => {
		if (!team?.rounds?.length) return null;
		return {
			labels: team.rounds.map((r: any) => `R${r.round_number}`),
			datasets: [{
				label: 'Queries Used',
				data: team.rounds.map((r: any) => r.queries_used),
				backgroundColor: CYBER.magenta + '60',
				borderColor: CYBER.magenta,
				borderWidth: 1
			}]
		};
	});
</script>

{#if loading}
	<p class="text-cyber-muted animate-pulse-glow">Loading...</p>
{:else if team}
	<div class="flex items-center gap-4 mb-6">
		<h1 class="text-2xl font-bold text-neon-cyan neon-text tracking-wider uppercase">{team.name}</h1>
		{#if team.is_admin}
			<span class="text-[10px] tracking-wider uppercase bg-neon-magenta/15 text-neon-magenta border border-neon-magenta/30 rounded px-2 py-0.5">Admin</span>
		{/if}
	</div>

	<!-- Stats cards -->
	<div class="grid grid-cols-4 gap-4 mb-6">
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Rank</div>
			<div class="text-3xl font-bold text-neon-gold neon-text-gold">#{team.rank}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Weighted Score</div>
			<div class="text-2xl font-bold text-neon-cyan">{team.cumulative_weighted_score.toFixed(1)}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Planets Explored</div>
			<div class="text-2xl font-bold text-neon-magenta">{team.rounds.length}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Total Scans</div>
			<div class="text-2xl font-bold text-neon-orange">{team.total_queries}</div>
		</div>
	</div>

	<!-- Charts -->
	<div class="grid grid-cols-2 gap-4 mb-6">
		{#if scoreHistoryData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Score History</h2>
				<CyberChart type="line" data={scoreHistoryData} height="240px" options={{
					scales: { y: { min: 0, max: 100, title: { display: true, text: 'Score', color: CYBER.muted } } }
				}} />
			</div>
		{/if}
		{#if queryHistoryData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Scan Usage</h2>
				<CyberChart type="bar" data={queryHistoryData} height="240px" options={{
					plugins: { legend: { display: false } },
					scales: { y: { title: { display: true, text: 'Queries', color: CYBER.muted } } }
				}} />
			</div>
		{/if}
	</div>

	<!-- Per-round table -->
	<div class="glass glass-glow overflow-hidden">
		<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase p-4 pb-2">Planet Performance</h2>
		<Table.Root>
			<Table.Header>
				<Table.Row class="border-b border-cyber-border bg-cyber-surface/50 hover:bg-cyber-surface/50">
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Planet</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Status</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Avg Score</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Seed Scores</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Queries</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Weight</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Contribution</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each team.rounds as round}
					<Table.Row class="border-b border-cyber-border/50 hover:bg-neon-cyan/5 transition-colors">
						<Table.Cell class="font-bold text-neon-cyan">#{round.round_number}</Table.Cell>
						<Table.Cell><StatusBadge status={round.status} /></Table.Cell>
						<Table.Cell>
							{#if round.avg_score != null}
								<span class="font-mono" style="color: {scoreColor(round.avg_score)}">{round.avg_score.toFixed(1)}</span>
							{:else}
								<span class="text-cyber-muted">—</span>
							{/if}
						</Table.Cell>
						<Table.Cell class="font-mono text-xs">
							{#each round.seed_scores as s, i}
								{#if s != null}
									<span style="color: {scoreColor(s)}">{s.toFixed(0)}</span>
								{:else}
									<span class="text-cyber-border">—</span>
								{/if}
								{#if i < round.seed_scores.length - 1}
									<span class="text-cyber-border/40 mx-0.5">/</span>
								{/if}
							{/each}
						</Table.Cell>
						<Table.Cell class="text-cyber-fg">{round.queries_used}</Table.Cell>
						<Table.Cell class="text-cyber-muted font-mono text-xs">{round.round_weight.toFixed(2)}</Table.Cell>
						<Table.Cell class="text-neon-gold font-mono text-xs">{round.weighted_contribution.toFixed(1)}</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>
		</Table.Root>
	</div>

	<div class="mt-4 text-xs text-cyber-muted">
		Registered: {team.created_at?.slice(0, 10)}
	</div>
{:else}
	<p class="text-cyber-muted">Team not found.</p>
{/if}
