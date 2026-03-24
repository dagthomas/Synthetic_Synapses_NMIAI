<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';
	import { base } from '$app/paths';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import { scoreColor } from '$lib/chart-theme';

	let dashboard = $state<any>(null);
	let topTeams = $state<any[]>([]);
	let recentRounds = $state<any[]>([]);
	let loading = $state(true);

	onMount(async () => {
		try {
			const [d, ts, rs] = await Promise.all([
				admin.dashboard(),
				admin.statsTeams().catch(() => ({ teams: [] })),
				admin.rounds().catch(() => [])
			]);
			dashboard = d;
			topTeams = (ts.teams || [])
				.sort((a: any, b: any) => b.cumulative_weighted_score - a.cumulative_weighted_score)
				.slice(0, 5);
			recentRounds = (rs || []).slice(0, 5);
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});
</script>

<h1 class="text-2xl font-bold mb-6 text-neon-cyan neon-text tracking-wider uppercase">COMMAND CENTER</h1>

{#if loading}
	<p class="text-cyber-muted animate-pulse-glow">Loading...</p>
{:else if dashboard}
	<div class="grid grid-cols-4 gap-4 mb-6">
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Corporations</div>
			<div class="text-3xl font-bold text-neon-cyan neon-text">{dashboard.team_count}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Planets</div>
			<div class="text-3xl font-bold text-neon-gold neon-text-gold">{dashboard.total_rounds}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Scan Reports</div>
			<div class="text-3xl font-bold text-neon-magenta">{dashboard.total_predictions}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Active Planet</div>
			<div class="text-3xl font-bold text-cyber-fg">
				{#if dashboard.active_round}
					<span class="text-score-great">#{dashboard.active_round.round_number}</span>
				{:else}
					<span class="text-cyber-muted">None</span>
				{/if}
			</div>
		</div>
	</div>

	{#if dashboard.active_round}
		<div class="glass glass-glow p-6 mb-6">
			<div class="flex items-center gap-3 mb-3">
				<div class="w-1 h-5 bg-neon-cyan rounded-full animate-pulse-glow"></div>
				<h2 class="text-lg font-semibold text-cyber-fg tracking-wide">Planet #{dashboard.active_round.round_number}</h2>
				<StatusBadge status={dashboard.active_round.status} />
			</div>
			<div class="text-sm text-cyber-muted">
				Closes: <span class="text-neon-orange">{dashboard.active_round.closes_at || 'N/A'}</span>
			</div>
		</div>
	{/if}

	<div class="grid grid-cols-2 gap-4">
		<!-- Top 5 Teams -->
		{#if topTeams.length > 0}
			<div class="glass glass-glow p-4">
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-3">Top Corporations</h2>
				<div class="space-y-2">
					{#each topTeams as team, i}
						<a href="{base}/teams/{team.team_id}" class="flex items-center justify-between py-1.5 px-2 rounded hover:bg-neon-cyan/5 transition-colors group">
							<div class="flex items-center gap-2">
								<span class="text-xs font-bold w-5" class:text-neon-gold={i === 0} class:text-cyber-muted={i > 0}>
									{i === 0 ? '★' : `#${i + 1}`}
								</span>
								<span class="text-sm text-cyber-fg group-hover:text-neon-cyan transition-colors">{team.team_name}</span>
							</div>
							<span class="text-xs font-mono text-neon-cyan">{team.cumulative_weighted_score.toFixed(1)} pts</span>
						</a>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Recent Rounds -->
		{#if recentRounds.length > 0}
			<div class="glass glass-glow p-4">
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-3">Recent Planets</h2>
				<div class="space-y-2">
					{#each recentRounds as round}
						<a href="{base}/rounds/{round.id}" class="flex items-center justify-between py-1.5 px-2 rounded hover:bg-neon-cyan/5 transition-colors group">
							<div class="flex items-center gap-3">
								<span class="text-sm font-bold text-neon-cyan group-hover:text-neon-cyan/80">#{round.round_number}</span>
								<StatusBadge status={round.status} />
							</div>
							<div class="flex items-center gap-3 text-xs">
								<span class="text-cyber-muted">{round.teams_participated} corps</span>
								{#if round.avg_score != null}
									<span class="font-mono" style="color: {scoreColor(round.avg_score)}">{round.avg_score.toFixed(1)}</span>
								{:else}
									<span class="text-cyber-border">—</span>
								{/if}
							</div>
						</a>
					{/each}
				</div>
			</div>
		{/if}
	</div>
{/if}
