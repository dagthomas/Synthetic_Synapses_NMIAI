<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';
	import { Separator } from '$lib/components/ui/separator';
	import CyberChart from '$lib/components/CyberChart.svelte';
	import { CYBER, TEAM_PALETTE, scoreColor, bucketColors } from '$lib/chart-theme';

	let stats = $state<any>(null);
	let roundStats = $state<any>(null);
	let teamStats = $state<any>(null);
	let predStats = $state<any>(null);
	let queryStats = $state<any>(null);
	let paramStats = $state<any>(null);
	let loading = $state(true);
	let selectedParam = $state('base_survival');

	onMount(async () => {
		try {
			const [s, rs, ts, ps, qs, prs] = await Promise.all([
				admin.stats(),
				admin.statsRounds(),
				admin.statsTeams(),
				admin.statsPredictions(),
				admin.statsQueries(),
				admin.statsParams()
			]);
			stats = s;
			roundStats = rs;
			teamStats = ts;
			predStats = ps;
			queryStats = qs;
			paramStats = prs;
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});

	// Derived chart data
	let histogramData = $derived.by(() => {
		if (!predStats) return null;
		const labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'];
		return {
			labels,
			datasets: [{
				label: 'Predictions',
				data: predStats.score_histogram,
				backgroundColor: bucketColors(10),
				borderColor: bucketColors(10),
				borderWidth: 1
			}]
		};
	});

	let scoresOverRoundsData = $derived.by(() => {
		if (!roundStats?.rounds?.length) return null;
		const completed = roundStats.rounds.filter((r: any) => r.status === 'completed');
		return {
			labels: completed.map((r: any) => `R${r.round_number}`),
			datasets: [
				{
					label: 'Average',
					data: completed.map((r: any) => r.avg_score),
					borderColor: CYBER.cyan,
					backgroundColor: 'rgba(0,255,240,0.1)',
					fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: CYBER.cyan
				},
				{
					label: 'Max',
					data: completed.map((r: any) => r.max_score),
					borderColor: CYBER.green,
					borderDash: [4, 4], tension: 0.3, pointRadius: 2
				},
				{
					label: 'Min',
					data: completed.map((r: any) => r.min_score),
					borderColor: CYBER.red,
					borderDash: [4, 4], tension: 0.3, pointRadius: 2
				}
			]
		};
	});

	let teamRankingData = $derived.by(() => {
		if (!teamStats?.teams?.length) return null;
		const sorted = [...teamStats.teams].sort((a: any, b: any) => b.cumulative_weighted_score - a.cumulative_weighted_score);
		return {
			labels: sorted.map((t: any) => t.team_name.length > 20 ? t.team_name.slice(0, 18) + '…' : t.team_name),
			datasets: [{
				label: 'Weighted Score',
				data: sorted.map((t: any) => Math.round(t.cumulative_weighted_score * 10) / 10),
				backgroundColor: sorted.map((_: any, i: number) => TEAM_PALETTE[i % TEAM_PALETTE.length]),
				borderWidth: 0
			}]
		};
	});

	let queryBudgetData = $derived.by(() => {
		if (!queryStats?.per_team_budget?.length) return null;
		const sorted = [...queryStats.per_team_budget].sort((a: any, b: any) => b.avg_per_round - a.avg_per_round);
		return {
			labels: sorted.map((t: any) => t.team_name.length > 20 ? t.team_name.slice(0, 18) + '…' : t.team_name),
			datasets: [{
				label: 'Avg Queries / Round',
				data: sorted.map((t: any) => Math.round(t.avg_per_round * 10) / 10),
				backgroundColor: CYBER.magenta + '80',
				borderColor: CYBER.magenta,
				borderWidth: 1
			}]
		};
	});

	let paramCorrelationData = $derived.by(() => {
		if (!paramStats?.rounds?.length) return null;
		const points = paramStats.rounds
			.filter((r: any) => r.avg_score != null && r.params[selectedParam] != null)
			.map((r: any) => ({ x: r.params[selectedParam], y: r.avg_score, label: `R${r.round_number}` }));
		return {
			datasets: [{
				label: selectedParam,
				data: points,
				backgroundColor: CYBER.gold,
				borderColor: CYBER.gold,
				pointRadius: 6,
				pointHoverRadius: 8
			}]
		};
	});

	let leaderboardProgressionData = $derived.by(() => {
		if (!teamStats?.teams?.length || !roundStats?.rounds?.length) return null;
		const completed = roundStats.rounds.filter((r: any) => r.status === 'completed');
		const labels = completed.map((r: any) => `R${r.round_number}`);
		const sorted = [...teamStats.teams].sort((a: any, b: any) => b.cumulative_weighted_score - a.cumulative_weighted_score).slice(0, 10);

		const datasets = sorted.map((team: any, i: number) => {
			let cumulative = 0;
			const data = completed.map((r: any) => {
				const roundEntry = team.scores_by_round?.find((s: any) => s.round_number === r.round_number);
				if (roundEntry?.avg_score != null) {
					cumulative += roundEntry.avg_score * roundEntry.round_weight;
				}
				return Math.round(cumulative * 10) / 10;
			});
			return {
				label: team.team_name.length > 16 ? team.team_name.slice(0, 14) + '…' : team.team_name,
				data,
				borderColor: TEAM_PALETTE[i % TEAM_PALETTE.length],
				backgroundColor: 'transparent',
				tension: 0.3,
				pointRadius: 2,
				borderWidth: 2
			};
		});

		return { labels, datasets };
	});

	let paramNames = $derived.by(() => {
		if (!paramStats?.rounds?.[0]?.params) return [];
		return Object.keys(paramStats.rounds[0].params).sort();
	});

	let bestTeam = $derived.by(() => {
		if (!teamStats?.teams?.length) return null;
		return [...teamStats.teams].sort((a: any, b: any) => b.cumulative_weighted_score - a.cumulative_weighted_score)[0];
	});
</script>

<h1 class="text-2xl font-bold mb-6 text-neon-cyan neon-text tracking-wider uppercase">GALACTIC INTELLIGENCE</h1>

{#if loading}
	<p class="text-cyber-muted animate-pulse-glow">Scanning data streams...</p>
{:else if stats}
	<!-- Metric Cards -->
	<div class="grid grid-cols-3 gap-4 mb-6">
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Corporations</div>
			<div class="text-3xl font-bold text-neon-cyan neon-text">{stats.team_count}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Completed Planets</div>
			<div class="text-3xl font-bold text-neon-gold neon-text-gold">{stats.completed_rounds} / {stats.total_rounds}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Global Avg Score</div>
			<div class="text-3xl font-bold text-neon-magenta">{stats.avg_score != null ? stats.avg_score.toFixed(1) : '-'}</div>
		</div>
	</div>
	<div class="grid grid-cols-3 gap-4 mb-8">
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Total Predictions</div>
			<div class="text-2xl font-bold text-score-great">{stats.total_predictions}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Total Scans</div>
			<div class="text-2xl font-bold text-neon-orange">{stats.total_queries}</div>
		</div>
		<div class="glass glass-glow p-4">
			<div class="text-[10px] text-cyber-muted tracking-widest uppercase mb-1">Top Corporation</div>
			<div class="text-lg font-bold text-neon-gold neon-text-gold truncate">{bestTeam?.team_name ?? '-'}</div>
			<div class="text-xs text-cyber-muted">{bestTeam ? `${bestTeam.cumulative_weighted_score.toFixed(1)} pts` : ''}</div>
		</div>
	</div>

	<!-- Charts Row 1: Histogram + Scores over rounds -->
	<div class="grid grid-cols-2 gap-4 mb-4">
		{#if histogramData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Score Distribution</h2>
				<CyberChart type="bar" data={histogramData} height="280px" options={{
					plugins: { legend: { display: false } },
					scales: { y: { title: { display: true, text: 'Predictions', color: CYBER.muted } } }
				}} />
			</div>
		{/if}
		{#if scoresOverRoundsData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Scores Over Rounds</h2>
				<CyberChart type="line" data={scoresOverRoundsData} height="280px" options={{
					scales: { y: { min: 0, max: 100, title: { display: true, text: 'Score', color: CYBER.muted } } }
				}} />
			</div>
		{/if}
	</div>

	<!-- Charts Row 2: Team ranking + Query budget -->
	<div class="grid grid-cols-2 gap-4 mb-4">
		{#if teamRankingData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Corporation Rankings</h2>
				<CyberChart type="bar" data={teamRankingData} height="360px" options={{
					indexAxis: 'y',
					plugins: { legend: { display: false } },
					scales: { x: { title: { display: true, text: 'Cumulative Weighted Score', color: CYBER.muted } } }
				}} />
			</div>
		{/if}
		{#if queryBudgetData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Scan Budget Utilization</h2>
				<CyberChart type="bar" data={queryBudgetData} height="360px" options={{
					indexAxis: 'y',
					plugins: { legend: { display: false } },
					scales: { x: { title: { display: true, text: 'Avg Queries / Round', color: CYBER.muted }, max: 50 } }
				}} />
			</div>
		{/if}
	</div>

	<!-- Team x Round Heatmap -->
	{#if predStats?.team_round_matrix?.length}
		<div class="mb-4">
			<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Corporation × Planet Score Matrix</h2>
			<div class="glass glass-glow p-4 overflow-x-auto">
				<table class="w-full text-xs font-mono">
					<thead>
						<tr>
							<th class="text-left text-cyber-muted px-2 py-1 sticky left-0 bg-cyber-panel">Corp</th>
							{#each predStats.round_numbers as rn}
								<th class="text-center text-cyber-muted px-2 py-1 min-w-[48px]">R{rn}</th>
							{/each}
						</tr>
					</thead>
					<tbody>
						{#each predStats.team_round_matrix as row}
							<tr class="border-t border-cyber-border/20">
								<td class="text-cyber-fg px-2 py-1.5 truncate max-w-[160px] sticky left-0 bg-cyber-panel" title={row.team_name}>
									{row.team_name.length > 18 ? row.team_name.slice(0, 16) + '…' : row.team_name}
								</td>
								{#each row.round_scores as score}
									<td class="text-center px-2 py-1.5">
										{#if score != null}
											<span
												class="inline-block rounded px-1.5 py-0.5 text-[10px] font-bold"
												style="background-color: {scoreColor(score)}22; color: {scoreColor(score)}; border: 1px solid {scoreColor(score)}44;"
											>
												{score.toFixed(0)}
											</span>
										{:else}
											<span class="text-cyber-border">—</span>
										{/if}
									</td>
								{/each}
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	{/if}

	<!-- Charts Row 3: Parameter correlation + Leaderboard progression -->
	<div class="grid grid-cols-2 gap-4 mb-4">
		{#if paramCorrelationData}
			<div>
				<div class="flex items-center gap-3 mb-2">
					<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase">Parameter Correlation</h2>
					<select
						bind:value={selectedParam}
						class="bg-cyber-surface border border-cyber-border/50 text-cyber-fg text-xs rounded px-2 py-1 font-mono"
					>
						{#each paramNames as p}
							<option value={p}>{p}</option>
						{/each}
					</select>
				</div>
				<CyberChart type="scatter" data={paramCorrelationData} height="320px" options={{
					plugins: {
						legend: { display: false },
						tooltip: {
							callbacks: {
								label: (ctx: any) => `R${ctx.raw.label?.replace('R', '') ?? '?'}: ${ctx.raw.x.toFixed(3)} → ${ctx.raw.y.toFixed(1)} pts`
							}
						}
					},
					scales: {
						x: { title: { display: true, text: selectedParam, color: CYBER.muted } },
						y: { title: { display: true, text: 'Avg Score', color: CYBER.muted }, min: 0, max: 100 }
					}
				}} />
			</div>
		{/if}
		{#if leaderboardProgressionData}
			<div>
				<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Leaderboard Progression (Top 10)</h2>
				<CyberChart type="line" data={leaderboardProgressionData} height="320px" options={{
					plugins: { legend: { position: 'right', labels: { boxWidth: 10, font: { size: 9 } } } },
					scales: { y: { title: { display: true, text: 'Cumulative Score', color: CYBER.muted } } }
				}} />
			</div>
		{/if}
	</div>

	<!-- Queries by round -->
	{#if queryStats?.queries_by_round?.length}
		<div class="mb-4">
			<h2 class="text-xs font-semibold text-neon-gold tracking-wider uppercase mb-2">Scan Activity by Planet</h2>
			<CyberChart type="bar" data={{
				labels: queryStats.queries_by_round.map((r: any) => `R${r.round_number}`),
				datasets: [{
					label: 'Total Queries',
					data: queryStats.queries_by_round.map((r: any) => r.total),
					backgroundColor: CYBER.cyan + '60',
					borderColor: CYBER.cyan,
					borderWidth: 1
				}]
			}} height="200px" options={{
				plugins: { legend: { display: false } },
				scales: { y: { title: { display: true, text: 'Queries', color: CYBER.muted } } }
			}} />
		</div>
	{/if}

	<!-- Parameter ranges reference -->
	{#if stats.param_ranges}
		<div class="glass glass-glow p-4">
			<h2 class="text-sm font-semibold mb-1 text-neon-gold neon-text-gold tracking-wider uppercase">Physics Variable Ranges</h2>
			<p class="text-[10px] text-cyber-muted tracking-widest uppercase mb-4">26 Planetary Physics Variables</p>
			<Separator class="bg-cyber-border/50 mb-4" />
			<div class="grid grid-cols-2 gap-2 text-sm">
				{#each stats.param_ranges as p}
					<div class="flex items-center gap-3 bg-cyber-surface/80 border border-cyber-border/30 rounded px-3 py-1.5">
						<span class="text-cyber-muted w-48 truncate text-xs" title={p.description}>{p.name}</span>
						<span class="text-neon-cyan/50 font-mono text-[10px]">[{p.min}, {p.max}]</span>
						<span class="font-mono ml-auto text-neon-gold text-xs">def: {p.default}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
{/if}
