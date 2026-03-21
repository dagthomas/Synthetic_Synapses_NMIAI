<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import ScoreCard from '$lib/components/data/ScoreCard.svelte';
	import Sparkline from '$lib/components/charts/Sparkline.svelte';
	import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
	import { scoreColor, scoreClass, formatTime } from '$lib/api';
	import type { DaemonParams, DaemonAutoloop, DaemonRoundScore, AutoloopEntry } from '$lib/types';
	import { invalidateAll } from '$app/navigation';

	interface PageData {
		status: string[];
		params: DaemonParams | null;
		autoloop: DaemonAutoloop | null;
		scores: DaemonRoundScore[];
	}
	let { data }: { data: PageData } = $props();

	// Auto-refresh every 5 seconds
	$effect(() => {
		const interval = setInterval(() => {
			invalidateAll();
		}, 5000);
		return () => clearInterval(interval);
	});

	// Derived data
	let autoloop = $derived(data.autoloop);
	let params = $derived(data.params);
	let scores = $derived(data.scores ?? []);

	let totalExperiments = $derived(autoloop?.total_experiments ?? 0);
	let bestScore = $derived(autoloop?.best_score ?? 0);
	let bestBoom = $derived(autoloop?.best_boom ?? 0);
	let bestNonboom = $derived(autoloop?.best_nonboom ?? 0);
	let expPerHour = $derived(autoloop?.experiments_per_hour ?? 0);
	let topParams = $derived(autoloop?.top_params ?? []);

	// Accepted experiments for sparkline (last 20)
	let acceptedScores = $derived(
		topParams
			.slice(-20)
			.map((e: AutoloopEntry) => e.scores_quick?.avg ?? e.scores_full?.avg ?? 0)
			.filter((v: number) => v > 0)
	);

	// Last 10 accepted experiments for findings panel
	let recentAccepted = $derived(
		topParams
			.filter((e: AutoloopEntry) => e.accepted)
			.slice(-10)
			.reverse()
	);

	// Score bar chart helpers
	let maxScore = $derived(
		scores.length ? Math.max(...scores.map((s: DaemonRoundScore) => s.avg_score)) : 100
	);

	function regimeColor(regime: string): string {
		switch (regime.toLowerCase()) {
			case 'boom':
				return 'var(--color-score-great)';
			case 'moderate':
				return 'var(--color-score-ok)';
			case 'collapse':
				return 'var(--color-score-bad)';
			default:
				return 'var(--color-cyber-muted)';
		}
	}

	function regimeBg(regime: string): string {
		switch (regime.toLowerCase()) {
			case 'boom':
				return 'rgba(0, 230, 118, 0.15)';
			case 'moderate':
				return 'rgba(212, 168, 67, 0.15)';
			case 'collapse':
				return 'rgba(229, 57, 53, 0.15)';
			default:
				return 'rgba(107, 107, 123, 0.1)';
		}
	}

	// Default parameter values for color-coding
	const defaultParams: Record<string, number> = {
		prior_w: 0.3,
		emp_max: 0.7,
		exp_damp: 0.5,
		base_power: 1.0,
		T_high: 0.15,
		smooth_alpha: 0.3,
		floor: 0.01
	};

	function paramDiffClass(key: string, value: number): string {
		const def = defaultParams[key];
		if (def === undefined) return 'text-cyber-fg';
		const pctDiff = Math.abs(value - def) / (def || 1);
		if (pctDiff > 0.5) return 'text-neon-gold';
		if (pctDiff > 0.2) return 'text-neon-orange';
		return 'text-cyber-fg';
	}

	// Settlement % sparkline data
	let settlementPcts = $derived(
		scores.map((s: DaemonRoundScore) => s.settlement_pct)
	);

	// Simulated log entries from status array
	let logEntries = $derived(
		(data.status ?? []).slice(-20).reverse()
	);

	function logLevelClass(line: string): string {
		if (line.includes('ERROR') || line.includes('error')) return 'text-score-bad';
		if (line.includes('WARN') || line.includes('warn')) return 'text-neon-orange';
		return 'text-cyber-muted';
	}
</script>

<div class="space-y-4">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-lg text-neon-cyan neon-text tracking-wider">DAEMON</h1>
			<p class="text-xs text-cyber-muted">Autonomous round monitor & parameter daemon</p>
		</div>
		<div class="flex items-center gap-3">
			<StatusBadge state={data.status.length > 0 ? 'Running' : 'Stopped'} />
			<div class="text-xs text-cyber-muted">
				Auto-refresh <span class="text-neon-cyan">5s</span>
			</div>
		</div>
	</div>

	<!-- Row 1: Score Cards -->
	<div class="grid grid-cols-5 gap-3">
		<div class="glass p-3 text-center">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Experiments</div>
			<div class="text-2xl font-bold text-neon-cyan">{totalExperiments.toLocaleString()}</div>
			<div class="text-xs text-cyber-muted mt-0.5">total runs</div>
		</div>
		<ScoreCard
			label="Best Score"
			score={bestScore || null}
			subtitle="avg across rounds"
		/>
		<ScoreCard
			label="Boom Score"
			score={bestBoom || null}
			subtitle="boom rounds avg"
		/>
		<ScoreCard
			label="Non-boom"
			score={bestNonboom || null}
			subtitle="non-boom avg"
		/>
		<div class="glass p-3 text-center">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Exp/Hour</div>
			<div class="text-2xl font-bold text-neon-magenta">{expPerHour.toFixed(1)}</div>
			<div class="text-xs text-cyber-muted mt-0.5">throughput</div>
		</div>
	</div>

	<!-- Row 2: Charts -->
	<div class="grid grid-cols-2 gap-4">
		<!-- Left: Score by Round bar chart -->
		<GlassPanel>
			<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Score by Round</h2>
			{#if scores.length > 0}
				<div class="overflow-y-auto max-h-[300px] space-y-1">
					{#each scores as round}
						{@const barWidth = maxScore > 0 ? (round.avg_score / maxScore) * 100 : 0}
						<div class="flex items-center gap-2 group">
							<!-- Round label -->
							<div class="w-8 text-xs text-cyber-muted text-right shrink-0">
								R{round.round_number}
							</div>
							<!-- Bar -->
							<div class="flex-1 relative h-5">
								<div class="absolute inset-0 rounded-sm overflow-hidden" style="background: rgba(26, 26, 46, 0.6);">
									<div
										class="h-full rounded-sm transition-all duration-300"
										style="width: {barWidth}%; background: {regimeColor(round.regime)}; opacity: 0.7;"
									></div>
								</div>
								<!-- Overlay score text -->
								<div class="absolute inset-0 flex items-center px-2">
									<span class="text-xs font-bold" style="color: {regimeColor(round.regime)}; text-shadow: 0 0 4px rgba(0,0,0,0.8);">
										{(round.avg_score ?? 0).toFixed(1)}
									</span>
								</div>
							</div>
							<!-- Regime tag -->
							<div
								class="w-16 text-[13px] uppercase tracking-wider text-center py-0.5 rounded shrink-0"
								style="color: {regimeColor(round.regime)}; background: {regimeBg(round.regime)};"
							>
								{round.regime}
							</div>
						</div>
					{/each}
				</div>
			{:else}
				<div class="text-cyber-muted text-[13px] text-center py-8">No round scores yet</div>
			{/if}
		</GlassPanel>

		<!-- Right: Autoloop Convergence -->
		<GlassPanel>
			<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Autoloop Convergence</h2>
			{#if acceptedScores.length > 1}
				<div class="flex flex-col items-center justify-center py-4">
					<Sparkline data={acceptedScores} width={280} height={120} color="var(--color-neon-gold)" />
					<div class="flex justify-between w-full mt-2 text-[13px] text-cyber-muted px-2">
						<span>oldest</span>
						<span class="text-neon-gold">
							Latest: {acceptedScores[acceptedScores.length - 1]?.toFixed(2) ?? '—'}
						</span>
						<span>newest</span>
					</div>
				</div>
				<div class="grid grid-cols-3 gap-2 mt-3 text-center">
					<div>
						<div class="text-[13px] text-cyber-muted uppercase">Min</div>
						<div class="text-[13px] text-score-low">{Math.min(...acceptedScores).toFixed(2)}</div>
					</div>
					<div>
						<div class="text-[13px] text-cyber-muted uppercase">Avg</div>
						<div class="text-[13px] text-neon-cyan">
							{(acceptedScores.reduce((a: number, b: number) => a + b, 0) / acceptedScores.length).toFixed(2)}
						</div>
					</div>
					<div>
						<div class="text-[13px] text-cyber-muted uppercase">Max</div>
						<div class="text-[13px] text-score-great">{Math.max(...acceptedScores).toFixed(2)}</div>
					</div>
				</div>
			{:else}
				<div class="text-cyber-muted text-[13px] text-center py-8">
					Waiting for accepted experiments...
				</div>
			{/if}
		</GlassPanel>
	</div>

	<!-- Row 3: Best Parameters -->
	{#if params}
		<GlassPanel>
			<div class="flex items-center justify-between mb-3">
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider">Best Parameters</h2>
				<div class="flex items-center gap-3 text-xs text-cyber-muted">
					{#if params.source}
						<span>Source: <span class="text-neon-magenta">{params.source}</span></span>
					{/if}
					{#if params.updated_at}
						<span>Updated: <span class="text-cyber-fg">{formatTime(params.updated_at)}</span></span>
					{/if}
					{#if params.experiment_id}
						<span>Exp: <span class="text-neon-gold">#{params.experiment_id}</span></span>
					{/if}
				</div>
			</div>

			<!-- Parameter grid -->
			<div class="grid grid-cols-4 gap-3">
				{#each Object.entries({
					prior_w: params.prior_w,
					emp_max: params.emp_max,
					exp_damp: params.exp_damp,
					base_power: params.base_power,
					T_high: params.T_high,
					smooth_alpha: params.smooth_alpha,
					floor: params.floor
				}) as [key, value]}
					<div class="glass p-2.5 text-center relative overflow-hidden">
						<!-- Subtle glow for significantly different params -->
						{#if defaultParams[key] !== undefined && Math.abs(value - defaultParams[key]) / (defaultParams[key] || 1) > 0.5}
							<div class="absolute inset-0 bg-neon-gold/5 pointer-events-none"></div>
						{/if}
						<div class="text-[13px] uppercase tracking-wider text-cyber-muted mb-1">{key}</div>
						<div class="text-lg font-bold {paramDiffClass(key, value)}">
							{typeof value === 'number' ? value.toFixed(4) : value}
						</div>
						{#if defaultParams[key] !== undefined}
							<div class="text-xs text-cyber-muted/60 mt-0.5">
								default: {defaultParams[key]}
							</div>
						{/if}
					</div>
				{/each}

				<!-- Score summary card -->
				<div class="glass p-2.5 text-center">
					<div class="text-[13px] uppercase tracking-wider text-cyber-muted mb-1">Param Score</div>
					<div class="text-lg font-bold" style="color: {scoreColor(params.score_avg ?? 0)}">
						{(params.score_avg ?? 0).toFixed(2)}
					</div>
					<div class="text-xs text-cyber-muted/60 mt-0.5">
						B:{(params.score_boom ?? 0).toFixed(1)} / N:{(params.score_nonboom ?? 0).toFixed(1)}
					</div>
				</div>
			</div>
		</GlassPanel>
	{/if}

	<!-- Row 4: Findings + Logs -->
	<div class="grid grid-cols-2 gap-4">
		<!-- Left: Live autoloop findings -->
		<GlassPanel>
			<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Recent Accepted</h2>
			{#if recentAccepted.length > 0}
				<div class="overflow-y-auto max-h-[260px]">
					<table class="w-full text-[13px]">
						<thead class="sticky top-0 bg-cyber-panel">
							<tr class="text-cyber-muted border-b border-cyber-border">
								<th class="text-left py-1.5 px-2">#</th>
								<th class="text-left py-1.5 px-2">Name</th>
								<th class="text-right py-1.5 px-2">Score</th>
								<th class="text-right py-1.5 px-2">Delta</th>
							</tr>
						</thead>
						<tbody>
							{#each recentAccepted as entry, i}
								{@const avgScore = Object.values(entry.scores_quick || {}).length
									? Object.values(entry.scores_quick).reduce((a, b) => a + b, 0) /
										Object.values(entry.scores_quick).length
									: 0}
								{@const delta = avgScore - entry.baseline_avg}
								<tr class="border-b border-cyber-border/20 hover:bg-cyber-panel/50 transition-colors">
									<td class="py-1 px-2 text-cyber-muted text-xs">{entry.id}</td>
									<td class="py-1 px-2 truncate max-w-[140px] text-xs">{entry.name}</td>
									<td class="py-1 px-2 text-right font-bold" style="color: {scoreColor(avgScore)}">
										{avgScore.toFixed(2)}
									</td>
									<td class="py-1 px-2 text-right text-xs">
										{#if delta > 0}
											<span class="text-score-great">+{delta.toFixed(2)}</span>
										{:else if delta < 0}
											<span class="text-score-bad">{delta.toFixed(2)}</span>
										{:else}
											<span class="text-cyber-muted">0.00</span>
										{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<div class="text-cyber-muted text-[13px] text-center py-8">No accepted experiments yet</div>
			{/if}
		</GlassPanel>

		<!-- Right: Daemon Log -->
		<GlassPanel>
			<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Daemon Log</h2>
			{#if logEntries.length > 0}
				<div class="overflow-y-auto max-h-[260px] space-y-0.5">
					{#each logEntries as line}
						<div class="text-xs font-mono leading-relaxed py-0.5 px-2 rounded hover:bg-cyber-panel/50 {logLevelClass(line)}">
							{line}
						</div>
					{/each}
				</div>
			{:else}
				<div class="text-cyber-muted text-[13px] text-center py-8">No log entries</div>
			{/if}
		</GlassPanel>
	</div>

	<!-- Row 5: Settlement % by Round -->
	<GlassPanel>
		<div class="flex items-center justify-between mb-3">
			<h2 class="text-xs text-neon-cyan uppercase tracking-wider">Settlement % by Round</h2>
			<div class="text-[13px] text-cyber-muted">
				{scores.length} rounds tracked
			</div>
		</div>
		{#if settlementPcts.length > 1}
			<div class="flex flex-col items-center">
				<Sparkline data={settlementPcts} width={600} height={60} color="var(--color-neon-magenta)" />
				<div class="flex justify-between w-full mt-2 text-[13px] text-cyber-muted px-2">
					{#each scores as round, i}
						{#if i === 0 || i === scores.length - 1 || i === Math.floor(scores.length / 2)}
							<span>R{round.round_number}: {(round.settlement_pct ?? 0).toFixed(0)}%</span>
						{/if}
					{/each}
				</div>
			</div>
			<!-- Regime distribution -->
			<div class="flex items-center gap-4 mt-3 text-[13px] justify-center">
				<span class="text-score-great">Boom: {scores.filter((s) => s.regime.toLowerCase() === 'boom').length}</span>
				<span class="text-score-ok">Moderate: {scores.filter((s) => s.regime.toLowerCase() === 'moderate').length}</span>
				<span class="text-score-bad">Collapse: {scores.filter((s) => s.regime.toLowerCase() === 'collapse').length}</span>
			</div>
		{:else if settlementPcts.length === 1}
			<div class="text-cyber-muted text-[13px] text-center py-4">
				Only 1 round — need 2+ for sparkline
			</div>
		{:else}
			<div class="text-cyber-muted text-[13px] text-center py-4">No settlement data</div>
		{/if}
	</GlassPanel>
</div>
