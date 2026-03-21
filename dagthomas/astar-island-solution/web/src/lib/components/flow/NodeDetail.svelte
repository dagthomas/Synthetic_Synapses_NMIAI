<script lang="ts">
	import { NODES } from './flow-data';
	import { TIER_COLORS, PIPELINE_STAGES } from './flow-types';
	import type { FlowNodeDef, NodeStatus } from './flow-types';
	import Sparkline from '../charts/Sparkline.svelte';

	let {
		nodeId,
		flowState,
		onclose,
		sparklineData = [],
	}: {
		nodeId: string;
		flowState: { nodeStatuses: Record<string, NodeStatus>; nodeMetrics: Record<string, Record<string, string | number>> };
		onclose: () => void;
		sparklineData?: number[];
	} = $props();

	let node = $derived(NODES.find((n) => n.id === nodeId));
	let status = $derived(node ? flowState.nodeStatuses[node.id] || 'unknown' : 'unknown');
	let metrics = $derived(node ? flowState.nodeMetrics[node.id] || {} : {});
	let tierColor = $derived(node ? TIER_COLORS[node.tier] : 'var(--color-cyber-fg)');

	function statusLabel(s: NodeStatus): string {
		switch (s) {
			case 'active': return 'RUNNING';
			case 'idle': return 'IDLE';
			case 'error': return 'ERROR';
			default: return 'UNKNOWN';
		}
	}

	function statusColor(s: NodeStatus): string {
		switch (s) {
			case 'active': return 'var(--color-score-great)';
			case 'idle': return 'var(--color-cyber-muted)';
			case 'error': return 'var(--color-score-bad)';
			default: return 'var(--color-cyber-muted)';
		}
	}

	// Node descriptions
	const descriptions: Record<string, string> = {
		daemon: 'Central orchestrator that spawns autoloop, monitors for new rounds, triggers exploration/prediction/submission, and downloads calibration data from completed rounds.',
		autoloop: 'Metropolis-Hastings parameter optimizer running ~160k experiments/day. Evaluates each parameter set against leave-one-out cross-validation using the FastHarness (~70ms/eval).',
		api: 'Competition API at api.ainm.no/astar-island. Provides round info, accepts viewport queries (/simulate), and prediction submissions (/submit). 50 queries per round shared across 5 seeds.',
		gemini: 'Autonomous AI researcher using Gemini 3.1 Pro to propose STRUCTURAL algorithm improvements. Each proposal is backtested against historical rounds.',
		multi: 'Dual-model researcher: Gemini Pro writes code, Gemini Flash analyzes results and picks directions. Faster iteration cycle (~15-40s per experiment).',
		exploration: 'Adaptive data collection using 9-viewport grid strategy covering ~97% of the 40x40 map. Queries the API for 15x15 viewport observations across all 5 seeds.',
		prediction: '9-stage production pipeline: feature keys -> calibration prior -> empirical -> FK blending -> global multipliers -> temperature -> smoothing -> structural zeros -> floor.',
		submission: 'Submits the final 40x40x6 probability tensor to the competition API for each seed. Validates probabilities sum to 1.0 and applies minimum floor.',
		fast_harness: 'Pre-computes everything independent of parameters: terrain arrays, feature key grids, idx_grid, coastal/static masks, cluster density. Enables ~70ms evaluation per experiment.',
		cal_model: '4-level hierarchical prior model. Fine (terrain+dist+coastal+forest+port+cluster) -> Coarse (terrain+dist+coastal+port) -> Base (terrain) -> Global.',
		best_params: 'Production parameter file updated by autoloop when improvements found. Contains prior_w, emp_max, exp_damp, T_high, smooth_alpha, floor, and current scores.',
		calibration_data: 'Ground truth data from completed rounds. Contains round_detail.json and analysis_seed_*.json for rounds 1-15. Used for leave-one-out backtesting.',
		autoloop_log: 'Append-only JSONL log of every autoloop experiment. Each entry contains parameters, per-round scores, accepted/rejected status, and timing.',
		research_logs: 'JSONL logs from Gemini and Multi researchers. Contains proposals, hypotheses, backtested scores, and improvement deltas.',
	};
</script>

<div
	class="absolute top-0 right-0 h-full w-[320px] z-50 flex flex-col transition-transform duration-300"
	style="
		background: rgba(18, 18, 30, 0.95);
		backdrop-filter: blur(20px);
		border-left: 1px solid rgba(0, 255, 240, 0.1);
		box-shadow: -10px 0 30px rgba(0, 0, 0, 0.5);
	"
>
	{#if node}
		<!-- Header -->
		<div class="p-4 border-b border-cyber-border">
			<div class="flex items-center justify-between mb-2">
				<div class="flex items-center gap-2">
					<span class="text-lg" style="color: {node.color}; text-shadow: 0 0 8px {node.color};">{node.icon}</span>
					<div>
						<div class="text-[12px] font-bold tracking-wider" style="color: {node.color};">{node.label}</div>
						{#if node.sublabel}
							<div class="text-[13px] text-cyber-muted">{node.sublabel}</div>
						{/if}
					</div>
				</div>
				<button
					class="w-6 h-6 flex items-center justify-center text-cyber-muted hover:text-neon-cyan transition-colors text-sm"
					onclick={onclose}
				>&times;</button>
			</div>

			<!-- Status badge -->
			<div class="flex items-center gap-2 mt-2">
				<div class="w-2 h-2 rounded-full" style="background: {statusColor(status)}; box-shadow: 0 0 6px {statusColor(status)};"></div>
				<span class="text-xs uppercase tracking-wider" style="color: {statusColor(status)};">{statusLabel(status)}</span>
				<span class="text-[13px] text-cyber-muted ml-auto">{node.tier} process</span>
			</div>
		</div>

		<!-- Content -->
		<div class="flex-1 overflow-y-auto p-4 space-y-4">
			<!-- Description -->
			<div>
				<div class="text-[13px] text-cyber-muted uppercase tracking-wider mb-1.5">About</div>
				<p class="text-xs text-cyber-fg leading-relaxed">
					{descriptions[node.id] || 'System component'}
				</p>
			</div>

			<!-- Metrics -->
			{#if Object.keys(metrics).length > 0}
				<div>
					<div class="text-[13px] text-cyber-muted uppercase tracking-wider mb-1.5">Metrics</div>
					<div class="space-y-1.5">
						{#each Object.entries(metrics) as [key, value]}
							<div class="flex items-baseline justify-between py-1 px-2 rounded" style="background: rgba(26, 26, 46, 0.6);">
								<span class="text-[13px] text-cyber-muted uppercase">{key}</span>
								<span class="text-[13px] font-bold text-cyber-fg">{value}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Sparkline for autoloop -->
			{#if node.id === 'autoloop' && sparklineData.length > 1}
				<div>
					<div class="text-[13px] text-cyber-muted uppercase tracking-wider mb-1.5">Convergence</div>
					<div class="flex justify-center p-2 rounded" style="background: rgba(26, 26, 46, 0.6);">
						<Sparkline data={sparklineData} width={240} height={60} color="var(--color-neon-gold)" />
					</div>
				</div>
			{/if}

			<!-- Pipeline stages for prediction -->
			{#if node.id === 'prediction'}
				<div>
					<div class="text-[13px] text-cyber-muted uppercase tracking-wider mb-1.5">Pipeline Stages</div>
					<div class="space-y-1">
						{#each PIPELINE_STAGES as stage}
							<div class="flex items-center gap-2 py-1 px-2 rounded" style="background: rgba(26, 26, 46, 0.6);">
								<span class="text-xs font-bold w-4 text-center" style="color: {stage.color};">{stage.id}</span>
								<span class="text-xs text-cyber-fg">{stage.name}</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Data connections -->
			<div>
				<div class="text-[13px] text-cyber-muted uppercase tracking-wider mb-1.5">Connections</div>
				<div class="text-xs text-cyber-muted space-y-1">
					{#if node.id === 'daemon'}
						<div>&#x2192; Spawns <span class="text-neon-cyan">AUTOLOOP</span></div>
						<div>&#x2192; Triggers <span class="text-neon-gold">EXPLORATION</span></div>
						<div>&#x2192; Monitors <span class="text-neon-cyan">API SERVER</span></div>
						<div>&#x2192; Downloads <span class="text-neon-orange">Calibration Data</span></div>
					{:else if node.id === 'autoloop'}
						<div>&#x2190; Spawned by <span class="text-neon-cyan">DAEMON</span></div>
						<div>&#x2194; Evaluates via <span class="text-neon-orange">FastHarness</span></div>
						<div>&#x2192; Writes <span class="text-neon-orange">best_params.json</span></div>
						<div>&#x2192; Logs to <span class="text-neon-orange">autoloop_log.jsonl</span></div>
					{:else if node.id === 'prediction'}
						<div>&#x2190; Observations from <span class="text-neon-gold">EXPLORATION</span></div>
						<div>&#x2190; Priors from <span class="text-neon-orange">CalibModel</span></div>
						<div>&#x2190; Params from <span class="text-neon-orange">best_params</span></div>
						<div>&#x2192; Tensor to <span class="text-neon-gold">SUBMISSION</span></div>
					{:else if node.id === 'exploration'}
						<div>&#x2190; Triggered by <span class="text-neon-cyan">DAEMON</span></div>
						<div>&#x2194; Queries <span class="text-neon-cyan">API SERVER</span></div>
						<div>&#x2192; Feeds <span class="text-neon-gold">PREDICTION</span></div>
					{:else if node.id === 'gemini' || node.id === 'multi'}
						<div>&#x2192; Backtests via <span class="text-neon-orange">FastHarness</span></div>
						<div>&#x2192; Logs to <span class="text-neon-orange">Research Logs</span></div>
					{:else if node.id === 'fast_harness'}
						<div>&#x2190; Data from <span class="text-neon-orange">CalibModel</span></div>
						<div>&#x2192; Evaluates for <span class="text-neon-cyan">AUTOLOOP</span></div>
						<div>&#x2192; Backtests for <span class="text-neon-magenta">Researchers</span></div>
					{:else}
						<div class="text-cyber-muted/60">See edge connections on the map</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- Footer -->
		<div class="p-3 border-t border-cyber-border text-[13px] text-cyber-muted text-center">
			{node.tier} &middot; {node.id}
		</div>
	{/if}
</div>
