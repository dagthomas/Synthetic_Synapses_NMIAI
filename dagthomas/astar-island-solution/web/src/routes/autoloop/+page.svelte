<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
	import ExperimentTable from '$lib/components/data/ExperimentTable.svelte';
	import Sparkline from '$lib/components/charts/Sparkline.svelte';
	import { fetchAPI, GO_API } from '$lib/api';
	import type { AutoloopEntry, ProcessInfo } from '$lib/types';

	let { data }: { data: { entries: AutoloopEntry[]; processes: ProcessInfo[] } } = $props();

	let entries = $derived(data.entries);
	let autoloopProc = $derived(data.processes.find((p: ProcessInfo) => p.name === 'autoloop'));

	let acceptedScores = $derived(
		entries
			.filter((e: AutoloopEntry) => e.accepted)
			.map((e: AutoloopEntry) => {
				const vals = Object.values(e.scores_quick || {});
				return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
			})
	);

	let totalCount = $derived(entries.length);
	let acceptedCount = $derived(entries.filter((e: AutoloopEntry) => e.accepted).length);
	let bestScore = $derived(
		acceptedScores.length ? Math.max(...acceptedScores) : 0
	);

	async function toggleAutoloop() {
		if (autoloopProc?.state === 'Running') {
			await fetch(`${GO_API}/api/processes/autoloop/stop`, { method: 'POST' });
		} else {
			await fetch(`${GO_API}/api/processes/autoloop/start`, { method: 'POST' });
		}
	}
</script>

<div class="space-y-4">
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-lg text-neon-cyan neon-text tracking-wider">AUTOLOOP</h1>
			<p class="text-xs text-cyber-muted">Autonomous parameter optimization</p>
		</div>
		<div class="flex items-center gap-3">
			<StatusBadge state={autoloopProc?.state ?? 'Stopped'} />
			<button
				class="px-3 py-1.5 text-[13px] rounded border transition-all
					{autoloopProc?.state === 'Running'
					? 'border-score-bad/60 text-score-bad hover:bg-score-bad/10'
					: 'border-score-great/60 text-score-great hover:bg-score-great/10'}"
				onclick={toggleAutoloop}
			>
				{autoloopProc?.state === 'Running' ? 'Stop' : 'Start'}
			</button>
		</div>
	</div>

	<!-- Stats -->
	<div class="grid grid-cols-4 gap-3">
		<div class="glass p-3 text-center">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Total</div>
			<div class="text-2xl font-bold text-neon-cyan">{totalCount}</div>
		</div>
		<div class="glass p-3 text-center">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Accepted</div>
			<div class="text-2xl font-bold text-score-great">{acceptedCount}</div>
		</div>
		<div class="glass p-3 text-center">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Best</div>
			<div class="text-2xl font-bold text-neon-gold">{bestScore.toFixed(2)}</div>
		</div>
		<div class="glass p-3 text-center">
			<div class="text-xs uppercase tracking-wider text-cyber-muted mb-1">Convergence</div>
			{#if acceptedScores.length > 1}
				<Sparkline data={acceptedScores} width={80} height={30} color="var(--color-neon-gold)" />
			{:else}
				<div class="text-cyber-muted">—</div>
			{/if}
		</div>
	</div>

	<!-- Experiment table -->
	<GlassPanel>
		<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Experiments</h2>
		<ExperimentTable {entries} />
	</GlassPanel>
</div>
