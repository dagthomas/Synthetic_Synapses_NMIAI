<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { Budget, ProcessInfo } from '$lib/types';
	import { simulateTrigger } from '$lib/stores/simulate';

	let {
		budget,
		processes = [],
		connected = true
	}: {
		budget: Budget | null;
		processes?: ProcessInfo[];
		connected?: boolean;
	} = $props();

	let runningCount = $derived(processes.filter((p) => p.state === 'Running').length);
	let showSeconds = $state(false);
	let now = $state(new Date());
	let clockInterval: ReturnType<typeof setInterval> | undefined;

	onMount(() => {
		clockInterval = setInterval(() => { now = new Date(); }, 1000);
	});
	onDestroy(() => { if (clockInterval) clearInterval(clockInterval); });

	let timeStr = $derived(
		now.toLocaleTimeString('nb-NO', {
			hour: '2-digit',
			minute: '2-digit',
			...(showSeconds ? { second: '2-digit' } : {})
		})
	);
</script>

<div
	class="flex items-center gap-4 px-4 py-1.5 bg-cyber-surface/90 border-t border-cyber-border text-[10px] backdrop-blur-sm relative"
>
	<!-- Top edge glow line -->
	<div class="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neon-cyan/30 to-transparent"></div>

	<!-- Connection -->
	<div class="flex items-center gap-1.5">
		<span class="h-1.5 w-1.5 rounded-full {connected ? 'bg-score-great animate-breathe' : 'bg-score-bad'}"
			style={connected ? 'box-shadow: 0 0 6px rgba(0,230,118,0.5)' : 'box-shadow: 0 0 6px rgba(229,57,53,0.5)'}
		></span>
		<span class="text-cyber-muted">{connected ? 'Connected' : 'Offline'}</span>
	</div>

	<!-- Budget -->
	{#if budget}
		<div class="flex items-center gap-1.5 text-cyber-muted">
			<span>Queries:</span>
			<span class="text-neon-cyan" style="text-shadow: 0 0 6px rgba(0,255,240,0.3)">{budget.queries_used}</span>
			<span>/</span>
			<span>{budget.queries_max}</span>
		</div>
	{/if}

	<!-- Processes -->
	<div class="flex items-center gap-1.5 text-cyber-muted">
		<span>Processes:</span>
		<span class={runningCount > 0 ? 'text-score-great' : 'text-cyber-muted'}
			style={runningCount > 0 ? 'text-shadow: 0 0 6px rgba(0,230,118,0.4)' : ''}
		>{runningCount}</span>
		<span>running</span>
	</div>

	<!-- Spacer -->
	<div class="flex-1"></div>

	<!-- Timestamp (click to trigger simulation on flow page) -->
	<button
		class="text-cyber-muted hover:text-neon-cyan transition-colors tabular-nums flex items-center gap-1.5"
		onclick={() => { showSeconds = !showSeconds; simulateTrigger.update(n => n + 1); }}
		title="Toggle seconds"
	>
		{#if showSeconds}
			<span class="w-1 h-1 rounded-full bg-neon-cyan animate-pulse-glow"></span>
		{/if}
		{timeStr}
	</button>
</div>
