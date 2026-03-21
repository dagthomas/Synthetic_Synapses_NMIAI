<script lang="ts">
	let { state }: { state: string } = $props();
	const colors: Record<string, string> = {
		Running: 'bg-score-great/20 text-score-great border-score-great/40',
		Stopped: 'bg-cyber-muted/20 text-cyber-muted border-cyber-muted/40',
		Failed: 'bg-score-bad/20 text-score-bad border-score-bad/40',
		active: 'bg-neon-cyan/20 text-neon-cyan border-neon-cyan/40',
		closed: 'bg-cyber-muted/20 text-cyber-muted border-cyber-muted/40'
	};
	const glows: Record<string, string> = {
		Running: '0 0 8px rgba(0,230,118,0.3), 0 0 16px rgba(0,230,118,0.1)',
		active: '0 0 8px rgba(0,255,240,0.3), 0 0 16px rgba(0,255,240,0.1)',
		Failed: '0 0 8px rgba(229,57,53,0.3)',
		Stopped: 'none',
		closed: 'none'
	};
	let cls = $derived(colors[state] || colors['Stopped']);
	let glow = $derived(glows[state] || 'none');
</script>

<span class="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] uppercase tracking-wider transition-all duration-300 {cls}"
	style="box-shadow: {glow}"
>
	{#if state === 'Running' || state === 'active'}
		<span class="relative inline-block h-1.5 w-1.5">
			<span class="absolute inset-0 rounded-full bg-current animate-ping opacity-40"></span>
			<span class="relative inline-block h-1.5 w-1.5 rounded-full bg-current"></span>
		</span>
	{/if}
	{state}
</span>
