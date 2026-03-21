<script lang="ts">
	import { scoreColor } from '$lib/api';

	let {
		label,
		score,
		subtitle = ''
	}: {
		label: string;
		score: number | null;
		subtitle?: string;
	} = $props();

	let glowColor = $derived(score != null ? scoreColor(score) : 'transparent');
</script>

<div class="glass glass-border-glow p-3 text-center relative overflow-hidden score-shimmer group"
	style="--glow: {glowColor}"
>
	<!-- Top accent line -->
	<div class="absolute top-0 left-2 right-2 h-px" style="background: linear-gradient(90deg, transparent, {glowColor}, transparent); opacity: 0.5"></div>

	<div class="text-[10px] uppercase tracking-wider text-cyber-muted mb-1">{label}</div>
	{#if score != null}
		<div class="text-2xl font-bold transition-all duration-300" style="color: {glowColor}; text-shadow: 0 0 12px {glowColor}, 0 0 24px color-mix(in srgb, {glowColor} 30%, transparent)">
			{score.toFixed(2)}
		</div>
	{:else}
		<div class="text-2xl text-cyber-muted">—</div>
	{/if}
	{#if subtitle}
		<div class="text-[10px] text-cyber-muted mt-0.5">{subtitle}</div>
	{/if}

	<!-- Corner accents -->
	<div class="absolute top-0 left-0 w-2 h-2 border-t border-l border-neon-cyan/20"></div>
	<div class="absolute top-0 right-0 w-2 h-2 border-t border-r border-neon-cyan/20"></div>
	<div class="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-neon-cyan/20"></div>
	<div class="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-neon-cyan/20"></div>
</div>
