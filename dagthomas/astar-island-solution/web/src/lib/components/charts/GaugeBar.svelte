<script lang="ts">
	let {
		value,
		max,
		label = '',
		color = 'var(--color-neon-cyan)'
	}: {
		value: number;
		max: number;
		label?: string;
		color?: string;
	} = $props();

	let pct = $derived(max > 0 ? Math.min((value / max) * 100, 100) : 0);
</script>

<div class="space-y-1">
	{#if label}
		<div class="flex justify-between text-[10px] text-cyber-muted">
			<span>{label}</span>
			<span><span style="color: {color}; text-shadow: 0 0 6px {color}">{value}</span> / {max}</span>
		</div>
	{/if}
	<div class="h-2.5 rounded-full bg-cyber-surface overflow-hidden relative">
		<!-- Track glow -->
		<div class="absolute inset-0 rounded-full" style="box-shadow: inset 0 0 4px rgba(0,0,0,0.5)"></div>
		<!-- Fill -->
		<div
			class="h-full rounded-full transition-all duration-700 ease-out relative overflow-hidden"
			style="width: {pct}%; background: linear-gradient(90deg, {color}, color-mix(in srgb, {color} 80%, white)); box-shadow: 0 0 10px {color}, 0 0 20px color-mix(in srgb, {color} 40%, transparent);"
		>
			<!-- Sweep highlight -->
			<div class="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
				style="animation: gauge-sweep 2s ease-in-out infinite; width: 30%; position: absolute;"
			></div>
		</div>
	</div>
</div>
