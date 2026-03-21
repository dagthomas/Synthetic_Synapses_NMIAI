<script lang="ts">
	import { scoreColor } from '$lib/api';

	let {
		items,
		maxValue
	}: {
		items: { label: string; value: number }[];
		maxValue?: number;
	} = $props();

	let max = $derived(maxValue ?? Math.max(...items.map((i) => i.value), 1));
</script>

<div class="space-y-1.5">
	{#each items as item}
		{@const pct = (item.value / max) * 100}
		<div class="flex items-center gap-2 text-[11px]">
			<span class="w-20 truncate text-cyber-muted">{item.label}</span>
			<div class="flex-1 h-3 bg-cyber-surface rounded overflow-hidden">
				<div
					class="h-full rounded transition-all duration-300"
					style="width: {pct}%; background: {scoreColor(item.value)};"
				></div>
			</div>
			<span class="w-12 text-right" style="color: {scoreColor(item.value)}"
				>{item.value.toFixed(1)}</span
			>
		</div>
	{/each}
</div>
