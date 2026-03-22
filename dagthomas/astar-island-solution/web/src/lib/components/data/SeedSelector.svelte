<script lang="ts">
	let {
		count = 5,
		selected = 0,
		onselect
	}: {
		count?: number;
		selected?: number;
		onselect: (idx: number) => void;
	} = $props();
</script>

<div class="flex flex-wrap gap-1">
	{#each Array.from({ length: count }, (_, i) => i) as idx}
		<button
			class="px-2 py-1 text-[11px] rounded border transition-all min-w-0
				{idx === selected
				? 'bg-neon-cyan/20 border-neon-cyan/60 text-neon-cyan shadow-[0_0_6px_var(--color-neon-cyan)]'
				: 'bg-cyber-surface border-cyber-border text-cyber-muted hover:border-cyber-fg/30'}"
			onclick={() => onselect(idx)}
		>
			{idx}
		</button>
	{/each}
</div>
