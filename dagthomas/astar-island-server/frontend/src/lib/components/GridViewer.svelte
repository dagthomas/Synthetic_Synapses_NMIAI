<script lang="ts">
	import { CELL_COLORS, CELL_NAMES } from '$lib/types';

	interface Props {
		grid: number[][];
		cellSize?: number;
	}

	let { grid, cellSize = 10 }: Props = $props();
	let tooltip = $state('');
	let tooltipX = $state(0);
	let tooltipY = $state(0);
	let showTooltip = $state(false);

	function handleHover(e: MouseEvent, y: number, x: number) {
		const cell = grid[y]?.[x];
		tooltip = `(${x}, ${y}) ${CELL_NAMES[cell] || 'Unknown'}`;
		tooltipX = e.clientX + 10;
		tooltipY = e.clientY + 10;
		showTooltip = true;
	}
</script>

<div class="glass glass-glow p-3">
	<div class="relative inline-block overflow-auto max-h-[450px] rounded border border-neon-cyan/15">
		<div
			class="grid"
			style="grid-template-columns: repeat({grid[0]?.length || 40}, {cellSize}px); gap: 0;"
		>
			{#each grid as row, y}
				{#each row as cell, x}
					<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
						class="cursor-crosshair"
						role="gridcell"
						style="width: {cellSize}px; height: {cellSize}px; background: {CELL_COLORS[cell] || '#333'};"
						onmouseenter={(e) => handleHover(e, y, x)}
						onmouseleave={() => (showTooltip = false)}
					></div>
				{/each}
			{/each}
		</div>
	</div>
</div>

{#if showTooltip}
	<div
		class="fixed z-50 glass text-neon-cyan text-xs px-3 py-1.5 rounded pointer-events-none border-neon-cyan/30 shadow-[0_0_10px_rgba(0,255,240,0.15)]"
		style="left: {tooltipX}px; top: {tooltipY}px;"
	>
		{tooltip}
	</div>
{/if}
