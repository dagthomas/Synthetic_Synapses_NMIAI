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

<div class="relative inline-block overflow-auto max-h-[450px] border border-gray-700 rounded">
	<div
		class="grid"
		style="grid-template-columns: repeat({grid[0]?.length || 40}, {cellSize}px); gap: 0;"
	>
		{#each grid as row, y}
			{#each row as cell, x}
				<div
					class="cursor-crosshair"
					style="width: {cellSize}px; height: {cellSize}px; background: {CELL_COLORS[cell] || '#333'};"
					onmouseenter={(e) => handleHover(e, y, x)}
					onmouseleave={() => (showTooltip = false)}
				></div>
			{/each}
		{/each}
	</div>
</div>

{#if showTooltip}
	<div
		class="fixed z-50 bg-gray-900 text-white text-xs px-2 py-1 rounded pointer-events-none"
		style="left: {tooltipX}px; top: {tooltipY}px;"
	>
		{tooltip}
	</div>
{/if}
