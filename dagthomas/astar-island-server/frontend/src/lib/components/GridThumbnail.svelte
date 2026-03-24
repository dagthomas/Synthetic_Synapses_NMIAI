<script lang="ts">
	import { CELL_COLORS } from '$lib/types';

	interface Props {
		grid: number[][];
		size?: number;
	}

	let { grid, size = 80 }: Props = $props();
	let canvas: HTMLCanvasElement | undefined = $state();

	$effect(() => {
		if (!canvas || !grid || !grid.length) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const rows = grid.length;
		const cols = grid[0]?.length || 40;
		const cellW = size / cols;
		const cellH = size / rows;
		ctx.clearRect(0, 0, size, size);
		for (let y = 0; y < rows; y++) {
			for (let x = 0; x < cols; x++) {
				ctx.fillStyle = CELL_COLORS[grid[y][x]] || '#333';
				ctx.fillRect(x * cellW, y * cellH, Math.ceil(cellW), Math.ceil(cellH));
			}
		}
	});
</script>

<canvas
	bind:this={canvas}
	width={size}
	height={size}
	class="rounded border border-neon-cyan/15 transition-shadow hover:shadow-[0_0_8px_rgba(0,255,240,0.25)]"
></canvas>
