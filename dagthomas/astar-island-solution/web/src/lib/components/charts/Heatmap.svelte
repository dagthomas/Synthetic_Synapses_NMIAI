<script lang="ts">
	import { onMount } from 'svelte';

	let {
		data,
		size = 400
	}: {
		data: number[][];
		size?: number;
	} = $props();

	let canvas: HTMLCanvasElement;

	function draw() {
		if (!canvas || !data?.length) return;
		const ctx = canvas.getContext('2d')!;
		const rows = data.length;
		const cols = data[0]?.length || 0;
		if (!cols) return;

		const cellW = size / cols;
		const cellH = size / rows;

		ctx.clearRect(0, 0, size, size);

		for (let y = 0; y < rows; y++) {
			for (let x = 0; x < cols; x++) {
				const v = data[y][x] ?? 0;
				const norm = Math.max(0, Math.min(v / 100, 1));
				// Red to green gradient
				const r = Math.round(255 * (1 - norm));
				const g = Math.round(255 * norm);
				ctx.fillStyle = `rgb(${r}, ${g}, 40)`;
				ctx.fillRect(x * cellW, y * cellH, cellW + 0.5, cellH + 0.5);
			}
		}
	}

	onMount(() => draw());

	$effect(() => {
		data;
		draw();
	});
</script>

<canvas bind:this={canvas} width={size} height={size} class="rounded border border-cyber-border"></canvas>
