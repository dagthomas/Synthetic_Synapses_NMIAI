<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { Chart, registerables, type ChartConfiguration } from 'chart.js';
	import { cyberChartDefaults } from '$lib/chart-theme';

	Chart.register(...registerables);

	interface Props {
		type: ChartConfiguration['type'];
		data: ChartConfiguration['data'];
		options?: ChartConfiguration['options'];
		height?: string;
		title?: string;
	}

	let { type, data, options = {}, height = '300px', title = '' }: Props = $props();
	let canvasEl: HTMLCanvasElement | undefined = $state();
	let chartInstance: Chart | null = null;
	let mounted = $state(false);

	function deepMerge(target: any, source: any): any {
		const out = { ...target };
		for (const key of Object.keys(source)) {
			if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
				out[key] = deepMerge(out[key] || {}, source[key]);
			} else {
				out[key] = source[key];
			}
		}
		return out;
	}

	function createChart() {
		if (!canvasEl || !data) return;

		if (chartInstance) {
			chartInstance.destroy();
			chartInstance = null;
		}

		const merged = deepMerge(cyberChartDefaults, options || {});
		chartInstance = new Chart(canvasEl, {
			type: type as any,
			data: JSON.parse(JSON.stringify(data)),
			options: merged
		});
	}

	onMount(() => {
		mounted = true;
		createChart();
	});

	onDestroy(() => {
		if (chartInstance) {
			chartInstance.destroy();
			chartInstance = null;
		}
	});

	// Re-create chart when data changes
	$effect(() => {
		// Read reactive deps
		const _d = data;
		const _t = type;
		const _o = options;
		if (mounted && canvasEl && _d) {
			// Use microtask to avoid Svelte effect ordering issues
			queueMicrotask(() => createChart());
		}
	});
</script>

<div class="glass glass-glow relative w-full rounded-lg border border-neon-cyan/15 p-4">
	<div style="position: relative; height: {height}; width: 100%;">
		<canvas bind:this={canvasEl}></canvas>
	</div>
</div>
