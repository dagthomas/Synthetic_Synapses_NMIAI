<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';

	let stats = $state<any>(null);
	let loading = $state(true);

	onMount(async () => {
		try {
			stats = await admin.stats();
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});
</script>

<h1 class="text-2xl font-bold mb-6">Statistics</h1>

{#if loading}
	<p class="text-gray-500">Loading...</p>
{:else if stats}
	<div class="grid grid-cols-3 gap-4 mb-8">
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Total Teams</div>
			<div class="text-3xl font-bold">{stats.team_count}</div>
		</div>
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Completed Rounds</div>
			<div class="text-3xl font-bold">{stats.completed_rounds} / {stats.total_rounds}</div>
		</div>
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Avg Score</div>
			<div class="text-3xl font-bold">{stats.avg_score != null ? stats.avg_score.toFixed(1) : '-'}</div>
		</div>
	</div>

	<div class="grid grid-cols-2 gap-4 mb-8">
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Total Predictions</div>
			<div class="text-2xl font-bold">{stats.total_predictions}</div>
		</div>
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Total Queries</div>
			<div class="text-2xl font-bold">{stats.total_queries}</div>
		</div>
	</div>

	<!-- Parameter ranges reference -->
	{#if stats.param_ranges}
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<h2 class="text-lg font-semibold mb-3">Parameter Ranges (26 hidden params)</h2>
			<div class="grid grid-cols-2 gap-2 text-sm">
				{#each stats.param_ranges as p}
					<div class="flex items-center gap-3 bg-gray-800 rounded px-3 py-1.5">
						<span class="text-gray-400 w-48 truncate" title={p.description}>{p.name}</span>
						<span class="text-gray-500 font-mono text-xs">[{p.min}, {p.max}]</span>
						<span class="font-mono ml-auto">def: {p.default}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
{/if}
