<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { admin } from '$lib/api';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import GridViewer from '$lib/components/GridViewer.svelte';

	let detail = $state<any>(null);
	let loading = $state(true);
	let activeSeed = $state(0);

	onMount(async () => {
		try {
			detail = await admin.roundDetail(page.params.id);
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});
</script>

{#if loading}
	<p class="text-gray-500">Loading...</p>
{:else if detail}
	<div class="flex items-center gap-3 mb-6">
		<h1 class="text-2xl font-bold">Round #{detail.round_number}</h1>
		<StatusBadge status={detail.status} />
	</div>

	<!-- Hidden Parameters -->
	<div class="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
		<h2 class="text-lg font-semibold mb-3">Hidden Parameters (26)</h2>
		<div class="grid grid-cols-4 gap-2 text-sm">
			{#each Object.entries(detail.hidden_params) as [key, value]}
				<div class="flex justify-between bg-gray-800 rounded px-3 py-1.5">
					<span class="text-gray-400">{key}</span>
					<span class="font-mono">{typeof value === 'number' ? value.toFixed(3) : value}</span>
				</div>
			{/each}
		</div>
	</div>

	<!-- Seed tabs + grid viewer -->
	<div class="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6">
		<div class="flex gap-2 mb-4">
			{#each detail.seeds as seed, i}
				<button
					onclick={() => (activeSeed = i)}
					class="px-3 py-1 rounded text-sm {activeSeed === i ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}"
				>
					Seed {seed.seed_index} ({seed.settlement_count} setts)
				</button>
			{/each}
		</div>

		{#if detail.seeds[activeSeed]?.initial_grid}
			<GridViewer grid={detail.seeds[activeSeed].initial_grid} />
		{/if}
	</div>

	<!-- Team Scores -->
	{#if detail.team_scores.length > 0}
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<h2 class="text-lg font-semibold mb-3">Team Scores</h2>
			<table class="w-full text-sm">
				<thead class="bg-gray-800">
					<tr>
						<th class="px-3 py-2 text-left">Team</th>
						{#each { length: 5 } as _, i}
							<th class="px-3 py-2 text-center">Seed {i}</th>
						{/each}
						<th class="px-3 py-2 text-center">Avg</th>
						<th class="px-3 py-2 text-center">Queries</th>
					</tr>
				</thead>
				<tbody>
					{#each detail.team_scores as ts}
						<tr class="border-t border-gray-800">
							<td class="px-3 py-2">{ts.team_name}</td>
							{#each ts.seed_scores as s}
								<td class="px-3 py-2 text-center font-mono {s != null && s >= 80 ? 'text-green-400' : s != null && s >= 50 ? 'text-yellow-400' : s != null ? 'text-red-400' : 'text-gray-600'}">
									{s != null ? s.toFixed(1) : '-'}
								</td>
							{/each}
							<td class="px-3 py-2 text-center font-bold">
								{ts.average_score != null ? ts.average_score.toFixed(1) : '-'}
							</td>
							<td class="px-3 py-2 text-center text-gray-400">{ts.queries_used}/50</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
{/if}
