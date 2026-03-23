<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';
	import { base } from '$app/paths';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	let rounds = $state<any[]>([]);
	let loading = $state(true);
	let creating = $state(false);
	let selectedRegime = $state('random');

	onMount(loadRounds);

	async function loadRounds() {
		loading = true;
		try {
			rounds = await admin.rounds();
		} catch (e) {
			console.error(e);
		}
		loading = false;
	}

	async function createRound() {
		creating = true;
		try {
			await admin.createRound(selectedRegime);
			await loadRounds();
		} catch (e: any) {
			alert(`Error: ${e.message}`);
		}
		creating = false;
	}

	async function activate(id: string) {
		try {
			await admin.activateRound(id);
			await loadRounds();
		} catch (e: any) {
			alert(`Error: ${e.message}`);
		}
	}

	async function score(id: string) {
		try {
			const result = await admin.scoreRound(id);
			alert(`Scored ${result.predictions_scored} predictions`);
			await loadRounds();
		} catch (e: any) {
			alert(`Error: ${e.message}`);
		}
	}
</script>

<div class="flex items-center justify-between mb-6">
	<h1 class="text-2xl font-bold">Rounds</h1>
	<div class="flex items-center gap-3">
		<select bind:value={selectedRegime} class="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm">
			<option value="random">Random</option>
			<option value="collapse">Collapse</option>
			<option value="moderate">Moderate</option>
			<option value="boom">Boom</option>
		</select>
		<button
			onclick={createRound}
			disabled={creating}
			class="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white px-4 py-1.5 rounded text-sm font-medium"
		>
			{creating ? 'Creating...' : 'Create Round'}
		</button>
	</div>
</div>

{#if loading}
	<p class="text-gray-500">Loading...</p>
{:else}
	<div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
		<table class="w-full text-sm">
			<thead class="bg-gray-800">
				<tr>
					<th class="px-4 py-2 text-left">#</th>
					<th class="px-4 py-2 text-left">Status</th>
					<th class="px-4 py-2 text-left">Weight</th>
					<th class="px-4 py-2 text-left">Teams</th>
					<th class="px-4 py-2 text-left">Avg Score</th>
					<th class="px-4 py-2 text-left">Created</th>
					<th class="px-4 py-2 text-left">Actions</th>
				</tr>
			</thead>
			<tbody>
				{#each rounds as round}
					<tr class="border-t border-gray-800 hover:bg-gray-800/50">
						<td class="px-4 py-2">
							<a href="{base}/rounds/{round.id}" class="text-blue-400 hover:underline">
								{round.round_number}
							</a>
						</td>
						<td class="px-4 py-2"><StatusBadge status={round.status} /></td>
						<td class="px-4 py-2">{round.round_weight?.toFixed(2)}</td>
						<td class="px-4 py-2">{round.teams_participated}</td>
						<td class="px-4 py-2">
							{round.avg_score != null ? round.avg_score.toFixed(1) : '-'}
						</td>
						<td class="px-4 py-2 text-gray-400">{round.created_at?.slice(0, 10)}</td>
						<td class="px-4 py-2 flex gap-2">
							{#if round.status === 'pending'}
								<button onclick={() => activate(round.id)} class="text-green-400 hover:underline text-xs">Activate</button>
							{/if}
							{#if round.status === 'active'}
								<button onclick={() => score(round.id)} class="text-yellow-400 hover:underline text-xs">Score</button>
							{/if}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}
