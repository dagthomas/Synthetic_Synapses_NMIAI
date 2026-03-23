<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';

	let teams = $state<any[]>([]);
	let loading = $state(true);

	onMount(async () => {
		try {
			teams = await admin.teams();
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});
</script>

<h1 class="text-2xl font-bold mb-6">Teams</h1>

{#if loading}
	<p class="text-gray-500">Loading...</p>
{:else}
	<div class="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
		<table class="w-full text-sm">
			<thead class="bg-gray-800">
				<tr>
					<th class="px-4 py-2 text-left">Name</th>
					<th class="px-4 py-2 text-left">Admin</th>
					<th class="px-4 py-2 text-left">Rounds</th>
					<th class="px-4 py-2 text-left">Queries</th>
					<th class="px-4 py-2 text-left">Joined</th>
				</tr>
			</thead>
			<tbody>
				{#each teams as team}
					<tr class="border-t border-gray-800 hover:bg-gray-800/50">
						<td class="px-4 py-2 font-medium">{team.name}</td>
						<td class="px-4 py-2">{team.is_admin ? 'Yes' : 'No'}</td>
						<td class="px-4 py-2">{team.rounds_participated}</td>
						<td class="px-4 py-2">{team.total_queries}</td>
						<td class="px-4 py-2 text-gray-400">{team.created_at?.slice(0, 10)}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}
