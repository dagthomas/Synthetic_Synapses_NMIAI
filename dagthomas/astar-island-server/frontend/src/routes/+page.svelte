<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';
	import StatusBadge from '$lib/components/StatusBadge.svelte';

	let dashboard = $state<any>(null);
	let loading = $state(true);

	onMount(async () => {
		try {
			dashboard = await admin.dashboard();
		} catch (e) {
			console.error(e);
		}
		loading = false;
	});
</script>

<h1 class="text-2xl font-bold mb-6">Dashboard</h1>

{#if loading}
	<p class="text-gray-500">Loading...</p>
{:else if dashboard}
	<div class="grid grid-cols-4 gap-4 mb-8">
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Teams</div>
			<div class="text-3xl font-bold">{dashboard.team_count}</div>
		</div>
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Rounds</div>
			<div class="text-3xl font-bold">{dashboard.total_rounds}</div>
		</div>
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Predictions</div>
			<div class="text-3xl font-bold">{dashboard.total_predictions}</div>
		</div>
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
			<div class="text-sm text-gray-400">Active Round</div>
			<div class="text-3xl font-bold">
				{#if dashboard.active_round}
					#{dashboard.active_round.round_number}
				{:else}
					None
				{/if}
			</div>
		</div>
	</div>

	{#if dashboard.active_round}
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-6">
			<div class="flex items-center gap-3 mb-3">
				<h2 class="text-lg font-semibold">Round #{dashboard.active_round.round_number}</h2>
				<StatusBadge status={dashboard.active_round.status} />
			</div>
			<div class="text-sm text-gray-400">
				Closes: {dashboard.active_round.closes_at || 'N/A'}
			</div>
		</div>
	{/if}
{/if}
