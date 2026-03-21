<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import RoundTable from '$lib/components/data/RoundTable.svelte';
	import type { MyRound } from '$lib/types';
	import { fetchAPI } from '$lib/api';
	import { onMount } from 'svelte';

	let rounds = $state<MyRound[]>([]);

	onMount(async () => {
		try {
			rounds = await fetchAPI<MyRound[]>('/api/my-rounds');
		} catch {
			// offline
		}
	});
</script>

<div class="space-y-4">
	<h1 class="text-lg text-neon-cyan neon-text tracking-wider">ROUNDS</h1>

	<GlassPanel>
		{#if rounds.length}
			<RoundTable {rounds} />
		{:else}
			<div class="text-cyber-muted text-center py-8">Loading rounds...</div>
		{/if}
	</GlassPanel>
</div>
