<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';
	import { base } from '$app/paths';
	import StatusBadge from '$lib/components/StatusBadge.svelte';
	import GridThumbnail from '$lib/components/GridThumbnail.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Table from '$lib/components/ui/table';
	import { scoreColor } from '$lib/chart-theme';

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
	<h1 class="text-2xl font-bold text-neon-cyan neon-text tracking-wider uppercase">PLANETS</h1>
	<div class="flex items-center gap-3">
		<select
			bind:value={selectedRegime}
			class="bg-cyber-surface border border-cyber-border rounded px-3 py-1.5 text-sm text-cyber-fg focus:border-neon-cyan focus:outline-none focus:ring-1 focus:ring-neon-cyan/30"
		>
			<option value="random">Unknown</option>
			<option value="collapse">Barren</option>
			<option value="moderate">Temperate</option>
			<option value="boom">Volatile</option>
		</select>
		<Button
			onclick={createRound}
			disabled={creating}
			class="bg-neon-cyan/10 border border-neon-cyan/40 text-neon-cyan hover:bg-neon-cyan/20 hover:border-neon-cyan/60 hover:shadow-[0_0_15px_rgba(0,255,240,0.2)] disabled:opacity-50 transition-all text-xs tracking-wider uppercase"
		>
			{creating ? 'Generating...' : 'Generate Planet'}
		</Button>
	</div>
</div>

{#if loading}
	<p class="text-cyber-muted animate-pulse-glow">Loading...</p>
{:else}
	<div class="glass glass-glow overflow-hidden">
		<Table.Root>
			<Table.Header>
				<Table.Row class="border-b border-cyber-border bg-cyber-surface/50 hover:bg-cyber-surface/50">
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Map</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">#</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Status</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Weight</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Corps</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Avg Score</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Created</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Actions</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each rounds as round}
					<Table.Row class="border-b border-cyber-border/50 hover:bg-neon-cyan/5 transition-colors">
						<Table.Cell class="py-1">
							{#if round.first_seed_grid}
								<a href="{base}/rounds/{round.id}">
									<GridThumbnail grid={round.first_seed_grid} size={48} />
								</a>
							{:else}
								<div class="w-12 h-12 rounded border border-cyber-border/30 bg-cyber-surface/50 flex items-center justify-center text-cyber-muted text-[10px]">?</div>
							{/if}
						</Table.Cell>
						<Table.Cell>
							<a href="{base}/rounds/{round.id}" class="text-neon-cyan hover:text-neon-cyan/80 hover:underline font-bold">
								{round.round_number}
							</a>
						</Table.Cell>
						<Table.Cell><StatusBadge status={round.status} /></Table.Cell>
						<Table.Cell class="text-cyber-fg font-mono text-xs">{round.round_weight?.toFixed(2)}</Table.Cell>
						<Table.Cell class="text-cyber-fg">{round.teams_participated}</Table.Cell>
						<Table.Cell>
							{#if round.avg_score != null}
								<span class="font-mono" style="color: {scoreColor(round.avg_score)}">{round.avg_score.toFixed(1)}</span>
							{:else}
								<span class="text-cyber-muted">—</span>
							{/if}
						</Table.Cell>
						<Table.Cell class="text-cyber-muted text-xs">{round.created_at?.slice(0, 10)}</Table.Cell>
						<Table.Cell>
							<div class="flex gap-2">
								{#if round.status === 'pending'}
									<button
										onclick={() => activate(round.id)}
										class="text-score-great hover:text-score-great/80 text-xs tracking-wider uppercase hover:underline"
									>
										Activate
									</button>
								{/if}
								{#if round.status === 'active'}
									<button
										onclick={() => score(round.id)}
										class="text-neon-gold hover:text-neon-gold/80 text-xs tracking-wider uppercase hover:underline"
									>
										Score
									</button>
								{/if}
							</div>
						</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>
		</Table.Root>
	</div>
{/if}
