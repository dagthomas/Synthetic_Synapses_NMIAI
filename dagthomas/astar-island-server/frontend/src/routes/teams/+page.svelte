<script lang="ts">
	import { onMount } from 'svelte';
	import { admin } from '$lib/api';
	import { base } from '$app/paths';
	import { goto } from '$app/navigation';
	import * as Table from '$lib/components/ui/table';
	import { Badge } from '$lib/components/ui/badge';

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

<h1 class="text-2xl font-bold mb-6 text-neon-cyan neon-text tracking-wider uppercase">CORPORATIONS</h1>

{#if loading}
	<p class="text-cyber-muted animate-pulse-glow">Loading...</p>
{:else}
	<div class="glass glass-glow overflow-hidden">
		<Table.Root>
			<Table.Header>
				<Table.Row class="border-b border-cyber-border bg-cyber-surface/50 hover:bg-cyber-surface/50">
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Corporation</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Admin</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Planets</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Scans</Table.Head>
					<Table.Head class="text-neon-cyan/70 text-[10px] tracking-widest uppercase">Registered</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each teams as team}
					<Table.Row
						class="border-b border-cyber-border/50 hover:bg-neon-cyan/5 transition-colors cursor-pointer"
						onclick={() => goto(`${base}/teams/${team.id}`)}
					>
						<Table.Cell class="font-medium text-cyber-fg hover:text-neon-cyan transition-colors">{team.name}</Table.Cell>
						<Table.Cell>
							{#if team.is_admin}
								<Badge variant="outline" class="text-[10px] tracking-wider uppercase bg-neon-magenta/15 text-neon-magenta border-neon-magenta/30">
									Admin
								</Badge>
							{:else}
								<span class="text-cyber-muted text-xs">--</span>
							{/if}
						</Table.Cell>
						<Table.Cell class="text-neon-gold">{team.rounds_participated}</Table.Cell>
						<Table.Cell class="text-cyber-fg">{team.total_queries}</Table.Cell>
						<Table.Cell class="text-cyber-muted">{team.created_at?.slice(0, 10)}</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>
		</Table.Root>
	</div>
{/if}
