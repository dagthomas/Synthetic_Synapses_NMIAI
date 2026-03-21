<script lang="ts">
	import type { MyRound } from '$lib/types';
	import { scoreColor, formatTime } from '$lib/api';
	import StatusBadge from '$lib/components/ui/StatusBadge.svelte';

	let { rounds }: { rounds: MyRound[] } = $props();
</script>

<div class="overflow-x-auto">
	<table class="w-full text-[11px]">
		<thead>
			<tr class="text-cyber-muted border-b border-cyber-border">
				<th class="text-left py-2 px-2">#</th>
				<th class="text-left py-2 px-2">Status</th>
				<th class="text-right py-2 px-2">Score</th>
				<th class="text-right py-2 px-2">Rank</th>
				<th class="text-right py-2 px-2">Seeds</th>
				<th class="text-right py-2 px-2">Queries</th>
			</tr>
		</thead>
		<tbody>
			{#each rounds as round}
				<tr class="border-b border-cyber-border/30 hover:bg-cyber-panel/50 transition-colors">
					<td class="py-2 px-2">
						<a href="/rounds/{round.id}" class="text-neon-cyan hover:underline">
							R{round.round_number}
						</a>
					</td>
					<td class="py-2 px-2"><StatusBadge state={round.status} /></td>
					<td class="py-2 px-2 text-right font-bold">
						{#if round.round_score != null}
							<span style="color: {scoreColor(round.round_score)}"
								>{round.round_score.toFixed(2)}</span
							>
						{:else}
							<span class="text-cyber-muted">—</span>
						{/if}
					</td>
					<td class="py-2 px-2 text-right">
						{#if round.rank != null}
							<span class={round.rank <= 3 ? 'text-neon-gold' : ''}>{round.rank}</span>
						{:else}
							<span class="text-cyber-muted">—</span>
						{/if}
					</td>
					<td class="py-2 px-2 text-right text-cyber-muted">
						{round.seeds_submitted}/{round.seeds_count}
					</td>
					<td class="py-2 px-2 text-right text-cyber-muted">
						{round.queries_used}/{round.queries_max}
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
