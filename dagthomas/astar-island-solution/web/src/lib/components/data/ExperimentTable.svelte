<script lang="ts">
	import type { AutoloopEntry } from '$lib/types';
	import { scoreColor, formatDuration } from '$lib/api';

	let { entries }: { entries: AutoloopEntry[] } = $props();
</script>

<div class="overflow-y-auto max-h-[400px]">
	<table class="w-full text-[13px]">
		<thead class="sticky top-0 bg-cyber-panel">
			<tr class="text-cyber-muted border-b border-cyber-border">
				<th class="text-left py-1.5 px-2">#</th>
				<th class="text-left py-1.5 px-2">Name</th>
				<th class="text-right py-1.5 px-2">Score</th>
				<th class="text-right py-1.5 px-2">Baseline</th>
				<th class="text-center py-1.5 px-2">Status</th>
				<th class="text-right py-1.5 px-2">Time</th>
			</tr>
		</thead>
		<tbody>
			{#each entries.slice().reverse() as entry}
				{@const avgScore = Object.values(entry.scores_quick || {}).length
					? Object.values(entry.scores_quick).reduce((a, b) => a + b, 0) /
						Object.values(entry.scores_quick).length
					: 0}
				<tr class="border-b border-cyber-border/20 hover:bg-cyber-panel/50 transition-colors">
					<td class="py-1 px-2 text-cyber-muted">{entry.id}</td>
					<td class="py-1 px-2 truncate max-w-[150px]">{entry.name}</td>
					<td class="py-1 px-2 text-right font-bold" style="color: {scoreColor(avgScore)}">
						{avgScore.toFixed(2)}
					</td>
					<td class="py-1 px-2 text-right text-cyber-muted">
						{entry.baseline_avg.toFixed(2)}
					</td>
					<td class="py-1 px-2 text-center">
						{#if entry.accepted}
							<span class="text-score-great text-xs">ACCEPTED</span>
						{:else}
							<span class="text-cyber-muted text-xs">rejected</span>
						{/if}
					</td>
					<td class="py-1 px-2 text-right text-cyber-muted">
						{formatDuration(entry.elapsed)}
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
