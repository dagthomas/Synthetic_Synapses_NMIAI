<script lang="ts">
	import type { LeaderboardEntry } from '$lib/types';
	import { scoreColor } from '$lib/api';

	let { entries }: { entries: LeaderboardEntry[] } = $props();

	const medals = ['', '\u{1F947}', '\u{1F948}', '\u{1F949}'];
</script>

<div class="overflow-x-auto">
	<table class="w-full text-[11px]">
		<thead>
			<tr class="text-cyber-muted border-b border-cyber-border">
				<th class="text-left py-2 px-2">Rank</th>
				<th class="text-left py-2 px-2">Team</th>
				<th class="text-right py-2 px-2">Score</th>
				<th class="text-right py-2 px-2">Rounds</th>
				<th class="text-right py-2 px-2">Streak</th>
			</tr>
		</thead>
		<tbody>
			{#each entries as entry}
				<tr class="border-b border-cyber-border/30 hover:bg-cyber-panel/50 transition-colors">
					<td class="py-1.5 px-2">
						{#if entry.rank <= 3}
							<span class="text-neon-gold">{medals[entry.rank]} {entry.rank}</span>
						{:else}
							{entry.rank}
						{/if}
					</td>
					<td class="py-1.5 px-2 truncate max-w-[120px]">{entry.team_name}</td>
					<td class="py-1.5 px-2 text-right font-bold" style="color: {scoreColor(entry.weighted_score)}">
						{entry.weighted_score.toFixed(2)}
					</td>
					<td class="py-1.5 px-2 text-right text-cyber-muted">{entry.rounds_participated}</td>
					<td class="py-1.5 px-2 text-right text-neon-orange">{entry.hot_streak_score.toFixed(1)}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
