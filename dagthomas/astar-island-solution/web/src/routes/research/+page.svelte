<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
	import { formatDuration, scoreColor } from '$lib/api';
	import type { ResearchEntry, ProcessInfo } from '$lib/types';

	interface PageData {
		adk: ResearchEntry[];
		gemini: ResearchEntry[];
		multi: ResearchEntry[];
		processes: ProcessInfo[];
	}

	let { data }: { data: PageData } = $props();

	type AgentTab = 'adk' | 'gemini' | 'multi';
	let activeTab = $state<AgentTab>('adk');

	const agents: { key: AgentTab; label: string; color: string }[] = [
		{ key: 'adk', label: 'ADK Agent', color: 'var(--color-neon-orange)' },
		{ key: 'gemini', label: 'Gemini', color: 'var(--color-neon-magenta)' },
		{ key: 'multi', label: 'Multi', color: 'var(--color-neon-gold)' }
	];

	let currentEntries = $derived(data[activeTab] ?? []);

	function getAvgScore(entries: ResearchEntry[]): number {
		const scored = entries.filter((e) => e.scores && Object.keys(e.scores).length);
		if (!scored.length) return 0;
		return (
			scored.reduce((sum: number, e) => {
				const vals = Object.values(e.scores as Record<string, number>);
				return sum + vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
			}, 0) / scored.length
		);
	}
</script>

<div class="space-y-4">
	<h1 class="text-lg text-neon-cyan neon-text tracking-wider">RESEARCH</h1>

	<!-- Agent tabs -->
	<div class="flex gap-2">
		{#each agents as agent}
			{@const proc = data.processes.find((p: ProcessInfo) => p.name === agent.key)}
			<button
				class="flex items-center gap-2 px-3 py-2 text-[13px] rounded border transition-all
					{activeTab === agent.key
					? 'border-current bg-current/10'
					: 'border-cyber-border text-cyber-muted hover:border-cyber-fg/30'}"
				style={activeTab === agent.key ? `color: ${agent.color}` : ''}
				onclick={() => (activeTab = agent.key)}
			>
				{agent.label}
				<span class="text-[13px]">({data[agent.key]?.length ?? 0})</span>
				{#if proc}
					<StatusBadge state={proc.state} />
				{/if}
			</button>
		{/each}
	</div>

	<!-- Stats -->
	<div class="grid grid-cols-3 gap-3">
		<div class="glass p-3 text-center">
			<div class="text-xs text-cyber-muted uppercase mb-1">Experiments</div>
			<div class="text-xl font-bold text-neon-cyan">{currentEntries.length}</div>
		</div>
		<div class="glass p-3 text-center">
			<div class="text-xs text-cyber-muted uppercase mb-1">Avg Score</div>
			<div class="text-xl font-bold" style="color: {scoreColor(getAvgScore(currentEntries))}">
				{getAvgScore(currentEntries).toFixed(2)}
			</div>
		</div>
		<div class="glass p-3 text-center">
			<div class="text-xs text-cyber-muted uppercase mb-1">Errors</div>
			<div class="text-xl font-bold text-score-bad">
				{currentEntries.filter((e: ResearchEntry) => e.error).length}
			</div>
		</div>
	</div>

	<!-- Entries table -->
	<GlassPanel>
		<div class="overflow-y-auto max-h-[500px]">
			<table class="w-full text-[13px]">
				<thead class="sticky top-0 bg-cyber-panel">
					<tr class="text-cyber-muted border-b border-cyber-border">
						<th class="text-left py-1.5 px-2">#</th>
						<th class="text-left py-1.5 px-2">Name</th>
						<th class="text-left py-1.5 px-2">Hypothesis</th>
						<th class="text-right py-1.5 px-2">Score</th>
						<th class="text-right py-1.5 px-2">Improv</th>
						<th class="text-right py-1.5 px-2">Time</th>
						<th class="text-center py-1.5 px-2">Status</th>
					</tr>
				</thead>
				<tbody>
					{#each [...currentEntries].reverse() as entry}
						{@const avgScore = entry.scores
							? Object.values(entry.scores).reduce((a: number, b: number) => a + b, 0) /
								Math.max(Object.values(entry.scores).length, 1)
							: 0}
						<tr
							class="border-b border-cyber-border/20 hover:bg-cyber-panel/50 transition-colors"
						>
							<td class="py-1 px-2 text-cyber-muted">{entry.id}</td>
							<td class="py-1 px-2 truncate max-w-[100px]">{entry.name}</td>
							<td class="py-1 px-2 truncate max-w-[200px] text-cyber-muted">
								{entry.hypothesis || '—'}
							</td>
							<td class="py-1 px-2 text-right font-bold" style="color: {scoreColor(avgScore)}">
								{avgScore ? avgScore.toFixed(2) : '—'}
							</td>
							<td class="py-1 px-2 text-right">
								{#if typeof entry.improvement === 'number' && entry.improvement > 0}
									<span class="text-score-great">+{entry.improvement.toFixed(2)}</span>
								{:else if typeof entry.improvement === 'number' && entry.improvement < 0}
									<span class="text-score-bad">{entry.improvement.toFixed(2)}</span>
								{:else}
									<span class="text-cyber-muted">—</span>
								{/if}
							</td>
							<td class="py-1 px-2 text-right text-cyber-muted">
								{entry.elapsed ? formatDuration(entry.elapsed) : '—'}
							</td>
							<td class="py-1 px-2 text-center">
								{#if entry.error}
									<span class="text-score-bad text-xs">ERR</span>
								{:else}
									<span class="text-score-great text-xs">OK</span>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	</GlassPanel>
</div>
