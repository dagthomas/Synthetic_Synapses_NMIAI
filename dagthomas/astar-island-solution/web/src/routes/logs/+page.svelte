<script lang="ts">
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import { onMount, onDestroy } from 'svelte';
	import { fetchAPI, GO_API } from '$lib/api';

	let { data }: { data: { entries: any[] } } = $props();

	type LogSource = 'autoloop' | 'adk' | 'gemini' | 'multi' | 'history';
	let activeSource = $state<LogSource>('autoloop');
	let entries = $state<any[]>([...(data.entries ?? [])]);
	let eventSource: EventSource | null = null;
	let logContainer: HTMLDivElement;

	const sources: { key: LogSource; label: string }[] = [
		{ key: 'autoloop', label: 'Autoloop' },
		{ key: 'adk', label: 'ADK' },
		{ key: 'gemini', label: 'Gemini' },
		{ key: 'multi', label: 'Multi' },
		{ key: 'history', label: 'History' }
	];

	async function switchSource(source: LogSource) {
		activeSource = source;
		// Load initial entries
		try {
			entries = await fetchAPI<any[]>(`/api/logs/${source}?last=200`);
		} catch {
			entries = [];
		}
		connectSSE(source);
	}

	function connectSSE(source: string) {
		if (eventSource) {
			eventSource.close();
		}
		eventSource = new EventSource(`${GO_API}/api/logs/${source}/stream`);
		eventSource.onmessage = (event) => {
			try {
				const entry = JSON.parse(event.data);
				entries = [...entries, entry].slice(-500);
				// Auto-scroll to bottom
				if (logContainer) {
					requestAnimationFrame(() => {
						logContainer.scrollTop = logContainer.scrollHeight;
					});
				}
			} catch {
				// ignore parse errors
			}
		};
	}

	onMount(() => {
		connectSSE(activeSource);
	});

	onDestroy(() => {
		if (eventSource) eventSource.close();
	});

	function formatEntry(entry: any): string {
		try {
			return JSON.stringify(entry, null, 2);
		} catch {
			return String(entry);
		}
	}

	function entryPreview(entry: any): string {
		const id = entry.id ?? '';
		const name = entry.name ?? '';
		const ts = entry.timestamp ?? '';
		const shortTs = ts ? ts.split('T')[1]?.split('.')[0] ?? ts : '';
		return `[${shortTs}] #${id} ${name}`;
	}
</script>

<div class="space-y-4">
	<h1 class="text-lg text-neon-cyan neon-text tracking-wider">LOGS</h1>

	<!-- Source tabs -->
	<div class="flex gap-2">
		{#each sources as source}
			<button
				class="px-3 py-1.5 text-[13px] rounded border transition-all
					{activeSource === source.key
					? 'bg-neon-cyan/10 border-neon-cyan/60 text-neon-cyan'
					: 'border-cyber-border text-cyber-muted hover:border-cyber-fg/30'}"
				onclick={() => switchSource(source.key)}
			>
				{source.label}
			</button>
		{/each}
	</div>

	<!-- Log viewer -->
	<GlassPanel class="!p-0">
		<div bind:this={logContainer} class="h-[calc(100vh-220px)] overflow-y-auto p-3 font-mono">
			{#each entries as entry, i}
				<div
					class="py-1 px-2 text-[13px] border-b border-cyber-border/10 hover:bg-cyber-panel/30 transition-colors leading-relaxed"
				>
					<span class="text-cyber-muted">{entryPreview(entry)}</span>
					{#if entry.accepted !== undefined}
						{#if entry.accepted}
							<span class="text-score-great ml-2">ACCEPTED</span>
						{/if}
					{/if}
					{#if entry.error}
						<span class="text-score-bad ml-2">ERR</span>
					{/if}
				</div>
			{/each}
			{#if !entries.length}
				<div class="text-cyber-muted text-center py-8">No log entries</div>
			{/if}
		</div>
	</GlassPanel>
</div>
