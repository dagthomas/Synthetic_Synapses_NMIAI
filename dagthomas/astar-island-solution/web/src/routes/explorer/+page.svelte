<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import SeedSelector from '$lib/components/data/SeedSelector.svelte';
	import GlassPanel from '$lib/components/ui/GlassPanel.svelte';
	import IslandMap from '$lib/components/map/IslandMap.svelte';
	import TimeSlider from '$lib/components/map/TimeSlider.svelte';
	import { TerrainNames, TerrainColors } from '$lib/types';
	import type { GalleryEntry, GeneratedImage, RoundDetail } from '$lib/types';
	import { fetchAPI, postAPI, deleteAPI, GO_API } from '$lib/api';

	export const ssr = false;

	let { data } = $props();

	let selectedSeed = $state(0);
	let showViewport = $state(true);
	let showScores = $state(false);
	let viewportX = $state(12);
	let viewportY = $state(12);
	let timeOfDay = $state(getCurrentHour());
	let autoTime = $state(true);
	let clockInterval: ReturnType<typeof setInterval> | undefined;

	// Round switching state
	let rounds = $state(data.rounds);
	let selectedRoundId = $state(data.activeRoundId);
	let currentDetail: RoundDetail | null = $state(data.detail);
	let loadingRound = $state(false);

	// Imagen state
	let captureFn: (() => string) | null = $state(null);
	let generating = $state(false);
	let generationError = $state('');
	let gallery: GalleryEntry[] = $state([]);
	let showGallery = $state(true);
	let expandedImage: string | null = $state(null);
	let modalImage: { url: string; prompt: string } | null = $state(null);

	// Generate All state
	let generatingAll = $state(false);
	let generateAllProgress = $state({ current: 0, total: 0 });
	let generateAllErrors: string[] = $state([]);
	let generateAllAbort = $state(false);

	let grid = $derived(currentDetail?.initial_states?.[selectedSeed]?.grid ?? null);
	let settlements = $derived(currentDetail?.initial_states?.[selectedSeed]?.settlements ?? []);

	function getCurrentHour(): number {
		const now = new Date();
		return now.getHours() + now.getMinutes() / 60;
	}

	async function selectRound(roundId: string) {
		if (roundId === selectedRoundId && currentDetail) return;
		loadingRound = true;
		generationError = '';
		try {
			const detail = await fetchAPI<RoundDetail>(`/api/rounds/${roundId}`);
			currentDetail = detail;
			selectedRoundId = roundId;
		} catch (e: any) {
			generationError = `Failed to load round: ${e.message}`;
		} finally {
			loadingRound = false;
		}
	}

	function waitForRender(): Promise<void> {
		return new Promise((resolve) => {
			requestAnimationFrame(() => {
				requestAnimationFrame(() => {
					resolve();
				});
			});
		});
	}

	async function handleGenerateAll() {
		if (!captureFn || generatingAll) return;

		generatingAll = true;
		generateAllErrors = [];
		generateAllAbort = false;

		const sortedRounds = [...rounds].sort((a, b) => a.round_number - b.round_number);
		generateAllProgress = { current: 0, total: sortedRounds.length };

		for (let i = 0; i < sortedRounds.length; i++) {
			if (generateAllAbort) break;

			const round = sortedRounds[i];
			generateAllProgress = { current: i + 1, total: sortedRounds.length };

			try {
				const detail = await fetchAPI<RoundDetail>(`/api/rounds/${round.id}`);
				currentDetail = detail;
				selectedRoundId = round.id;

				await waitForRender();

				const dataUrl = captureFn();
				const base64 = dataUrl.replace(/^data:image\/png;base64,/, '');
				await postAPI<GeneratedImage>('/api/imagen/generate', { image: base64 });
			} catch (e: any) {
				generateAllErrors.push(`Round ${round.round_number}: ${e.message ?? 'failed'}`);
			}
		}

		await loadGallery();
		generatingAll = false;
	}

	function cancelGenerateAll() {
		generateAllAbort = true;
	}

	async function loadGallery() {
		try {
			gallery = await fetchAPI<GalleryEntry[]>('/api/imagen/gallery');
		} catch {
			// gallery not available yet
		}
	}

	async function handleGenerate() {
		if (!captureFn || generating) return;
		generating = true;
		generationError = '';
		try {
			const dataUrl = captureFn();
			const base64 = dataUrl.replace(/^data:image\/png;base64,/, '');
			const result = await postAPI<GeneratedImage>('/api/imagen/generate', { image: base64 });
			await loadGallery();
			modalImage = {
				url: `${GO_API}/api/imagen/images/${result.filename}`,
				prompt: result.prompt
			};
		} catch (e: any) {
			generationError = e.message ?? 'Generation failed';
		} finally {
			generating = false;
		}
	}

	async function handleDelete(entry: GalleryEntry) {
		try {
			await deleteAPI(`/api/imagen/images/${entry.filename}`);
			gallery = gallery.filter((g) => g.id !== entry.id);
		} catch {
			// ignore
		}
	}

	onMount(() => {
		clockInterval = setInterval(() => {
			if (autoTime) {
				timeOfDay = getCurrentHour();
			}
		}, 30000);
		loadGallery();
	});

	onDestroy(() => {
		if (clockInterval) clearInterval(clockInterval);
	});

	// Terrain breakdown
	let terrainCounts = $derived.by(() => {
		if (!grid) return {};
		const counts: Record<number, number> = {};
		for (const row of grid) {
			for (const cell of row) {
				counts[cell] = (counts[cell] || 0) + 1;
			}
		}
		return counts;
	});

	let totalCells = $derived(grid ? grid.length * (grid[0]?.length ?? 0) : 0);
</script>

<div class="flex gap-4 h-[calc(100vh-80px)]">
	<!-- 3D Map area -->
	<div class="flex-1 relative">
		{#if grid}
			<IslandMap
				{grid}
				{settlements}
				{showViewport}
				{showScores}
				{viewportX}
				{viewportY}
				{timeOfDay}
				freezeCamera={generatingAll}
				onCaptureFn={(fn) => (captureFn = fn)}
			/>
		{:else}
			<div class="flex items-center justify-center h-full text-cyber-muted">
				No round data available. Start the Go API.
			</div>
		{/if}
	</div>

	<!-- Controls panel -->
	<div class="w-64 space-y-3 overflow-y-auto shrink-0">
		<GlassPanel>
			<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Controls</h2>

			{#if currentDetail}
				<div class="space-y-3">
					<!-- Round selector -->
					<div>
						<div class="text-[10px] text-cyber-muted mb-1">Round</div>
						<select
							class="w-full px-2 py-1 text-[11px] rounded border border-cyber-border
								bg-cyber-surface text-cyber-fg
								focus:border-neon-cyan/60 focus:outline-none"
							value={selectedRoundId}
							onchange={(e) => selectRound(e.currentTarget.value)}
							disabled={loadingRound || generatingAll}
						>
							{#each [...rounds].sort((a, b) => a.round_number - b.round_number) as round}
								<option value={round.id}>
									Round {round.round_number} ({round.status})
								</option>
							{/each}
						</select>
						{#if loadingRound}
							<div class="text-[10px] text-neon-cyan mt-1 animate-pulse">Loading...</div>
						{/if}
					</div>

					<!-- Seed selector -->
					<div>
						<div class="text-[10px] text-cyber-muted mb-1">Seed</div>
						<SeedSelector
							count={currentDetail.initial_states?.length ?? 5}
							selected={selectedSeed}
							onselect={(i) => (selectedSeed = i)}
						/>
					</div>

					<!-- Time of day slider -->
					<TimeSlider bind:timeOfDay bind:autoMode={autoTime} />

					<!-- Viewport position -->
					<div>
						<div class="text-[10px] text-cyber-muted mb-1">Viewport Position</div>
						<div class="flex items-center gap-2">
							<button
								class="px-2 py-1 text-xs rounded border border-cyber-border hover:border-neon-cyan/40"
								onclick={() => (viewportX = Math.max(0, viewportX - 1))}>&larr;</button
							>
							<button
								class="px-2 py-1 text-xs rounded border border-cyber-border hover:border-neon-cyan/40"
								onclick={() => (viewportY = Math.max(0, viewportY - 1))}>&uarr;</button
							>
							<button
								class="px-2 py-1 text-xs rounded border border-cyber-border hover:border-neon-cyan/40"
								onclick={() => (viewportY = Math.min(25, viewportY + 1))}>&darr;</button
							>
							<button
								class="px-2 py-1 text-xs rounded border border-cyber-border hover:border-neon-cyan/40"
								onclick={() => (viewportX = Math.min(25, viewportX + 1))}>&rarr;</button
							>
							<span class="text-[10px] text-cyber-muted">({viewportX},{viewportY})</span>
						</div>
					</div>

					<!-- Toggles -->
					<div class="space-y-2">
						<label class="flex items-center gap-2 text-[11px] cursor-pointer">
							<input
								type="checkbox"
								bind:checked={showViewport}
								class="accent-neon-cyan"
							/>
							<span class="text-cyber-fg">Viewport overlay</span>
						</label>
						<label class="flex items-center gap-2 text-[11px] cursor-pointer">
							<input
								type="checkbox"
								bind:checked={showScores}
								class="accent-neon-cyan"
							/>
							<span class="text-cyber-fg">Score overlay</span>
						</label>
					</div>
				</div>
			{/if}
		</GlassPanel>

		<!-- Terrain breakdown -->
		{#if Object.keys(terrainCounts).length}
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">Terrain</h2>
				<div class="space-y-1">
					{#each Object.entries(terrainCounts).sort((a, b) => Number(b[1]) - Number(a[1])) as [code, count]}
						{@const pct = ((count as number) / totalCells) * 100}
						<div class="flex items-center gap-2 text-[10px]">
							<span
								class="w-2.5 h-2.5 rounded-sm shrink-0"
								style="background: {TerrainColors[Number(code)] ?? '#333'}"
							></span>
							<span class="flex-1 truncate text-cyber-muted">
								{TerrainNames[Number(code)] ?? `Code ${code}`}
							</span>
							<span class="text-cyber-muted">{pct.toFixed(0)}%</span>
							<span class="text-cyber-muted w-6 text-right">{count}</span>
						</div>
					{/each}
				</div>
			</GlassPanel>
		{/if}

		<!-- Settlements -->
		{#if settlements.length}
			<GlassPanel>
				<h2 class="text-xs text-neon-cyan uppercase tracking-wider mb-3">
					Settlements ({settlements.length})
				</h2>
				<div class="space-y-1 max-h-48 overflow-y-auto">
					{#each settlements as s, i}
						<div class="text-[10px] flex items-center gap-1 {s.alive ? 'text-cyber-fg' : 'text-cyber-muted line-through'}">
							<span class="text-neon-gold">#{i}</span>
							<span>({s.x},{s.y})</span>
							<span class="text-cyber-muted">pop:{s.population.toFixed(0)}</span>
							{#if s.has_port}
								<span class="text-neon-cyan">port</span>
							{/if}
						</div>
					{/each}
				</div>
			</GlassPanel>
		{/if}

		<!-- AI Scene Generator -->
		<GlassPanel>
			<h2 class="text-xs text-neon-magenta uppercase tracking-wider mb-3">AI Render</h2>
			<p class="text-[10px] text-cyber-muted mb-2">
				Capture the current camera view and generate a photorealistic scene using Imagen 3.
			</p>
			<button
				class="w-full px-3 py-2 text-[11px] font-medium rounded border
					border-neon-magenta/60 text-neon-magenta
					hover:bg-neon-magenta/10 hover:border-neon-magenta
					disabled:opacity-40 disabled:cursor-not-allowed
					transition-colors"
				onclick={handleGenerate}
				disabled={!captureFn || generating || generatingAll}
			>
				{#if generating}
					<span class="inline-flex items-center gap-1.5">
						<span class="w-3 h-3 border-2 border-neon-magenta/40 border-t-neon-magenta rounded-full animate-spin"></span>
						Generating...
					</span>
				{:else}
					Generate Scene
				{/if}
			</button>

			<!-- Generate All button -->
			<button
				class="w-full px-3 py-2 mt-2 text-[11px] font-medium rounded border
					border-neon-gold/60 text-neon-gold
					hover:bg-neon-gold/10 hover:border-neon-gold
					disabled:opacity-40 disabled:cursor-not-allowed
					transition-colors"
				onclick={handleGenerateAll}
				disabled={!captureFn || generating || generatingAll || rounds.length === 0}
			>
				{#if generatingAll}
					<span class="inline-flex items-center gap-1.5">
						<span class="w-3 h-3 border-2 border-neon-gold/40 border-t-neon-gold rounded-full animate-spin"></span>
						Generating {generateAllProgress.current}/{generateAllProgress.total}...
					</span>
				{:else}
					Generate All Rounds ({rounds.length})
				{/if}
			</button>

			{#if generatingAll}
				<button
					class="w-full px-2 py-1 mt-1 text-[10px] text-score-bad border border-score-bad/40
						rounded hover:bg-score-bad/10 transition-colors"
					onclick={cancelGenerateAll}
				>
					Cancel
				</button>
			{/if}

			{#if generationError}
				<p class="text-[10px] text-score-bad mt-2 break-words">{generationError}</p>
			{/if}
			{#if generateAllErrors.length > 0}
				<div class="mt-2 space-y-0.5">
					{#each generateAllErrors as err}
						<p class="text-[9px] text-score-bad break-words">{err}</p>
					{/each}
				</div>
			{/if}
		</GlassPanel>

		<!-- Gallery -->
		{#if gallery.length > 0}
			<GlassPanel>
				<div class="flex items-center justify-between mb-3">
					<h2 class="text-xs text-neon-cyan uppercase tracking-wider">
						Gallery ({gallery.length})
					</h2>
					<button
						class="text-[10px] text-cyber-muted hover:text-neon-cyan transition-colors"
						onclick={() => (showGallery = !showGallery)}
					>
						{showGallery ? 'Collapse' : 'Expand'}
					</button>
				</div>
				{#if showGallery}
					<div class="space-y-2 max-h-80 overflow-y-auto">
						{#each gallery as entry}
							<div class="group relative">
								<button
									class="w-full text-left"
									onclick={() => (modalImage = { url: `${GO_API}${entry.url}`, prompt: entry.prompt })}
								>
									<img
										src="{GO_API}{entry.url}"
										alt={entry.prompt}
										class="w-full rounded border border-cyber-border hover:border-neon-cyan/40 transition-colors"
									/>
								</button>
								<button
									class="absolute top-1 right-1 w-5 h-5 rounded bg-black/70 text-score-bad
										text-[11px] leading-none flex items-center justify-center
										opacity-0 group-hover:opacity-100 transition-opacity
										hover:bg-score-bad hover:text-white"
									onclick={(e) => { e.stopPropagation(); handleDelete(entry); }}
									title="Delete image"
								>
									x
								</button>
								<div class="text-[9px] text-cyber-muted mt-0.5">
									{new Date(entry.created_at).toLocaleString('nb-NO', {
										month: 'short',
										day: 'numeric',
										hour: '2-digit',
										minute: '2-digit'
									})}
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</GlassPanel>
		{/if}
	</div>
</div>

<!-- Image modal -->
{#if modalImage}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
		onclick={() => (modalImage = null)}
		onkeydown={(e) => e.key === 'Escape' && (modalImage = null)}
		role="dialog"
	>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="relative max-w-[85vw] max-h-[85vh]" onclick={(e) => e.stopPropagation()}>
			<img
				src={modalImage.url}
				alt={modalImage.prompt}
				class="max-w-full max-h-[85vh] rounded-lg border border-cyber-border shadow-2xl"
			/>
			<button
				class="absolute top-2 right-2 w-8 h-8 rounded-full bg-black/70 text-cyber-fg
					text-sm flex items-center justify-center hover:bg-score-bad hover:text-white
					transition-colors"
				onclick={() => (modalImage = null)}
			>
				x
			</button>
			<p class="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent
				text-[10px] text-cyber-muted rounded-b-lg break-words">
				{modalImage.prompt}
			</p>
		</div>
	</div>
{/if}
