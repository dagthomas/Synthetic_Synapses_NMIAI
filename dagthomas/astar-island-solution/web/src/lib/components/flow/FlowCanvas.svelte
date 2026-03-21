<script lang="ts">
	import FlowNode from './FlowNode.svelte';
	import FlowEdge from './FlowEdge.svelte';
	import { NODES, EDGES } from './flow-data';
	import type { FlowState } from './flow-types';
	import { TIER_COLORS } from './flow-types';

	let {
		flowState,
		onSelectNode,
		selectedNodeId = null,
	}: {
		flowState: FlowState;
		onSelectNode?: (id: string | null) => void;
		selectedNodeId?: string | null;
	} = $props();

	// Zoom & pan state
	let containerEl: HTMLDivElement | undefined = $state();
	let viewX = $state(0);
	let viewY = $state(0);
	let scale = $state(1);
	let isPanning = $state(false);
	let panStartX = $state(0);
	let panStartY = $state(0);
	let panStartViewX = $state(0);
	let panStartViewY = $state(0);

	// Canvas dimensions
	let canvasWidth = $state(1800);
	let canvasHeight = $state(800);

	// Compute bounding box for auto-fit
	function autoFit(containerWidth: number, containerHeight: number) {
		if (!containerWidth || !containerHeight) return;
		let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
		for (const node of NODES) {
			minX = Math.min(minX, node.position.x);
			minY = Math.min(minY, node.position.y);
			maxX = Math.max(maxX, node.position.x + node.size.w);
			maxY = Math.max(maxY, node.position.y + node.size.h);
		}
		const padding = 60;
		const bw = maxX - minX + padding * 2;
		const bh = maxY - minY + padding * 2;
		scale = Math.min(containerWidth / bw, containerHeight / bh, 1.2);
		viewX = (containerWidth - bw * scale) / 2 - minX * scale + padding * scale;
		viewY = (containerHeight - bh * scale) / 2 - minY * scale + padding * scale;
	}

	$effect(() => {
		if (containerEl) {
			const rect = containerEl.getBoundingClientRect();
			autoFit(rect.width, rect.height);
		}
	});

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY > 0 ? 0.92 : 1.08;
		const rect = containerEl!.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;

		// Zoom toward mouse position
		const newScale = Math.max(0.3, Math.min(2.5, scale * factor));
		viewX = mx - (mx - viewX) * (newScale / scale);
		viewY = my - (my - viewY) * (newScale / scale);
		scale = newScale;
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button !== 0) return;
		// Only pan if clicking on canvas background (not a node)
		if ((e.target as HTMLElement).closest('.flow-node')) return;
		isPanning = true;
		panStartX = e.clientX;
		panStartY = e.clientY;
		panStartViewX = viewX;
		panStartViewY = viewY;
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isPanning) return;
		viewX = panStartViewX + (e.clientX - panStartX);
		viewY = panStartViewY + (e.clientY - panStartY);
	}

	function handleMouseUp() {
		isPanning = false;
	}

	function isEdgeActive(edge: typeof EDGES[0]): boolean {
		if (!edge.animated || !edge.activeWhen) return false;
		return flowState.nodeStatuses[edge.activeWhen] === 'active';
	}

	function handleNodeClick(nodeId: string) {
		onSelectNode?.(nodeId === selectedNodeId ? null : nodeId);
	}

	function handleCanvasClick(e: MouseEvent) {
		if (!(e.target as HTMLElement).closest('.flow-node') && !(e.target as HTMLElement).closest('.pipeline-panel')) {
			onSelectNode?.(null);
		}
	}
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	bind:this={containerEl}
	class="relative w-full h-full overflow-hidden rounded-lg"
	style="background: radial-gradient(ellipse at 30% 40%, rgba(0, 255, 240, 0.03) 0%, transparent 50%), radial-gradient(ellipse at 70% 60%, rgba(255, 0, 255, 0.02) 0%, transparent 50%), var(--color-cyber-bg);"
	onwheel={handleWheel}
	onmousedown={handleMouseDown}
	onmousemove={handleMouseMove}
	onmouseup={handleMouseUp}
	onmouseleave={handleMouseUp}
	onclick={handleCanvasClick}
>
	<!-- Grid background -->
	<div class="absolute inset-0 pointer-events-none" style="
		background-image:
			linear-gradient(rgba(0, 255, 240, 0.025) 1px, transparent 1px),
			linear-gradient(90deg, rgba(0, 255, 240, 0.025) 1px, transparent 1px);
		background-size: {30 * scale}px {30 * scale}px;
		background-position: {viewX % (30 * scale)}px {viewY % (30 * scale)}px;
	"></div>

	<!-- Transform wrapper -->
	<div
		class="absolute origin-top-left"
		style="transform: translate({viewX}px, {viewY}px) scale({scale}); will-change: transform;"
	>
		<!-- SVG Edge Layer -->
		<svg
			class="absolute top-0 left-0 pointer-events-none"
			width={canvasWidth}
			height={canvasHeight}
			style="overflow: visible;"
		>
			<defs>
				<filter id="edge-glow" x="-50%" y="-50%" width="200%" height="200%">
					<feGaussianBlur stdDeviation="3" result="blur" />
					<feMerge>
						<feMergeNode in="blur" />
						<feMergeNode in="SourceGraphic" />
					</feMerge>
				</filter>
			</defs>

			{#each EDGES as edge}
				<FlowEdge {edge} nodes={NODES} active={isEdgeActive(edge)} />
			{/each}
		</svg>

		<!-- Node Layer -->
		{#each NODES as node}
			<FlowNode
				{node}
				status={flowState.nodeStatuses[node.id] || 'unknown'}
				metrics={flowState.nodeMetrics[node.id] || {}}
				selected={selectedNodeId === node.id}
				onclick={() => handleNodeClick(node.id)}
			/>
		{/each}
	</div>

	<!-- Legend -->
	<div class="absolute bottom-4 left-4 glass p-3 text-[11px] space-y-1.5 pointer-events-none z-30" style="backdrop-filter: blur(12px);">
		<div class="text-cyber-muted uppercase tracking-widest mb-2 text-[10px]">Node Types</div>
		{#each Object.entries(TIER_COLORS) as [tier, color]}
			<div class="flex items-center gap-2">
				<div class="w-3 h-3 rounded-sm" style="background: {color}; opacity: 0.6; box-shadow: 0 0 4px {color};"></div>
				<span class="text-cyber-fg capitalize">{tier}</span>
			</div>
		{/each}
		<div class="border-t border-cyber-border pt-1.5 mt-1.5">
			<div class="flex items-center gap-2">
				<div class="w-3 h-0 border-t border-dashed border-cyber-muted"></div>
				<span class="text-cyber-muted">data flow</span>
			</div>
			<div class="flex items-center gap-2 mt-1">
				<div class="w-3 h-0 border-t border-solid border-cyber-muted"></div>
				<span class="text-cyber-muted">process flow</span>
			</div>
		</div>
	</div>

	<!-- Zoom controls -->
	<div class="absolute bottom-4 right-4 flex flex-col gap-1 z-30">
		<button
			class="glass w-8 h-8 flex items-center justify-center text-cyber-muted hover:text-neon-cyan transition-colors text-sm"
			onclick={() => { scale = Math.min(2.5, scale * 1.2); }}
		>+</button>
		<button
			class="glass w-8 h-8 flex items-center justify-center text-cyber-muted hover:text-neon-cyan transition-colors text-sm"
			onclick={() => { if (containerEl) { const r = containerEl.getBoundingClientRect(); autoFit(r.width, r.height); } }}
		>
			<span class="text-[10px]">fit</span>
		</button>
		<button
			class="glass w-8 h-8 flex items-center justify-center text-cyber-muted hover:text-neon-cyan transition-colors text-sm"
			onclick={() => { scale = Math.max(0.3, scale * 0.83); }}
		>&minus;</button>
	</div>
</div>
