<script lang="ts">
	import type { FlowNodeDef, NodeStatus } from './flow-types';
	import { TIER_COLORS } from './flow-types';

	let {
		node,
		status = 'unknown',
		metrics = {},
		selected = false,
		onclick,
	}: {
		node: FlowNodeDef;
		status: NodeStatus;
		metrics: Record<string, string | number>;
		selected?: boolean;
		onclick?: () => void;
	} = $props();

	let isData = $derived(node.tier === 'data');
	let isActive = $derived(status === 'active');
	let tierColor = $derived(TIER_COLORS[node.tier]);

	let glowIntensity = $derived(
		isActive ? '0 0 20px' : selected ? '0 0 12px' : '0 0 6px'
	);

	let borderOpacity = $derived(isActive ? 0.6 : selected ? 0.4 : 0.15);

	function statusDotColor(s: NodeStatus): string {
		switch (s) {
			case 'active': return '#00e676';
			case 'idle': return '#6b6b7b';
			case 'error': return '#e53935';
			default: return '#4a4a5a';
		}
	}

	function formatMetric(def: { format: string }, value: string | number): string {
		if (value === undefined || value === null) return '\u2014';
		switch (def.format) {
			case 'number':
				return typeof value === 'number' ? value.toLocaleString() : String(value);
			case 'score':
				return typeof value === 'number' ? value.toFixed(2) : String(value);
			case 'rate':
				return typeof value === 'number' ? `${value.toFixed(0)}/hr` : String(value);
			default:
				return String(value);
		}
	}
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="flow-node absolute cursor-pointer transition-all duration-300 select-none"
	class:flow-node-data={isData}
	class:flow-node-active={isActive}
	class:flow-node-selected={selected}
	style="
		left: {node.position.x}px;
		top: {node.position.y}px;
		width: {node.size.w}px;
		height: {node.size.h}px;
		--node-color: {node.color};
		--glow: {glowIntensity};
		--border-opacity: {borderOpacity};
	"
	onclick={onclick}
>
	<!-- Background -->
	<div class="absolute inset-0 rounded-lg overflow-hidden" style="
		background: {isData ? 'rgba(20, 20, 35, 0.7)' : 'rgba(26, 26, 46, 0.85)'};
		backdrop-filter: blur(12px);
		border: {isData ? '1px dashed' : '1px solid'};
		border-color: color-mix(in srgb, {node.color} {borderOpacity * 100}%, transparent);
		border-radius: 8px;
		box-shadow: {glowIntensity} color-mix(in srgb, {node.color} 20%, transparent){isActive ? `, inset 0 0 15px color-mix(in srgb, ${node.color} 5%, transparent)` : ''};
	">
		<!-- Active shimmer effect -->
		{#if isActive}
			<div class="absolute inset-0 animate-pulse-glow pointer-events-none" style="
				background: linear-gradient(135deg, transparent 40%, color-mix(in srgb, {node.color} 5%, transparent) 50%, transparent 60%);
			"></div>
		{/if}
	</div>

	<!-- Content -->
	<div class="relative z-10 p-3 h-full flex flex-col">
		<!-- Header: icon + label + status dot -->
		<div class="flex items-center gap-2 mb-1.5">
			<span class="text-base" style="color: {node.color}; {isActive ? `text-shadow: 0 0 8px ${node.color}` : ''}">{node.icon}</span>
			<div class="flex-1 min-w-0">
				<div class="text-[11px] font-bold tracking-wider truncate" style="color: {node.color}; {isActive ? `text-shadow: 0 0 6px ${node.color}` : ''}">{node.label}</div>
				{#if node.sublabel}
					<div class="text-[11px] text-cyber-muted truncate">{node.sublabel}</div>
				{/if}
			</div>
			<!-- Status dot -->
			<div class="relative flex-shrink-0">
				<div class="w-2.5 h-2.5 rounded-full" style="background: {statusDotColor(status)}; box-shadow: 0 0 6px {statusDotColor(status)};"></div>
				{#if isActive}
					<div class="absolute inset-0 w-2.5 h-2.5 rounded-full animate-ping" style="background: {statusDotColor(status)}; opacity: 0.4;"></div>
				{/if}
			</div>
		</div>

		<!-- Metrics -->
		<div class="flex-1 flex flex-col justify-center gap-0.5">
			{#each node.metrics as metricDef}
				{@const value = metrics[metricDef.key]}
				<div class="flex items-baseline justify-between gap-1">
					<span class="text-[11px] text-cyber-muted uppercase tracking-wider">{metricDef.label}</span>
					<span class="text-[11px] font-bold" style="color: {metricDef.format === 'score' ? (typeof value === 'number' && value >= 85 ? 'var(--color-score-ok)' : 'var(--color-cyber-fg)') : 'var(--color-cyber-fg)'}">{formatMetric(metricDef, value)}</span>
				</div>
			{/each}
		</div>

		<!-- Expand indicator for prediction group -->
		{#if node.isExpandable}
			<div class="text-center text-[11px] text-cyber-muted mt-1">
				<span class="opacity-60">click to expand</span>
			</div>
		{/if}
	</div>
</div>

<style>
	.flow-node:hover {
		transform: scale(1.04);
		z-index: 20;
	}

	.flow-node-selected {
		z-index: 25;
	}

	@keyframes ping {
		75%, 100% {
			transform: scale(2.5);
			opacity: 0;
		}
	}

	.animate-ping {
		animation: ping 1.5s cubic-bezier(0, 0, 0.2, 1) infinite;
	}
</style>
