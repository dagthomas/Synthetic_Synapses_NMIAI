<script lang="ts">
	import type { FlowEdgeDef, FlowNodeDef } from './flow-types';
	import { computeEdgePath, getNode } from './flow-data';

	let {
		edge,
		nodes,
		active = false,
	}: {
		edge: FlowEdgeDef;
		nodes: FlowNodeDef[];
		active: boolean;
	} = $props();

	let fromNode = $derived(nodes.find((n) => n.id === edge.from));
	let toNode = $derived(nodes.find((n) => n.id === edge.to));
	let path = $derived(fromNode && toNode ? computeEdgePath(fromNode, toNode) : '');
	let edgeColor = $derived(edge.color || 'var(--color-neon-cyan)');

	let pathId = $derived(`edge-path-${edge.id}`);

	// Midpoint for label
	let labelPos = $derived(() => {
		if (!fromNode || !toNode) return { x: 0, y: 0 };
		const fc = { x: fromNode.position.x + fromNode.size.w / 2, y: fromNode.position.y + fromNode.size.h / 2 };
		const tc = { x: toNode.position.x + toNode.size.w / 2, y: toNode.position.y + toNode.size.h / 2 };
		return { x: (fc.x + tc.x) / 2, y: (fc.y + tc.y) / 2 - 8 };
	});
</script>

{#if path}
	<!-- Glow layer (wider, blurred) -->
	{#if active}
		<path
			d={path}
			fill="none"
			stroke={edgeColor}
			stroke-width="4"
			stroke-opacity="0.15"
			filter="url(#edge-glow)"
			stroke-dasharray={edge.dashed ? '6 4' : 'none'}
		/>
	{/if}

	<!-- Main path -->
	<path
		id={pathId}
		d={path}
		fill="none"
		stroke={edgeColor}
		stroke-width={active ? 1.5 : 1}
		stroke-opacity={active ? 0.5 : 0.12}
		stroke-dasharray={edge.dashed ? '6 4' : 'none'}
		stroke-linecap="round"
	/>

	<!-- Animated dash flow for active edges -->
	{#if active && !edge.dashed}
		<path
			d={path}
			fill="none"
			stroke={edgeColor}
			stroke-width="2"
			stroke-opacity="0.3"
			stroke-dasharray="4 16"
			stroke-linecap="round"
		>
			<animate
				attributeName="stroke-dashoffset"
				from="0"
				to="-20"
				dur="1s"
				repeatCount="indefinite"
			/>
		</path>
	{/if}

	<!-- Animated particles -->
	{#if active}
		{#each [0, 1, 2] as i}
			<circle r="2.5" fill={edgeColor} opacity={0.9 - i * 0.2}>
				<animateMotion
					dur="{2.5 + i * 0.8}s"
					repeatCount="indefinite"
					begin="{i * 0.7}s"
				>
					<mpath href="#{pathId}" />
				</animateMotion>
			</circle>
		{/each}

		<!-- Bright lead particle -->
		<circle r="1.5" fill="white" opacity="0.9">
			<animateMotion
				dur="2.5s"
				repeatCount="indefinite"
				begin="0s"
			>
				<mpath href="#{pathId}" />
			</animateMotion>
		</circle>
	{/if}

	<!-- Label -->
	{#if edge.label}
		{@const lp = labelPos()}
		<text
			x={lp.x}
			y={lp.y}
			text-anchor="middle"
			fill={active ? edgeColor : 'rgba(107, 107, 123, 0.5)'}
			font-size="9"
			font-family="'JetBrains Mono', monospace"
			opacity={active ? 0.7 : 0.4}
		>
			{edge.label}
		</text>
	{/if}
{/if}
