<script lang="ts">
	let {
		data,
		width = 120,
		height = 30,
		color = 'var(--color-neon-cyan)',
		fill = true
	}: {
		data: number[];
		width?: number;
		height?: number;
		color?: string;
		fill?: boolean;
	} = $props();

	let points = $derived.by(() => {
		if (!data.length) return '';
		const min = Math.min(...data);
		const max = Math.max(...data);
		const range = max - min || 1;
		return data
			.map((v, i) => {
				const x = (i / (data.length - 1)) * width;
				const y = height - ((v - min) / range) * (height - 2) - 1;
				return `${x},${y}`;
			})
			.join(' ');
	});

	let fillPoints = $derived.by(() => {
		if (!points) return '';
		return `0,${height} ${points} ${width},${height}`;
	});

	// Last point for the glowing dot
	let lastPoint = $derived.by(() => {
		if (!data.length) return null;
		const min = Math.min(...data);
		const max = Math.max(...data);
		const range = max - min || 1;
		const x = width;
		const y = height - ((data[data.length - 1] - min) / range) * (height - 2) - 1;
		return { x, y };
	});
</script>

<svg {width} {height} class="overflow-visible">
	<defs>
		<linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
			<stop offset="0%" stop-color={color} stop-opacity="0.3" />
			<stop offset="100%" stop-color={color} stop-opacity="0.02" />
		</linearGradient>
		<filter id="spark-glow">
			<feGaussianBlur stdDeviation="2" result="blur" />
			<feMerge>
				<feMergeNode in="blur" />
				<feMergeNode in="SourceGraphic" />
			</feMerge>
		</filter>
	</defs>
	{#if fill && fillPoints}
		<polygon points={fillPoints} fill="url(#spark-fill)" />
	{/if}
	{#if points}
		<polyline {points} fill="none" stroke={color} stroke-width="1.5" filter="url(#spark-glow)" />
	{/if}
	{#if lastPoint}
		<circle cx={lastPoint.x} cy={lastPoint.y} r="2.5" fill={color} class="animate-pulse-glow">
			<animate attributeName="r" values="2;3.5;2" dur="2s" repeatCount="indefinite" />
		</circle>
		<circle cx={lastPoint.x} cy={lastPoint.y} r="5" fill={color} opacity="0.2">
			<animate attributeName="r" values="4;7;4" dur="2s" repeatCount="indefinite" />
			<animate attributeName="opacity" values="0.2;0.05;0.2" dur="2s" repeatCount="indefinite" />
		</circle>
	{/if}
</svg>
