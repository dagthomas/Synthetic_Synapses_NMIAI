<script>
	let {
		width,
		height,
		cellSize,
		wallSet,
		shelfSet,
		itemMap,
		dropOff,
		spawn,
		bots,
		botPositions,
		botColors,
		selectedBot = null,
		onSelectBot = () => {},
		activeTypes = new Set(),
		previewTypes = new Set(),
	} = $props();

	const ITEM_COLORS = {
		milk: '#00FF41', bread: '#FF0055', eggs: '#FF0055', butter: '#0DF0E3',
		cheese: '#FF0055', pasta: '#00FF41', rice: '#0DF0E3', juice: '#00FF41',
		yogurt: '#0DF0E3', cereal: '#0DF0E3', flour: '#FF0055', sugar: '#0DF0E3',
		coffee: '#FF0055', tea: '#00FF41', oil: '#0DF0E3', salt: '#00FF41',
		cream: '#0DF0E3', oats: '#FF0055',
	};

	function cellType(x, y) {
		const key = `${x},${y}`;
		if (x === dropOff[0] && y === dropOff[1]) return 'dropoff';
		if (x === spawn[0] && y === spawn[1]) return 'spawn';
		if (wallSet.has(key)) return 'wall';
		if (shelfSet.has(key)) return 'shelf';
		return 'floor';
	}

	function cellColor(type, x, y) {
		switch (type) {
			case 'wall':
				if (x === 0 || x === width - 1 || y === 0 || y === height - 1) return '#2d333b';
				return '#373e47';
			case 'shelf': return '#0d2818';
			case 'dropoff': return '#39d35330';
			case 'spawn': return '#58a6ff30';
			default: return '#010409';
		}
	}

	function cellStroke(type, x, y) {
		switch (type) {
			case 'wall':
				if (x === 0 || x === width - 1 || y === 0 || y === height - 1) return '#444c56';
				return '#545d68';
			case 'shelf': return '#1a4d2e';
			case 'dropoff': return '#39d353';
			case 'spawn': return '#58a6ff';
			default: return '#161b22';
		}
	}

	// Build cells
	let cells = $derived.by(() => {
		const result = [];
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const type = cellType(x, y);
				result.push({ x, y, type });
			}
		}
		return result;
	});

	// Items on shelves
	let shelfItems = $derived.by(() => {
		const result = [];
		for (const [key, items] of itemMap) {
			const [x, y] = key.split(',').map(Number);
			result.push({ x, y, items, type: items[0].type });
		}
		return result;
	});
</script>

<svg
	viewBox="0 0 {width * cellSize} {height * cellSize}"
	xmlns="http://www.w3.org/2000/svg"
	style="width: 100%; height: auto; display: block;"
>
	<defs>
		<!-- Drop-off indicator -->
		<pattern id="dropoff-pattern" width="6" height="6" patternUnits="userSpaceOnUse">
			<path d="M0 6L6 0" stroke="#39d35344" stroke-width="1"/>
		</pattern>
		<!-- Bot glow -->
		<filter id="bot-glow">
			<feGaussianBlur stdDeviation="2" result="coloredBlur"/>
			<feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
		</filter>
		<!-- Selected bot glow -->
		<filter id="selected-glow">
			<feGaussianBlur stdDeviation="3" result="coloredBlur"/>
			<feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
		</filter>
		<!-- Butter scan clip -->
		<clipPath id="butterClip">
			<path d="M 7 14 L 17 11 L 21 13 L 11 16 Z M 7 14 L 11 16 L 11 20 L 7 18 Z M 11 16 L 21 13 L 21 17 L 11 20 Z" />
		</clipPath>
	</defs>

	<!-- Grid cells -->
	{#each cells as cell}
		<rect
			x={cell.x * cellSize}
			y={cell.y * cellSize}
			width={cellSize}
			height={cellSize}
			fill={cellColor(cell.type, cell.x, cell.y)}
			stroke={cellStroke(cell.type, cell.x, cell.y)}
			stroke-width={cell.type === 'dropoff' || cell.type === 'spawn' ? 2 : 0.5}
		/>
		{#if cell.type === 'dropoff'}
			<svg x={cell.x * cellSize} y={cell.y * cellSize} width={cellSize} height={cellSize} viewBox="0 0 28 28">
				<rect x="1" y="1" width="26" height="26" fill="#39d353" opacity="0.12" />
				<rect x="1" y="1" width="26" height="26" fill="none" stroke="#39d353" stroke-width="1" stroke-dasharray="2 2" opacity="0.5" />
				<rect x="2" y="2" width="4" height="4" fill="#39d353" opacity="0.8" />
				<rect x="22" y="2" width="4" height="4" fill="#39d353" opacity="0.8" />
				<rect x="2" y="22" width="4" height="4" fill="#39d353" opacity="0.8" />
				<rect x="22" y="22" width="4" height="4" fill="#39d353" opacity="0.8" />
				<path d="M 11 4 L 14 6 L 17 4" fill="none" stroke="#b8d4a0" stroke-width="1.5" class="anim-arrow-flash-1" stroke-linejoin="round" />
				<path d="M 11 24 L 14 22 L 17 24" fill="none" stroke="#b8d4a0" stroke-width="1.5" class="anim-arrow-flash-1" stroke-linejoin="round" />
				<path d="M 4 11 L 6 14 L 4 17" fill="none" stroke="#b8d4a0" stroke-width="1.5" class="anim-arrow-flash-2" stroke-linejoin="round" />
				<path d="M 24 11 L 22 14 L 24 17" fill="none" stroke="#b8d4a0" stroke-width="1.5" class="anim-arrow-flash-2" stroke-linejoin="round" />
				<rect x="8" y="8" width="12" height="12" fill="#000000" opacity="0.6" rx="1" />
				<rect x="8" y="8" width="12" height="12" fill="none" stroke="#39d353" stroke-width="1" rx="1" />
				<line x1="8" y1="8" x2="20" y2="8" stroke="#b8d4a0" stroke-width="1.5" class="anim-hatch-scan" filter="url(#bot-glow)"/>
				<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="900" fill="#ffffff" font-family="sans-serif">D</text>
			</svg>
		{/if}
		{#if cell.type === 'spawn'}
			<svg x={cell.x * cellSize} y={cell.y * cellSize} width={cellSize} height={cellSize} viewBox="0 0 28 28">
				<rect x="1" y="1" width="26" height="26" fill="#58a6ff" opacity="0.12" />
				<path d="M 0 6 V 0 H 6" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.8"/>
				<path d="M 28 6 V 0 H 22" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.8"/>
				<path d="M 0 22 V 28 H 6" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.8"/>
				<path d="M 28 22 V 28 H 22" fill="none" stroke="#58a6ff" stroke-width="1.5" opacity="0.8"/>
				<circle cx="14" cy="14" r="6" fill="none" stroke="#58a6ff" stroke-width="1" class="anim-pad-pulse" />
				<circle cx="14" cy="14" r="10" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-dasharray="4 4" class="anim-pad-rotate" opacity="0.6"/>
				<circle cx="14" cy="14" r="7" fill="none" stroke="#58a6ff" stroke-width="1" stroke-dasharray="2 2" class="anim-pad-rotate-reverse" opacity="0.8"/>
				<circle cx="14" cy="14" r="5" fill="#58a6ff" opacity="0.3" />
				<circle cx="14" cy="14" r="5" fill="none" stroke="#58a6ff" stroke-width="1" />
				<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="900" fill="#ffffff" font-family="sans-serif">S</text>
			</svg>
		{/if}
	{/each}

	<!-- Animated items on shelves (cyberpunk) -->
	{#each shelfItems as si}
		{@const isActive = activeTypes.has(si.type)}
		{@const isPreview = !isActive && previewTypes.has(si.type)}
		<!-- Active/preview highlight behind item -->
		{#if isActive}
			<rect
				x={si.x * cellSize - 2}
				y={si.y * cellSize - 2}
				width={cellSize + 4}
				height={cellSize + 4}
				rx="3"
				fill="rgba(250, 204, 21, 0.2)"
				stroke="#facc15"
				stroke-width="2"
				opacity="0.9"
				class="anim-active-glow"
			/>
		{:else if isPreview}
			<rect
				x={si.x * cellSize - 2}
				y={si.y * cellSize - 2}
				width={cellSize + 4}
				height={cellSize + 4}
				rx="3"
				fill="rgba(244, 114, 182, 0.18)"
				stroke="#f472b6"
				stroke-width="2"
				opacity="0.85"
				class="anim-preview-glow"
			/>
		{/if}
		<svg
			x={si.x * cellSize}
			y={si.y * cellSize}
			width={cellSize}
			height={cellSize}
			viewBox="0 0 28 28"
		>
			<ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="cyber-shadow" />
			{#if si.type === 'milk'}
				<!-- Data Flow Carton -->
				<g>
					<path d="M10 11 L14 7 L18 11 V21 C18 21.5 17.5 22 17 22 H11 C10.5 22 10 21.5 10 21 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round" opacity="0.3"/>
					<path d="M10 11 L14 7 L18 11 V21 C18 21.5 17.5 22 17 22 H11 C10.5 22 10 21.5 10 21 Z" fill="none" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round" class="cyber-flow"/>
					<path d="M9 11 H19" stroke="#00FF41" stroke-width="1.2" stroke-linecap="round"/>
					<rect x="12" y="14" width="4" height="4" fill="none" stroke="#0DF0E3" stroke-width="1"/>
					<circle cx="14" cy="16" r="1" fill="#FF0055" class="cyber-blink"/>
				</g>
			{:else if si.type === 'bread'}
				<!-- Wireframe Loaf -->
				<g>
					<path class="cyber-flicker" d="M 6 15 C 6 10, 22 10, 22 15 L 21 21 C 21 21.5, 20.5 22, 20 22 H 8 C 7.5 22, 7 21.5, 7 21 Z" fill="#161B22" stroke="#FF0055" stroke-width="1.2"/>
					<path class="cyber-score" d="M 10 13 V 20 M 14 13 V 20 M 18 13 V 20" stroke="#0DF0E3" stroke-width="1.2" stroke-linecap="square"/>
				</g>
			{:else if si.type === 'eggs'}
				<!-- Incubation Pods -->
				<g>
					<ellipse cx="10" cy="15" rx="3" ry="4" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<ellipse cx="14" cy="14" rx="3" ry="4" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<ellipse cx="18" cy="15" rx="3" ry="4" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 5 17 L 6 21 C 6 21.5, 6.5 22, 7 22 H 21 C 21.5 22, 22 21.5, 22 21 L 23 17 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/>
					<circle cx="10" cy="15" r="1.5" class="cyber-node-1"/>
					<circle cx="14" cy="14" r="1.5" class="cyber-node-2"/>
					<circle cx="18" cy="15" r="1.5" class="cyber-node-3"/>
				</g>
			{:else if si.type === 'butter'}
				<!-- Isometric Block + Laser Scan -->
				<g>
					<path d="M 7 14 L 17 11 L 21 13 L 11 16 Z M 7 14 L 11 16 L 11 20 L 7 18 Z M 11 16 L 21 13 L 21 17 L 11 20 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M 7 14 L 13 12.5 L 13 16.5 L 7 18 Z" fill="none" stroke="#0DF0E3" stroke-width="1" stroke-dasharray="1 2"/>
					<line x1="0" y1="0" x2="28" y2="0" stroke="#FF0055" stroke-width="1.5" clip-path="url(#butterClip)" class="cyber-scan"/>
				</g>
			{:else if si.type === 'cheese'}
				<!-- Glitch Wedge -->
				<g>
					<path class="cyber-glitch" d="M 7 19 L 21 21 L 18 10 Z" fill="#161B22" stroke="#FF0055" stroke-width="1.2" stroke-linejoin="round"/>
					<rect x="11" y="16" width="2" height="2" fill="#0DF0E3"/>
					<rect x="16" y="16" width="2" height="2" fill="#0DF0E3"/>
					<rect x="14.5" y="12.5" width="1" height="1" fill="#0DF0E3"/>
				</g>
			{:else if si.type === 'pasta'}
				<!-- Rotating Core -->
				<g>
					<path d="M 5 10 L 13 13 L 13 15 L 5 18 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 23 10 L 15 13 L 15 15 L 23 18 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<rect x="12" y="12" width="4" height="4" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2" class="cyber-core"/>
				</g>
			{:else if si.type === 'rice'}
				<!-- Terminal Sack -->
				<g>
					<path d="M 9 12 C 7 12, 7 22, 10 22 H 18 C 21 22, 21 12, 19 12 Z" fill="#161B22" stroke="#0DF0E3" stroke-width="1.2"/>
					<path d="M 12 8 L 14 11 L 16 8 Z" fill="none" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 11 11 H 17" stroke="#00FF41" stroke-width="1.2" stroke-linecap="square"/>
					<text x="12.5" y="19" text-anchor="middle" font-size="6" font-weight="bold" font-family="monospace" fill="#00FF41">[R]</text>
					<rect x="16.5" y="14.5" width="2" height="5" fill="#00FF41" class="cyber-cursor"/>
				</g>
			{:else if si.type === 'juice'}
				<!-- Battery Box -->
				<g>
					<rect x="9" y="10" width="10" height="12" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 15 10 V 6 H 17" fill="none" stroke="#FF0055" stroke-width="1.2" stroke-linecap="square"/>
					<rect x="11" y="12" width="6" height="8" fill="#0D1117"/>
					<rect x="11" y="12" width="6" height="8" class="cyber-battery"/>
				</g>
			{:else if si.type === 'yogurt'}
				<!-- Radar Node -->
				<g>
					<path d="M 9 12 L 10 21 C 10 21.5, 11 22, 14 22 C 17 22, 18 21.5, 18 21 L 19 12 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="none" stroke="#0DF0E3" class="cyber-wave-1"/>
					<ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="none" stroke="#0DF0E3" class="cyber-wave-2"/>
					<ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/>
					<path d="M 12 16 H 16 M 13 19 H 15" stroke="#0DF0E3" stroke-width="1.2" stroke-dasharray="2 2"/>
				</g>
			{:else if si.type === 'cereal'}
				<!-- Bit Flakes -->
				<g>
					<rect x="9" y="8" width="10" height="14" fill="#161B22" stroke="#0DF0E3" stroke-width="1.2"/>
					<rect x="13" y="14" width="2" height="2" fill="#00FF41" class="cyber-hex-1"/>
					<rect x="11" y="15" width="1" height="1" fill="#FF0055" class="cyber-hex-2"/>
					<rect x="16" y="13" width="1" height="1" fill="#FF0055" class="cyber-hex-3"/>
					<rect x="15" y="16" width="1" height="1" fill="#0DF0E3" class="cyber-hex-4"/>
					<rect x="12" y="12" width="1" height="1" fill="#0DF0E3" class="cyber-hex-2"/>
				</g>
			{:else if si.type === 'flour'}
				<!-- Compressed Archive -->
				<g>
					<path d="M 9 10 L 19 10 L 19 21 C 19 21.5, 18.5 22, 18 22 L 10 22 C 9.5 22, 9 21.5, 9 21 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 9 10 L 14 13 L 19 10 L 19 7 C 19 6.5, 18.5 6, 18 6 L 10 6 C 9.5 6, 9 6.5, 9 7 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/>
					<path class="cyber-extract" d="M 14 14 V 19 M 12 16 H 16 M 12 18 H 16" stroke="#FF0055" stroke-width="1.2" stroke-linecap="round"/>
				</g>
			{:else if si.type === 'sugar'}
				<!-- Data Cubes -->
				<g>
					<rect x="9" y="15" width="6" height="6" stroke-width="1" class="cyber-cube-1"/>
					<rect x="15" y="15" width="6" height="6" stroke-width="1" class="cyber-cube-2"/>
					<rect x="12" y="10" width="6" height="6" stroke-width="1" class="cyber-cube-3"/>
					<circle cx="12" cy="18" r="0.5" fill="#00FF41"/>
					<circle cx="18" cy="18" r="0.5" fill="#0DF0E3"/>
					<circle cx="15" cy="13" r="0.5" fill="#FF0055"/>
				</g>
			{:else if si.type === 'coffee'}
				<!-- Hot Code -->
				<g>
					<path d="M 10 12 L 11 20 C 11 21.5, 12 22, 14 22 C 16 22, 17 21.5, 17 20 L 18 12 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 9.5 10 C 9.5 9, 18.5 9, 18.5 10 L 18.5 12 L 9.5 12 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/>
					<path class="cyber-exhaust-1" d="M 12 8 V 6 H 13 V 4" fill="none" stroke="#FF0055" stroke-width="1" stroke-linejoin="miter"/>
					<path class="cyber-exhaust-2" d="M 16 8 V 6 H 15 V 4" fill="none" stroke="#0DF0E3" stroke-width="1" stroke-linejoin="miter"/>
				</g>
			{:else if si.type === 'tea'}
				<!-- Syntax Infusion -->
				<g>
					<path d="M 9 13 V 17 C 9 19, 11 21, 14 21 C 17 21, 19 19, 19 17 V 13 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 19 14 H 21 C 22.5 14, 22.5 17, 21 17 H 19" fill="none" stroke="#0DF0E3" stroke-width="1.2"/>
					<path d="M 14 13 V 9 H 11" fill="none" stroke="#0DF0E3" stroke-width="1" stroke-linecap="square"/>
					<rect x="9" y="7" width="4" height="4" fill="#0D1117" stroke="#00FF41" stroke-width="1"/>
					<line x1="9" y1="7" x2="9" y2="11" stroke="#FF0055" stroke-width="1" class="cyber-tag-scan"/>
				</g>
			{:else if si.type === 'oil'}
				<!-- Coolant Liquid -->
				<g>
					<path d="M 12 11 C 10 14, 10 21, 10 21 C 10 21.5, 10.5 22, 11 22 H 17 C 17.5 22, 18 21.5, 18 21 C 18 21, 18 14, 16 11 V 8 H 12 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<rect x="12" y="6" width="4" height="2" fill="#0DF0E3"/>
					<rect x="13.5" y="15" width="1" height="1" fill="#0DF0E3" class="cyber-bubble-1"/>
					<rect x="12" y="17" width="1" height="1" fill="#0DF0E3" class="cyber-bubble-2"/>
					<rect x="15" y="16" width="1" height="1" fill="#0DF0E3" class="cyber-bubble-3"/>
					<path d="M 12 11 H 16" stroke="#00FF41" stroke-width="1" stroke-dasharray="2 2"/>
				</g>
			{:else if si.type === 'salt'}
				<!-- Crypto Hash Rain -->
				<g>
					<rect x="11" y="10" width="6" height="12" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path d="M 11 10 C 11 7, 17 7, 17 10 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/>
					<rect x="13" y="7.5" width="1" height="1" fill="#FF0055"/>
					<rect x="15" y="8.5" width="1" height="1" fill="#FF0055"/>
					<rect x="12" y="8.5" width="1" height="1" fill="#FF0055"/>
					<line x1="12.5" y1="12" x2="12.5" y2="14" stroke="#0DF0E3" stroke-width="1" class="cyber-rain-1"/>
					<line x1="14" y1="11" x2="14" y2="13" stroke="#FF0055" stroke-width="1" class="cyber-rain-2"/>
					<line x1="15.5" y1="13" x2="15.5" y2="15" stroke="#00FF41" stroke-width="1" class="cyber-rain-3"/>
				</g>
			{:else if si.type === 'cream'}
				<!-- Data Gel Pitcher -->
				<g>
					<path d="M 11 10 L 8 8 V 10 L 10 12 V 20 C 10 21.5, 10.5 22, 12 22 H 16 C 17.5 22, 18 21.5, 18 20 V 10 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M 11 10 H 18" stroke="#00FF41" stroke-width="1.2" stroke-linecap="round"/>
					<path d="M 18 13 H 21 V 17 H 18" fill="none" stroke="#0DF0E3" stroke-width="1.2" stroke-linejoin="round"/>
					<line x1="12" y1="18" x2="16" y2="18" stroke="#0DF0E3" stroke-width="1" stroke-dasharray="1 1"/>
					<line x1="12" y1="15" x2="15" y2="15" stroke="#FF0055" stroke-width="1" stroke-dasharray="1 1"/>
					<rect x="7.5" y="11" width="1" height="2" fill="#FF0055" class="cyber-drip-1"/>
					<rect x="7.5" y="12" width="1" height="1" fill="#0DF0E3" class="cyber-drip-2"/>
				</g>
			{:else if si.type === 'oats'}
				<!-- Fiber Frequency Core -->
				<g>
					<path d="M 9 9.5 V 20 C 9 21.5, 19 21.5, 19 20 V 9.5" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<path class="cyber-fiber" d="M 9 14 C 12 12, 16 16, 19 14" fill="none" stroke="#FF0055" stroke-width="1.2" stroke-dasharray="2 2"/>
					<ellipse cx="14" cy="9.5" rx="5.5" ry="1.5" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/>
					<ellipse cx="14" cy="8.5" rx="5.5" ry="1.5" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/>
					<text x="14" y="19.5" text-anchor="middle" font-size="5" font-family="monospace" font-weight="bold" fill="#0DF0E3">[O]</text>
					<circle cx="14" cy="11.5" r=".75" class="cyber-node-blink"/>
				</g>
			{:else}
				<!-- Fallback: cyber dot -->
				<g>
					<circle cx="14" cy="14" r="5" fill="none" stroke="#00FF41" stroke-width="1.2" class="cyber-flow"/>
					<circle cx="14" cy="14" r="2" fill="#0DF0E3" class="cyber-blink"/>
					<text x="14" y="15" text-anchor="middle" font-size="7" font-weight="bold" font-family="monospace" fill="#00FF41">{si.type.charAt(0).toUpperCase()}</text>
				</g>
			{/if}
		</svg>
	{/each}

	<!-- Bots (Cyber Drone design) -->
	{#each bots as bot (bot.id)}
		{@const bx = bot.position[0] * cellSize}
		{@const by = bot.position[1] * cellSize}
		{@const color = botColors[bot.id % botColors.length]}
		{@const isSelected = selectedBot === bot.id}

		<!-- Animated wrapper: CSS transform transitions smoothly each round -->
		<g
			style="transform: translate({bx}px, {by}px); transition: transform 0.25s ease-out; cursor: pointer;"
			onclick={() => onSelectBot(bot.id)}
		>
			<!-- Selection highlight -->
			{#if isSelected}
				<rect
					x={-2}
					y={-2}
					width={cellSize + 4}
					height={cellSize + 4}
					rx="4"
					fill="none"
					stroke={color}
					stroke-width="2"
					opacity="0.7"
					filter="url(#selected-glow)"
				/>
			{/if}

			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<!-- svelte-ignore a11y_no_static_element_interactions -->
			<svg
				x={0}
				y={0}
				width={cellSize}
				height={cellSize}
				viewBox="0 0 28 28"
				style="--bot-color: {color};"
			>
				<!-- Ground Shadow -->
				<ellipse cx="14" cy="23" rx="6" ry="1.5" fill={color} class="cyber-shadow" />

				<g class="bot-cyber-hover">
					<!-- Plasma Thrusters -->
					<path d="M 12 19 V 22 M 14 19 V 23 M 16 19 V 22" stroke="#0DF0E3" stroke-width="1.2" stroke-linecap="round" class="bot-exhaust"/>

					<!-- Uplink Antenna -->
					<line x1="14" y1="9" x2="14" y2="4" stroke={color} stroke-width="1.2" />
					<circle cx="14" cy="4" r="1.5" class="bot-antenna-node" fill={color} />

					<!-- Side Ports -->
					<rect x="5.5" y="11" width="2" height="4" fill={color} opacity="0.5"/>
					<rect x="20.5" y="11" width="2" height="4" fill={color} opacity="0.5"/>

					<!-- Angular Chassis -->
					<path d="M 9 9 L 19 9 L 21 15 L 7 15 Z" fill={color} opacity="0.7" stroke-linejoin="round"/>
					<path d="M 7 15 L 11 19 H 17 L 21 15" fill={color} opacity="0.5" stroke-linejoin="round"/>

					<!-- Visor & Scanner -->
					<rect x="10" y="11.5" width="8" height="3" fill="#0D1117" />
					<rect x="13" y="12" width="2" height="2" fill="#FF0055" class="bot-scanner" />

					<!-- Bot ID -->
					<text x="14" y="18" text-anchor="middle" dominant-baseline="central" font-size="5" font-weight="900" fill={color} font-family="monospace">{bot.id}</text>
				</g>
			</svg>

			<!-- Inventory indicator (dots below bot cell, relative to group origin) -->
			{#each bot.inventory as invItem, i}
				<circle
					cx={cellSize / 2 - (bot.inventory.length - 1) * 3 + i * 6}
					cy={cellSize + 4}
					r="2.5"
					fill={ITEM_COLORS[invItem] || '#aaa'}
					stroke="#000"
					stroke-width="0.5"
				/>
			{/each}
		</g>
	{/each}
</svg>

<style>
	/* === ACTIVE/PREVIEW ITEM GLOW === */
	.anim-active-glow {
		animation: activeGlowPulse 1.5s ease-in-out infinite;
		filter: drop-shadow(0 0 4px rgba(250, 204, 21, 0.5));
	}
	@keyframes activeGlowPulse {
		0%, 100% { opacity: 0.7; stroke-width: 2; }
		50% { opacity: 1; stroke-width: 3; }
	}
	.anim-preview-glow {
		animation: previewGlowPulse 2s ease-in-out infinite;
		filter: drop-shadow(0 0 3px rgba(244, 114, 182, 0.4));
	}
	@keyframes previewGlowPulse {
		0%, 100% { opacity: 0.65; stroke-width: 1.5; }
		50% { opacity: 1; stroke-width: 2.5; }
	}

	/* === CYBERPUNK ITEM ANIMATIONS === */

	/* Shadow breathing pulse */
	.cyber-shadow {
		animation: cyberShadowPulse 3s infinite ease-in-out;
		transform-origin: 14px 23px;
	}
	@keyframes cyberShadowPulse {
		0%, 100% { opacity: 0.15; transform: scale(1); }
		50% { opacity: 0.25; transform: scale(1.1); }
	}

	/* 1. Milk — Data flow (marching ants) */
	.cyber-flow {
		stroke-dasharray: 4 4;
		animation: cyberDashFlow 1.5s linear infinite;
	}
	@keyframes cyberDashFlow { to { stroke-dashoffset: -12; } }

	/* Milk + Fallback — Blinking status node */
	.cyber-blink {
		animation: cyberBlinkNode 1s steps(2, start) infinite;
	}
	@keyframes cyberBlinkNode {
		0%, 100% { opacity: 1; }
		50% { opacity: 0; }
	}

	/* 2. Bread — Neon flicker */
	.cyber-flicker {
		animation: cyberNeonFlicker 4s infinite;
	}
	@keyframes cyberNeonFlicker {
		0%, 18%, 22%, 25%, 53%, 57%, 100% { opacity: 1; }
		20%, 24%, 55% { opacity: 0.2; }
	}

	/* Bread — Score flow */
	.cyber-score {
		stroke-dasharray: 2 2;
		animation: cyberFlowUp 1.5s linear infinite;
	}
	@keyframes cyberFlowUp { to { stroke-dashoffset: -10; } }

	/* 3. Eggs — Sequential node pulse */
	.cyber-node-1 { animation: cyberNodePulse 1.5s infinite 0s; }
	.cyber-node-2 { animation: cyberNodePulse 1.5s infinite 0.5s; }
	.cyber-node-3 { animation: cyberNodePulse 1.5s infinite 1s; }
	@keyframes cyberNodePulse {
		0%, 100% { fill: #0DF0E3; opacity: 0.3; }
		50% { fill: #FF0055; opacity: 1; }
	}

	/* 4. Butter — Laser scan pass */
	.cyber-scan {
		animation: cyberScanPass 2.5s infinite linear;
	}
	@keyframes cyberScanPass {
		0%, 10% { transform: translateY(8px); opacity: 0; }
		15%, 85% { opacity: 1; }
		90%, 100% { transform: translateY(22px); opacity: 0; }
	}

	/* 5. Cheese — Chromatic glitch */
	.cyber-glitch {
		animation: cyberGlitchMove 3s infinite steps(1);
	}
	@keyframes cyberGlitchMove {
		0%, 93% { transform: translate(0, 0); opacity: 1; }
		94% { transform: translate(-1.5px, 0px); }
		96% { transform: translate(1.5px, 0px); }
		98% { transform: translate(0px, 1.5px); }
		100% { transform: translate(0, 0); }
	}

	/* 6. Pasta — Robotic core spin */
	.cyber-core {
		animation: cyberCoreSpin 2.5s cubic-bezier(0.6, -0.28, 0.735, 0.045) infinite;
		transform-origin: 14px 14px;
	}
	@keyframes cyberCoreSpin {
		0% { transform: rotate(0deg); }
		10%, 50% { transform: rotate(90deg); }
		60%, 100% { transform: rotate(180deg); }
	}

	/* 7. Rice — Terminal cursor blink */
	.cyber-cursor {
		animation: cyberTermBlink 0.9s infinite;
	}
	@keyframes cyberTermBlink {
		0%, 50% { opacity: 1; }
		51%, 100% { opacity: 0; }
	}

	/* 8. Juice — Battery charge level */
	.cyber-battery {
		animation: cyberChargeLevel 3s steps(8) infinite;
		transform-origin: center bottom;
	}
	@keyframes cyberChargeLevel {
		0% { transform: scaleY(0); opacity: 0.8; fill: #FF0055; }
		50% { fill: #0DF0E3; }
		80%, 100% { transform: scaleY(1); opacity: 1; fill: #00FF41; }
	}

	/* 9. Yogurt — Radar waves */
	.cyber-wave-1 {
		animation: cyberRadarWave 2s infinite 0s ease-out;
		transform-origin: 14px 12px;
	}
	.cyber-wave-2 {
		animation: cyberRadarWave 2s infinite 1s ease-out;
		transform-origin: 14px 12px;
	}
	@keyframes cyberRadarWave {
		0% { transform: scale(1); opacity: 0.8; stroke-width: 1; }
		100% { transform: scale(1.6); opacity: 0; stroke-width: 0.1; }
	}

	/* 10. Cereal — Floating hex bits */
	.cyber-hex-1 { animation: cyberFloatHex 2s infinite linear 0s; }
	.cyber-hex-2 { animation: cyberFloatHex 2s infinite linear 0.5s; }
	.cyber-hex-3 { animation: cyberFloatHex 2s infinite linear 1s; }
	.cyber-hex-4 { animation: cyberFloatHex 2s infinite linear 1.5s; }
	@keyframes cyberFloatHex {
		0% { transform: translateY(0) scale(1); opacity: 1; }
		100% { transform: translateY(-8px) scale(0.5); opacity: 0; }
	}

	/* 11. Flour — Zip extraction lines */
	.cyber-extract {
		stroke-dasharray: 2 4;
		animation: cyberZipExtract 1.5s linear infinite;
	}
	@keyframes cyberZipExtract { to { stroke-dashoffset: -12; } }

	/* 12. Sugar — Authorization cube sequence */
	.cyber-cube-1 { animation: cyberAuthCube 1.5s infinite 0s; transform-origin: center; }
	.cyber-cube-2 { animation: cyberAuthCube 1.5s infinite 0.5s; transform-origin: center; }
	.cyber-cube-3 { animation: cyberAuthCube 1.5s infinite 1s; transform-origin: center; }
	@keyframes cyberAuthCube {
		0%, 100% { transform: scale(1); stroke: #00FF41; fill: #161B22; }
		50% { transform: scale(1.15); stroke: #0DF0E3; fill: #0DF0E3; opacity: 0.9; }
	}

	/* 13. Coffee — Digital exhaust */
	.cyber-exhaust-1 { animation: cyberCodeExhaust 1.5s infinite 0s linear; }
	.cyber-exhaust-2 { animation: cyberCodeExhaust 1.5s infinite 0.75s linear; }
	@keyframes cyberCodeExhaust {
		0% { transform: translateY(2px); opacity: 0; }
		50% { opacity: 1; }
		100% { transform: translateY(-6px); opacity: 0; }
	}

	/* 14. Tea — Tag scanner */
	.cyber-tag-scan {
		animation: cyberTagScan 1.5s infinite linear;
	}
	@keyframes cyberTagScan {
		0% { transform: translateX(-1px); opacity: 0; }
		50% { opacity: 1; }
		100% { transform: translateX(5px); opacity: 0; }
	}

	/* 15. Oil — Coolant bubbles */
	.cyber-bubble-1 { animation: cyberBubbleRise 2s infinite 0s ease-in; }
	.cyber-bubble-2 { animation: cyberBubbleRise 2s infinite 0.6s ease-in; }
	.cyber-bubble-3 { animation: cyberBubbleRise 2s infinite 1.2s ease-in; }
	@keyframes cyberBubbleRise {
		0% { transform: translateY(6px) scale(0.5); opacity: 0; }
		20% { opacity: 1; }
		100% { transform: translateY(-4px) scale(1.2); opacity: 0; }
	}

	/* 16. Salt — Matrix rain */
	.cyber-rain-1 { animation: cyberMatrixRain 1.5s infinite 0s linear; }
	.cyber-rain-2 { animation: cyberMatrixRain 1.5s infinite 0.4s linear; }
	.cyber-rain-3 { animation: cyberMatrixRain 1.5s infinite 0.8s linear; }
	@keyframes cyberMatrixRain {
		0% { transform: translateY(-3px); opacity: 0; }
		20%, 80% { opacity: 1; }
		100% { transform: translateY(6px); opacity: 0; }
	}

	/* Cream — Data Drip */
	.cyber-drip-1 { animation: cyberDripFall 1.5s infinite 0s linear; }
	.cyber-drip-2 { animation: cyberDripFall 1.5s infinite 0.75s linear; }
	@keyframes cyberDripFall {
		0% { transform: translateY(0); opacity: 1; }
		70% { transform: translateY(6px); opacity: 0; }
		100% { transform: translateY(6px); opacity: 0; }
	}

	/* Oats — Fiber Flow + Node Blink */
	.cyber-fiber { stroke-dasharray: 2 2; animation: cyberFiberFlow 1s linear infinite; }
	@keyframes cyberFiberFlow { to { stroke-dashoffset: -8; } }
	.cyber-node-blink { animation: cyberNodeBlink 1.5s infinite steps(2, start); }
	@keyframes cyberNodeBlink { 0%, 100% { fill: #00FF41; } 50% { fill: #FF0055; } }

	/* Cyber Drone bot animations */
	.bot-cyber-hover {
		animation: cyberBotHover 3s ease-in-out infinite;
	}
	@keyframes cyberBotHover {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-3px); }
	}

	.bot-scanner {
		animation: cyberScanEye 2s ease-in-out infinite;
	}
	@keyframes cyberScanEye {
		0%, 10% { transform: translateX(-2.5px); }
		40%, 60% { transform: translateX(2.5px); }
		90%, 100% { transform: translateX(-2.5px); }
	}

	.bot-exhaust {
		animation: cyberExhaustFlicker 0.1s infinite;
		transform-origin: top;
		transform-box: fill-box;
	}
	@keyframes cyberExhaustFlicker {
		0%, 100% { opacity: 0.8; transform: scaleY(1); }
		50% { opacity: 0.3; transform: scaleY(0.5); }
	}

	.bot-antenna-node {
		animation: cyberNodeBlink 1.5s infinite steps(2, start);
	}
	@keyframes cyberNodeBlink {
		0%, 100% { fill: var(--bot-color, #00FF41); }
		50% { fill: #FF0055; }
	}

	/* Spawn pad animations */
	.anim-pad-rotate {
		transform-origin: 14px 14px;
		animation: padRotate 6s linear infinite;
	}
	.anim-pad-rotate-reverse {
		transform-origin: 14px 14px;
		animation: padRotateReverse 4s linear infinite;
	}
	.anim-pad-pulse {
		transform-origin: 14px 14px;
		animation: padPulse 2s ease-out infinite;
	}

	/* Drop-off hatch animations */
	.anim-hatch-scan {
		animation: hatchScan 2s ease-in-out infinite;
	}
	.anim-arrow-flash-1 { animation: arrowFlash 1.5s infinite; }
	.anim-arrow-flash-2 { animation: arrowFlash 1.5s infinite 0.75s; }

	@keyframes padRotate {
		100% { transform: rotate(360deg); }
	}
	@keyframes padRotateReverse {
		100% { transform: rotate(-360deg); }
	}
	@keyframes padPulse {
		0% { transform: scale(0.6); opacity: 0.8; }
		100% { transform: scale(1.8); opacity: 0; }
	}
	@keyframes hatchScan {
		0%, 100% { transform: translateY(0); opacity: 0; }
		10%, 90% { opacity: 1; }
		50% { transform: translateY(12px); }
	}
	@keyframes arrowFlash {
		0%, 100% { opacity: 0.2; }
		50% { opacity: 1; }
	}
</style>
