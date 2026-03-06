<script>
	let {
		width,
		height,
		cellSize,
		wallSet,
		shelfSet,
		itemMap,
		dropOff,
		dropOffZones = null,
		spawn,
		bots,
		botPositions,
		botColors,
		selectedBot = null,
		onSelectBot = () => {},
		activeTypes = new Set(),
		previewTypes = new Set(),
		difficulty = null,
	} = $props();

	const nightmare = $derived(difficulty === 'nightmare');

	// 16:9 cell aspect for wider grid view, icons stay square (≈4:3 feel)
	const cellW = $derived(Math.round(cellSize * 4 / 3));
	const cellH = $derived(cellSize);
	const iconOff = $derived((cellW - cellH) / 2); // center icon in wider cell

	// Build dropoff set from zones (multi-dropoff) or single dropOff
	const dropOffSet = $derived.by(() => {
		const s = new Set();
		if (dropOffZones && dropOffZones.length > 0) {
			for (const dz of dropOffZones) {
				s.add(`${dz[0]},${dz[1]}`);
			}
		} else if (dropOff) {
			s.add(`${dropOff[0]},${dropOff[1]}`);
		}
		return s;
	});

	const ITEM_COLORS_CYBER = {
		milk: '#00FF41', bread: '#FF0055', eggs: '#FF0055', butter: '#0DF0E3',
		cheese: '#FF0055', pasta: '#00FF41', rice: '#0DF0E3', juice: '#00FF41',
		yogurt: '#0DF0E3', cereal: '#0DF0E3', flour: '#FF0055', sugar: '#0DF0E3',
		coffee: '#FF0055', tea: '#00FF41', oil: '#0DF0E3', salt: '#00FF41',
		cream: '#0DF0E3', oats: '#FF0055',
	};
	const ITEM_COLORS_NM = {
		milk: '#AA0000', bread: '#880000', eggs: '#EEDDDD', butter: '#AA8866',
		cheese: '#DDDDBB', pasta: '#226633', rice: '#DDCCBB', juice: '#44FF88',
		yogurt: '#CC0000', cereal: '#DDDDDD', flour: '#777777', sugar: '#99AAAA',
		cream: '#887777', oats: '#FF0000', apples: '#33FF00', bananas: '#556655',
		carrots: '#AA0000', lettuce: '#AA6677', onions: '#665544', peppers: '#FF6600',
		tomatoes: '#DDAA00', coffee: '#FFAA00', tea: '#33FF00', oil: '#DD77AA',
		salt: '#332211',
	};
	const ITEM_COLORS = $derived(nightmare ? ITEM_COLORS_NM : ITEM_COLORS_CYBER);

	function cellType(x, y) {
		const key = `${x},${y}`;
		if (dropOffSet.has(key)) return 'dropoff';
		if (x === spawn[0] && y === spawn[1]) return 'spawn';
		if (wallSet.has(key)) return 'wall';
		if (shelfSet.has(key)) return 'shelf';
		return 'floor';
	}

	function cellColor(type, x, y) {
		if (nightmare) {
			switch (type) {
				case 'wall':
					if (x === 0 || x === width - 1 || y === 0 || y === height - 1) return '#2a1010';
					return '#351515';
				case 'shelf': return '#1a0a0a';
				case 'dropoff': return '#33000020';
				case 'spawn': return '#44001530';
				default: return '#1a1818';
			}
		}
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
		if (nightmare) {
			switch (type) {
				case 'wall':
					if (x === 0 || x === width - 1 || y === 0 || y === height - 1) return '#3d1515';
					return '#4a1a1a';
				case 'shelf': return '#442222';
				case 'dropoff': return '#880000';
				case 'spawn': return '#AA0033';
				default: return '#2a2626';
			}
		}
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
	viewBox="0 0 {width * cellW} {height * cellH}"
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
			x={cell.x * cellW}
			y={cell.y * cellH}
			width={cellW}
			height={cellH}
			fill={cellColor(cell.type, cell.x, cell.y)}
			stroke={cellStroke(cell.type, cell.x, cell.y)}
			stroke-width={cell.type === 'dropoff' || cell.type === 'spawn' ? 2 : 0.5}
		/>
		{#if cell.type === 'dropoff'}
			<svg x={cell.x * cellW + iconOff} y={cell.y * cellH} width={cellH} height={cellH} viewBox="0 0 28 28">
				{#if nightmare}
					<!-- Flaming Pentagram Drop-Off -->
					<!-- bg removed for transparency -->
					<circle cx="14" cy="14" r="10" fill="none" stroke="#880000" stroke-width="1.5" opacity="0.4"/>
					<g class="nm-star-rotate">
						<path d="M 14 4 L 16.9 22.1 L 1.5 10.9 H 26.5 L 11.1 22.1 Z" fill="none" stroke="#880000" stroke-width="1" opacity="0.8"/>
					</g>
					<circle cx="14" cy="4" r="1.5" fill="#FA0" class="nm-flame-pulse"/>
					<circle cx="4.5" cy="10.9" r="1.5" fill="#FA0" class="nm-flame-pulse" style="animation-delay: 0.2s"/>
					<circle cx="23.5" cy="10.9" r="1.5" fill="#FA0" class="nm-flame-pulse" style="animation-delay: 0.4s"/>
					<circle cx="8.2" cy="22.1" r="1.5" fill="#FA0" class="nm-flame-pulse" style="animation-delay: 0.6s"/>
					<circle cx="19.8" cy="22.1" r="1.5" fill="#FA0" class="nm-flame-pulse" style="animation-delay: 0.8s"/>
				{:else}
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
				{/if}
			</svg>
		{/if}
		{#if cell.type === 'spawn'}
			<svg x={cell.x * cellW + iconOff} y={cell.y * cellH} width={cellH} height={cellH} viewBox="0 0 28 28">
				{#if nightmare}
					<!-- Nightmare Spawn — Summoning Circle -->
					<!-- bg removed for transparency -->
					<circle cx="14" cy="14" r="10" fill="none" stroke="#AA0033" stroke-width="1.5" stroke-dasharray="4 4" class="anim-pad-rotate" opacity="0.6"/>
					<circle cx="14" cy="14" r="7" fill="none" stroke="#AA0033" stroke-width="1" stroke-dasharray="2 2" class="anim-pad-rotate-reverse" opacity="0.8"/>
					<circle cx="14" cy="14" r="6" fill="none" stroke="#AA0033" stroke-width="1" class="anim-pad-pulse" />
					<circle cx="14" cy="14" r="5" fill="#AA0033" opacity="0.2" />
					<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="900" fill="#FF0000" font-family="sans-serif">S</text>
				{:else}
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
				{/if}
			</svg>
		{/if}
	{/each}

	<!-- Animated items on shelves -->
	{#each shelfItems as si}
		{@const isActive = activeTypes.has(si.type)}
		{@const isPreview = !isActive && previewTypes.has(si.type)}
		<!-- Active/preview highlight behind item -->
		{#if isActive}
			<rect
				x={si.x * cellW + iconOff - 2}
				y={si.y * cellH - 2}
				width={cellH + 4}
				height={cellH + 4}
				rx="3"
				fill={nightmare ? "rgba(255, 0, 0, 0.25)" : "rgba(250, 204, 21, 0.2)"}
				stroke={nightmare ? "#ff0000" : "#facc15"}
				stroke-width="2"
				opacity="0.9"
				class="anim-active-glow"
			/>
		{:else if isPreview}
			<rect
				x={si.x * cellW + iconOff - 2}
				y={si.y * cellH - 2}
				width={cellH + 4}
				height={cellH + 4}
				rx="3"
				fill={nightmare ? "rgba(170, 0, 50, 0.2)" : "rgba(244, 114, 182, 0.18)"}
				stroke={nightmare ? "#aa0033" : "#f472b6"}
				stroke-width="2"
				opacity="0.85"
				class="anim-preview-glow"
			/>
		{/if}
		<svg
			x={si.x * cellW + iconOff}
			y={si.y * cellH}
			width={cellH}
			height={cellH}
			viewBox="0 0 28 28"
		>
		{#if nightmare}
			<!-- === NIGHTMARE SATANIC ITEMS === -->
			<ellipse cx="14" cy="23" rx="7" ry="2" fill="#330000" class="nm-shadow-pulse"/>
			{#if si.type === 'milk'}
				<!-- Blood Chalice -->
				<path d="M 8 7 C 8 7, 8 13, 14 16 C 20 13, 20 7, 20 7 Z" fill="#1A1111" stroke="#AA8844" stroke-width="1.2"/>
				<path d="M 13 15.5 V 21 M 10 21 H 18" stroke="#AA8844" stroke-width="1.2" stroke-linecap="round"/>
				<ellipse cx="14" cy="7" rx="6" ry="2" fill="#800" stroke="#AA8844" stroke-width="1.2"/>
				<circle cx="14" cy="9" r="1.5" fill="#FF0000" class="nm-blood-drop"/>
				<path d="M 12 7 V 10" stroke="#FF0000" stroke-width="1.5" stroke-linecap="round"/>
			{:else if si.type === 'bread'}
				<!-- Necronomicon -->
				<g class="nm-breathe">
					<rect x="7" y="6" width="14" height="16" fill="#331A1A" stroke="#220505" stroke-width="1.2" rx="1"/>
					<path d="M 10 6 V 22 M 18 6 V 22" stroke="#552222" stroke-width="1"/>
					<ellipse cx="14" cy="14" rx="4" ry="2.5" fill="#110505" stroke="#880000" stroke-width="1.2"/>
					<ellipse cx="14" cy="14" rx="1.5" ry="2.5" fill="#FFCC00" class="nm-eye-blink"/>
					<circle cx="14" cy="14" r="0.8" fill="#000" class="nm-eye-blink"/>
					<path d="M 7 10 H 9 M 7 18 H 9 M 19 10 H 21 M 19 18 H 21" stroke="#000" stroke-width="1.2"/>
				</g>
			{:else if si.type === 'eggs'}
				<!-- Eyeball Cluster -->
				<g class="nm-twitch">
					<circle cx="11" cy="16" r="4.5" fill="#EEDDDD" stroke="#800" stroke-width="1"/>
					<circle cx="17" cy="15" r="4" fill="#EEDDDD" stroke="#800" stroke-width="1"/>
					<circle cx="14" cy="11" r="5" fill="#EEDDDD" stroke="#800" stroke-width="1"/>
					<circle cx="10" cy="16" r="1.5" fill="#000"/>
					<circle cx="18" cy="14" r="1" fill="#000"/>
					<circle cx="14" cy="10" r="1.8" fill="#000"/>
					<path d="M 14 6 C 13 8, 11 9, 10 10 M 17 11 C 18 12, 19 12, 20 13 M 9 14 C 8 15, 7 15, 6 16" stroke="#C00" stroke-width="0.5" fill="none"/>
				</g>
			{:else if si.type === 'butter'}
				<!-- Voodoo Doll -->
				<path d="M 14 6 C 12 6, 11 8, 11 10 C 11 11, 12 12, 14 12 C 16 12, 17 11, 17 10 C 17 8, 16 6, 14 6 Z M 11 12 H 17 V 18 H 11 Z M 11 12 L 8 15 M 17 12 L 20 15 M 12 18 L 12 22 M 16 18 L 16 22" fill="none" stroke="#A86" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
				<line x1="12" y1="9" x2="13" y2="10" stroke="#000" stroke-width="1"/><line x1="13" y1="9" x2="12" y2="10" stroke="#000" stroke-width="1"/>
				<line x1="15" y1="9" x2="16" y2="10" stroke="#000" stroke-width="1"/><line x1="16" y1="9" x2="15" y2="10" stroke="#000" stroke-width="1"/>
				<line x1="13" y1="15" x2="17" y2="15" stroke="#000" stroke-width="1" stroke-dasharray="1 1"/>
				<line x1="18" y1="10" x2="14" y2="14" stroke="#F00" stroke-width="1" class="nm-stab"/>
				<circle cx="18" cy="10" r="1.5" fill="#F00" class="nm-stab"/>
			{:else if si.type === 'cheese'}
				<!-- Cracked Skull -->
				<path d="M 9 10 C 9 6, 19 6, 19 10 C 19 14, 17 16, 17 16 H 11 C 11 16, 9 14, 9 10 Z" fill="#DDB" stroke="#332" stroke-width="1.2"/>
				<path d="M 12 17 V 19 H 16 V 17 Z" fill="#DDB" stroke="#332" stroke-width="1.2" class="nm-jaw"/>
				<line x1="14" y1="17" x2="14" y2="19" stroke="#332" stroke-width="1" class="nm-jaw"/>
				<ellipse cx="12" cy="11" rx="1.5" ry="2" fill="#050202"/>
				<ellipse cx="16" cy="11" rx="1.5" ry="2" fill="#050202"/>
				<path d="M 14 14 L 13 15 H 15 Z" fill="#050202"/>
				<circle cx="12" cy="11" r="0.5" fill="#F00" class="nm-eye-flicker"/>
				<path d="M 12 6 L 14 9 M 18 8 L 16 10" stroke="#050202" stroke-width="1"/>
			{:else if si.type === 'pasta'}
				<!-- Entangled Snakes -->
				<path class="nm-writhe" d="M 8 10 C 12 6, 16 14, 20 10 C 24 6, 12 20, 8 16 C 4 12, 10 20, 14 20" fill="none" stroke="#263" stroke-width="2.5" stroke-linecap="round"/>
				<circle cx="8" cy="10" r="1.5" fill="#132"/>
				<circle cx="14" cy="20" r="1.5" fill="#132"/>
				<path d="M 8 10 L 5 9 L 4 10 M 5 9 L 6 7" fill="none" stroke="#F00" stroke-width="0.8" class="nm-tongue"/>
				<circle cx="8" cy="10" r="0.5" fill="#F00"/>
				<circle cx="14" cy="20" r="0.5" fill="#F00"/>
			{:else if si.type === 'rice'}
				<!-- Squirming Maggots -->
				<g fill="#DDCCBB" stroke="#443322" stroke-width="1">
					<rect x="10" y="10" width="8" height="3" rx="1.5" class="nm-squirm1"/>
					<rect x="12" y="14" width="7" height="3" rx="1.5" class="nm-squirm2"/>
					<rect x="7" y="15" width="6" height="3" rx="1.5" transform="rotate(45 10 16)" class="nm-squirm1"/>
					<rect x="15" y="8" width="6" height="3" rx="1.5" transform="rotate(-30 18 9)" class="nm-squirm2"/>
					<rect x="11" y="18" width="7" height="3" rx="1.5" transform="rotate(-15 14 19)" class="nm-squirm1"/>
				</g>
			{:else if si.type === 'juice'}
				<!-- Head in a Jar -->
				<path d="M 8 7 H 20 V 23 H 8 Z" fill="#0A1A10" stroke="#335544" stroke-width="1.2"/>
				<path d="M 7 6 H 21 V 8 H 7 Z" fill="#112211" stroke="#335544" stroke-width="1.2"/>
				<g class="nm-bob">
					<ellipse cx="14" cy="14" rx="4" ry="5" fill="#667755" stroke="#223322" stroke-width="1"/>
					<path d="M 12 13 L 13 14 M 16 13 L 15 14" stroke="#111" stroke-width="1"/>
					<line x1="13" y1="17" x2="15" y2="17" stroke="#111" stroke-width="1"/>
				</g>
				<circle cx="10" cy="18" r="1" fill="#44FF88" class="nm-bubble1"/>
				<circle cx="18" cy="20" r="1" fill="#44FF88" class="nm-bubble2"/>
				<circle cx="12" cy="21" r="0.5" fill="#44FF88" class="nm-bubble1"/>
			{:else if si.type === 'yogurt'}
				<!-- Beating Heart -->
				<g class="nm-heartbeat">
					<path d="M 14 19 C 14 19, 8 14, 8 10 C 8 7, 11 6, 14 9 C 17 6, 20 7, 20 10 C 20 14, 14 19, 14 19 Z" fill="#800" stroke="#400" stroke-width="1.2"/>
					<path d="M 14 9 V 5 M 12 7 V 4 M 16 8 V 6" stroke="#400" stroke-width="1.5" stroke-linecap="round"/>
					<path d="M 14 9 C 12 12, 12 15, 14 19" fill="none" stroke="#500" stroke-width="1"/>
				</g>
			{:else if si.type === 'cereal'}
				<!-- Bowl of Bones & Teeth -->
				<path d="M 6 14 C 6 20, 22 20, 22 14 Z" fill="#222" stroke="#444" stroke-width="1.2"/>
				<ellipse cx="14" cy="14" rx="8" ry="2" fill="#111"/>
				<g fill="#DDD" stroke="#222" stroke-width="0.8">
					<rect x="10" y="11" width="4" height="2" rx="1" class="nm-float1" transform="rotate(20 12 12)"/>
					<rect x="15" y="10" width="4" height="2" rx="1" class="nm-float2" transform="rotate(-30 17 11)"/>
					<circle cx="13" cy="14" r="1.5" class="nm-float3"/>
					<circle cx="16" cy="13" r="1" class="nm-float1"/>
					<path class="nm-float2" d="M 13 9 C 13 8, 15 8, 15 9 V 11 C 15 12, 13 12, 13 11 Z" transform="rotate(45 14 10)"/>
				</g>
			{:else if si.type === 'flour'}
				<!-- Urn of Ashes -->
				<path d="M 12 6 H 16 L 15 8 C 18 10, 19 15, 17 20 H 11 C 9 15, 10 10, 13 8 Z" fill="#2A2A30" stroke="#111" stroke-width="1.2"/>
				<path d="M 10 14 C 14 16, 18 12, 18 14" fill="none" stroke="#111" stroke-width="1"/>
				<circle cx="14" cy="6" r="1" fill="#777" class="nm-ash-drift1"/>
				<circle cx="15" cy="5" r="0.8" fill="#555" class="nm-ash-drift2"/>
				<circle cx="13" cy="4" r="1.2" fill="#999" class="nm-ash-drift1" style="animation-delay: 1.2s"/>
			{:else if si.type === 'sugar'}
				<!-- Sacrificial Dagger -->
				<path d="M 14 2 L 15 6 L 14 18 L 13 6 Z" fill="#99A" stroke="#556" stroke-width="1" stroke-linejoin="round"/>
				<path d="M 11 18 H 17 V 19 H 11 Z" fill="#DA4"/>
				<path d="M 13 19 H 15 V 24 L 14 25 L 13 24 Z" fill="#311" stroke="#DA4" stroke-width="0.8"/>
				<circle cx="14" cy="21" r="1" fill="#F00"/>
				<path d="M 14 2 L 14 18" stroke="#FFF" stroke-width="0.5" class="nm-glint"/>
				<circle cx="14" cy="18" r="1" fill="#800" class="nm-dagger-drip"/>
			{:else if si.type === 'cream'}
				<!-- Twitching Severed Hand -->
				<g transform="rotate(-15 14 14)">
					<path d="M 6 16 C 6 12, 10 12, 12 12 H 16 V 20 H 12 C 10 20, 6 20, 6 16 Z" fill="#887777" stroke="#443333" stroke-width="1.2"/>
					<path d="M 16 13 H 20 C 21 13, 21 15, 20 15 H 16 Z" fill="#887777" stroke="#443333" stroke-width="1" class="nm-finger-twitch"/>
					<path d="M 16 15 H 22 C 23 15, 23 17, 22 17 H 16 Z" fill="#887777" stroke="#443333" stroke-width="1" class="nm-finger-twitch" style="animation-delay:0.1s"/>
					<path d="M 16 17 H 21 C 22 17, 22 19, 21 19 H 16 Z" fill="#887777" stroke="#443333" stroke-width="1" class="nm-finger-twitch" style="animation-delay:0.2s"/>
					<path d="M 13 18 L 15 22 C 16 23, 14 24, 13 22 L 11 19 Z" fill="#887777" stroke="#443333" stroke-width="1"/>
					<ellipse cx="6" cy="16" rx="1.5" ry="4" fill="#A00"/>
				</g>
				<circle cx="8" cy="21" r="1" fill="#A00"/>
				<circle cx="5" cy="19" r="1" fill="#A00"/>
			{:else if si.type === 'oats'}
				<!-- Glowing Pentagram Talisman -->
				<circle cx="14" cy="14" r="10" fill="none" stroke="#F00" stroke-width="2" class="nm-aura-pulse"/>
				<g class="nm-penta-spin">
					<circle cx="14" cy="14" r="8" fill="none" stroke="#800" stroke-width="1"/>
					<path d="M 14 6 L 16.5 20 L 4.5 11 H 23.5 L 11.5 20 Z" fill="none" stroke="#F00" stroke-width="1"/>
					<circle cx="14" cy="14" r="2" fill="#800"/>
				</g>
				<circle cx="14" cy="4" r="1" fill="#FA0"/><circle cx="4.5" cy="11" r="1" fill="#FA0"/>
				<circle cx="23.5" cy="11" r="1" fill="#FA0"/><circle cx="8.5" cy="22" r="1" fill="#FA0"/>
				<circle cx="19.5" cy="22" r="1" fill="#FA0"/>
			{:else if si.type === 'apples'}
				<!-- Rotting Black Apple with Corpse Worm -->
				<path d="M 14 8 C 21 8, 22 14, 20 19 C 19 22, 15 22, 14 20 C 13 22, 9 22, 8 19 C 6 14, 7 8, 14 8 Z" fill="#2A0505" stroke="#110202" stroke-width="1.2"/>
				<path d="M 14 8 Q 15 4 18 5" fill="none" stroke="#111" stroke-width="1.5"/>
				<ellipse cx="10" cy="14" rx="2" ry="3" fill="#050202"/>
				<g class="nm-wriggle">
					<path d="M 10 14 Q 5 10 7 18" fill="none" stroke="#3F0" stroke-width="2" stroke-linecap="round" class="nm-worm-glow"/>
					<circle cx="7" cy="18" r="1.5" fill="#3F0" class="nm-worm-glow"/>
					<circle cx="7.5" cy="18" r="0.5" fill="#000"/>
				</g>
			{:else if si.type === 'bananas'}
				<!-- Severed Demon Fingers -->
				<g class="nm-finger-twitch">
					<path d="M 17 20 Q 12 18 8 10 L 6 12 Q 10 21 16 22 Z" fill="#4A554A" stroke="#223322" stroke-width="1"/>
					<path d="M 18 20 Q 15 15 12 7 L 10 8 Q 14 18 17 22 Z" fill="#556655" stroke="#223322" stroke-width="1"/>
					<path d="M 19 20 Q 19 14 18 6 L 16 6 Q 17 15 18 22 Z" fill="#334433" stroke="#223322" stroke-width="1"/>
					<path d="M 8 10 L 5 7 L 6 12 Z M 12 7 L 10 3 L 10 8 Z M 18 6 L 19 2 L 16 6 Z" fill="#111"/>
					<path d="M 15 18 H 21 V 23 H 15 Z" fill="#600" stroke="#300" stroke-width="1"/>
					<path d="M 16 19 H 20 M 16 21 H 20" stroke="#200" stroke-width="1"/>
				</g>
			{:else if si.type === 'carrots'}
				<!-- Bloody Coffin Spikes -->
				<path d="M 8 6 H 12 L 10 20 Z" fill="#444" stroke="#222" stroke-width="1"/>
				<path d="M 16 4 H 20 L 18 22 Z" fill="#444" stroke="#222" stroke-width="1"/>
				<path d="M 12 8 H 16 L 14 24 Z" fill="#555" stroke="#222" stroke-width="1"/>
				<path d="M 8 6 H 12 V 8 H 8 Z M 16 4 H 20 V 6 H 16 Z M 12 8 H 16 V 10 H 12 Z" fill="#222"/>
				<path d="M 9 14 L 11 20 M 17 14 L 19 22 M 13 16 L 15 24" stroke="#A00" stroke-width="1.5"/>
				<circle cx="10" cy="20" r="1.5" fill="#A00" class="nm-blood-drop"/>
				<circle cx="18" cy="22" r="1.5" fill="#A00" class="nm-blood-drop" style="animation-delay:0.5s"/>
				<circle cx="14" cy="24" r="1.5" fill="#A00" class="nm-blood-drop" style="animation-delay:1s"/>
			{:else if si.type === 'lettuce'}
				<!-- Pulsating Exposed Brain -->
				<path d="M 14 22 C 6 22, 4 16, 6 10 C 8 6, 20 6, 22 10 C 24 16, 22 22, 14 22 Z" fill="#1A2A1A" stroke="#051105" stroke-width="1.5"/>
				<g class="nm-throb">
					<ellipse cx="14" cy="14" rx="6" ry="5" fill="#A67" stroke="#423" stroke-width="1.2"/>
					<path d="M 14 9 V 19 M 11 10 C 13 12, 9 14, 11 16 M 17 10 C 15 12, 19 14, 17 16" fill="none" stroke="#634" stroke-width="1.2" stroke-linecap="round"/>
					<circle cx="14" cy="14" r="5" fill="none" stroke="#F00" stroke-width="0.5" opacity="0.5"/>
				</g>
			{:else if si.type === 'onions'}
				<!-- Shrunken Head -->
				<g class="nm-swing">
					<line x1="14" y1="0" x2="14" y2="8" stroke="#543" stroke-width="1" stroke-dasharray="2 1"/>
					<ellipse cx="14" cy="14" rx="5" ry="6" fill="#654" stroke="#321" stroke-width="1.2"/>
					<path d="M 12 12 L 16 12" stroke="#210" stroke-width="1.5"/>
					<path d="M 11 11 L 13 13 M 15 11 L 17 13 M 13 11 L 11 13 M 17 11 L 15 13" stroke="#111" stroke-width="1"/>
					<path d="M 12 17 L 16 17" stroke="#111" stroke-width="1"/>
					<line x1="12.5" y1="16" x2="12.5" y2="18" stroke="#111" stroke-width="1"/>
					<line x1="14" y1="16" x2="14" y2="18" stroke="#111" stroke-width="1"/>
					<line x1="15.5" y1="16" x2="15.5" y2="18" stroke="#111" stroke-width="1"/>
					<path d="M 9 14 Q 7 18 8 22 M 19 14 Q 21 18 20 22 M 11 20 Q 14 24 17 20" fill="none" stroke="#111" stroke-width="1"/>
				</g>
			{:else if si.type === 'peppers'}
				<!-- Flaming Demon Heart -->
				<path class="nm-flame-flicker" d="M 14 4 Q 12 8 14 10 Q 16 8 14 4 Z" fill="#FA0"/>
				<g class="nm-beat-heat">
					<path d="M 14 20 C 14 20, 9 14, 9 10 C 9 7, 13 7, 14 10 C 15 7, 19 7, 19 10 C 19 14, 14 20, 14 20 Z" fill="#A00" stroke="#400" stroke-width="1.2"/>
					<path d="M 9 12 Q 14 15 19 11 M 10 16 Q 14 18 17 14" fill="none" stroke="#333" stroke-width="1.5"/>
					<circle cx="14" cy="14" r="1.5" fill="#333"/>
					<path d="M 14 14 L 12 12 M 14 14 L 16 16" stroke="#333" stroke-width="1.5"/>
				</g>
			{:else if si.type === 'tomatoes'}
				<!-- Veiny Bloodshot Eyeball -->
				<ellipse cx="14" cy="14" rx="7" ry="6.5" fill="#EEDDDD" stroke="#511" stroke-width="1.2"/>
				<path class="nm-vein-pulse" d="M 7 14 Q 10 12 11 14 M 21 14 Q 18 16 17 14 M 14 7 Q 16 10 14 11 M 14 21 Q 12 18 14 17" fill="none" stroke="#600" stroke-width="1"/>
				<g class="nm-eye-dart">
					<circle cx="14" cy="14" r="3" fill="#DA0"/>
					<ellipse cx="14" cy="14" rx="1" ry="2.5" fill="#000"/>
					<circle cx="15" cy="13" r="0.5" fill="#FFF"/>
				</g>
			{:else}
				<!-- Nightmare fallback -->
				<circle cx="14" cy="14" r="5" fill="none" stroke="#880000" stroke-width="1.2"/>
				<circle cx="14" cy="14" r="2" fill="#FF0000" class="nm-eye-flicker"/>
				<text x="14" y="15" text-anchor="middle" font-size="7" font-weight="bold" font-family="monospace" fill="#880000">{si.type.charAt(0).toUpperCase()}</text>
			{/if}
		{:else}
			<!-- === CYBERPUNK ITEMS === -->
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
		{/if}
		</svg>
	{/each}

	<!-- Bots (Cyber Drone design) -->
	{#each bots as bot (bot.id)}
		{@const bx = bot.position[0] * cellW}
		{@const by = bot.position[1] * cellH}
		{@const color = botColors[bot.id % botColors.length]}
		{@const isSelected = selectedBot === bot.id}

		<!-- Bot wrapper: instant position updates for real-time accuracy -->
		<g
			style="transform: translate({bx}px, {by}px); cursor: pointer;"
			onclick={() => onSelectBot(bot.id)}
		>
			<!-- Cell position highlight — always visible so you know which cell the bot occupies -->
			<rect
				x={0}
				y={0}
				width={cellW}
				height={cellH}
				fill={color}
				opacity={isSelected ? 0.15 : 0.08}
				stroke={color}
				stroke-width={isSelected ? 1.5 : 0.5}
				stroke-opacity={isSelected ? 0.6 : 0.3}
			/>

			<!-- Selection highlight -->
			{#if isSelected}
				<rect
					x={-2}
					y={-2}
					width={cellW + 4}
					height={cellH + 4}
					rx="1"
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
				x={iconOff}
				y={0}
				width={cellH}
				height={cellH}
				viewBox="0 0 28 28"
				style="--bot-color: {color}; --goat-color: {color};"
			>
			{#if nightmare}
				<!-- Nightmare Satanic Goat Bot -->
				<!-- bg removed for transparency -->
				<ellipse cx="14" cy="24" rx="7" ry="2" fill="#330000" class="nm-shadow-pulse"/>
				<g class="nm-float-goat">
					<!-- Faint Pentagram Background -->
					<path d="M14 5 L16 11 L22 11 L17 15 L19 21 L14 17 L9 21 L11 15 L6 11 L12 11 Z" fill="none" stroke="#FF0000" stroke-width="0.5" opacity="0.2"/>
					<!-- Horns -->
					<path d="M11 9 C8 4, 3 3, 2 8 C3 9, 6 8, 9 11 Z" fill="#110505" stroke={color} stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M17 9 C20 4, 25 3, 26 8 C25 9, 22 8, 19 11 Z" fill="#110505" stroke={color} stroke-width="1.2" stroke-linejoin="round"/>
					<!-- Ears -->
					<path d="M9 11.5 L3 15 L8 16 Z" fill="#110505" stroke={color} stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M19 11.5 L25 15 L20 16 Z" fill="#110505" stroke={color} stroke-width="1.2" stroke-linejoin="round"/>
					<!-- Demonic Head -->
					<path class="nm-breathe-snout" d="M9 11 L19 11 L16 20 L12 20 Z" fill="#0A0303" stroke={color} stroke-width="1.2" stroke-linejoin="round"/>
					<!-- Eyes -->
					<path d="M10 13 L12 14 L10 15 Z" fill="#FF0000" class="nm-eye-glow"/>
					<path d="M18 13 L16 14 L18 15 Z" fill="#FF0000" class="nm-eye-glow"/>
					<!-- Snout -->
					<path d="M13 18 L14 19 L15 18" fill="none" stroke={color} stroke-width="1" stroke-linecap="round"/>
					<!-- Bot ID -->
					<text x="14" y="25" text-anchor="middle" dominant-baseline="central" font-size="4.5" font-weight="900" fill={color} font-family="monospace">{bot.id}</text>
				</g>
			{:else}
				<!-- Cyberpunk Drone Bot -->
				<ellipse cx="14" cy="23" rx="6" ry="1.5" fill={color} class="cyber-shadow" />
				<g class="bot-cyber-hover">
					<path d="M 12 19 V 22 M 14 19 V 23 M 16 19 V 22" stroke="#0DF0E3" stroke-width="1.2" stroke-linecap="round" class="bot-exhaust"/>
					<line x1="14" y1="9" x2="14" y2="4" stroke={color} stroke-width="1.2" />
					<circle cx="14" cy="4" r="1.5" class="bot-antenna-node" fill={color} />
					<rect x="5.5" y="11" width="2" height="4" fill={color} opacity="0.5"/>
					<rect x="20.5" y="11" width="2" height="4" fill={color} opacity="0.5"/>
					<path d="M 9 9 L 19 9 L 21 15 L 7 15 Z" fill={color} opacity="0.7" stroke-linejoin="round"/>
					<path d="M 7 15 L 11 19 H 17 L 21 15" fill={color} opacity="0.5" stroke-linejoin="round"/>
					<rect x="10" y="11.5" width="8" height="3" fill="#0D1117" />
					<rect x="13" y="12" width="2" height="2" fill="#FF0055" class="bot-scanner" />
					<text x="14" y="18" text-anchor="middle" dominant-baseline="central" font-size="5" font-weight="900" fill={color} font-family="monospace">{bot.id}</text>
				</g>
			{/if}
			</svg>

			<!-- Inventory indicator (dots below bot cell, relative to group origin) -->
			{#each bot.inventory as invItem, i}
				<circle
					cx={cellW / 2 - (bot.inventory.length - 1) * 3 + i * 6}
					cy={cellH + 4}
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

	/* ================================================================ */
	/* === NIGHTMARE SATANIC ANIMATIONS === */
	/* ================================================================ */

	/* Shadow pulse under all nightmare items */
	.nm-shadow-pulse {
		animation: nmShadowPulse 3s infinite ease-in-out;
		transform-origin: 14px 24px;
	}
	@keyframes nmShadowPulse {
		0%, 100% { opacity: .2; transform: scale(1); }
		50% { opacity: .4; transform: scale(1.2); }
	}

	/* Goat bot floating */
	.nm-float-goat {
		animation: nmFloatGoat 3s infinite ease-in-out;
		transform-origin: center;
	}
	@keyframes nmFloatGoat {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-2px); }
	}

	/* Goat eye glow */
	.nm-eye-glow {
		animation: nmEyeGlow 1.5s infinite;
	}
	@keyframes nmEyeGlow {
		0%, 100% { fill: var(--goat-color); filter: drop-shadow(0 0 2px var(--goat-color)); }
		50% { fill: #FF0000; filter: drop-shadow(0 0 4px #FF0000); }
	}

	/* Goat snout breathing */
	.nm-breathe-snout {
		animation: nmBreatheSnout 2.5s infinite ease-in-out;
		transform-origin: 14px 15px;
	}
	@keyframes nmBreatheSnout {
		0%, 100% { transform: scaleX(1); }
		50% { transform: scaleX(1.05); }
	}

	/* Pentagram star rotation (dropoff) */
	.nm-star-rotate {
		animation: nmStarRotate 20s infinite linear;
		transform-origin: 14px 14px;
	}
	@keyframes nmStarRotate {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}

	/* Flame pulse on pentagram points */
	.nm-flame-pulse {
		animation: nmFlamePulse 1.5s infinite alternate;
		transform-origin: center;
	}
	@keyframes nmFlamePulse {
		0%, 100% { transform: scale(1); opacity: .8; }
		50% { transform: scale(1.3); opacity: 1; }
	}

	/* Blood drop fall */
	.nm-blood-drop {
		animation: nmBloodDrop 1.5s infinite linear;
	}
	@keyframes nmBloodDrop {
		0% { transform: translateY(0) scale(1); opacity: 1; }
		80% { transform: translateY(8px) scale(0.8); opacity: 1; }
		100% { transform: translateY(10px) scale(0); opacity: 0; }
	}

	/* Necronomicon breathing */
	.nm-breathe {
		animation: nmBreathe 2.5s infinite ease-in-out;
		transform-origin: center;
	}
	@keyframes nmBreathe {
		0%, 100% { transform: scale(1); }
		50% { transform: scale(1.03); }
	}

	/* Eye blink (necronomicon, cheese skull) */
	.nm-eye-blink {
		animation: nmEyeBlink 4s infinite;
		transform-origin: 14px 14px;
	}
	@keyframes nmEyeBlink {
		0%, 45%, 55%, 100% { transform: scaleY(1); }
		50% { transform: scaleY(0); }
	}

	/* Eyeball cluster twitch */
	.nm-twitch {
		animation: nmTwitch 3s infinite steps(2, start);
	}
	@keyframes nmTwitch {
		0%, 100% { transform: translate(0, 0); }
		10%, 30% { transform: translate(-1px, 1px); }
		20%, 40% { transform: translate(1px, -1px); }
		50% { transform: translate(0, 0); }
	}

	/* Voodoo doll stab */
	.nm-stab {
		animation: nmStab 2s infinite ease-in;
	}
	@keyframes nmStab {
		0%, 100% { transform: translate(4px, -4px); opacity: 0; }
		20%, 80% { transform: translate(0, 0); opacity: 1; }
	}

	/* Skull jaw */
	.nm-jaw {
		animation: nmJaw 1.5s infinite steps(2, start);
		transform-origin: center;
	}
	@keyframes nmJaw {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(2px); }
	}

	/* Skull eye flicker */
	.nm-eye-flicker {
		animation: nmEyeFlicker 3s infinite;
	}
	@keyframes nmEyeFlicker {
		0%, 100% { opacity: .2; }
		50% { opacity: 1; }
	}

	/* Snake writhe */
	.nm-writhe {
		stroke-dasharray: 6 2;
		animation: nmWrithe 2s infinite linear;
	}
	@keyframes nmWrithe {
		0%, 100% { stroke-dashoffset: 0; }
		50% { stroke-dashoffset: 4; }
	}

	/* Snake tongue */
	.nm-tongue {
		animation: nmTongue 1s infinite steps(2, start);
		transform-origin: left;
	}
	@keyframes nmTongue {
		0%, 100% { opacity: 0; transform: scaleX(0); }
		50% { opacity: 1; transform: scaleX(1); }
	}

	/* Maggot squirm */
	.nm-squirm1 {
		animation: nmSquirm1 1s infinite ease-in-out;
		transform-origin: center;
	}
	.nm-squirm2 {
		animation: nmSquirm2 1.2s infinite ease-in-out;
		transform-origin: center;
	}
	@keyframes nmSquirm1 {
		0%, 100% { transform: rotate(0); }
		50% { transform: rotate(15deg); }
	}
	@keyframes nmSquirm2 {
		0%, 100% { transform: rotate(0); }
		50% { transform: rotate(-15deg); }
	}

	/* Head in jar bob */
	.nm-bob {
		animation: nmBob 3s infinite ease-in-out;
	}
	@keyframes nmBob {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(2px); }
	}

	/* Jar bubbles */
	.nm-bubble1 {
		animation: nmBubble 2s infinite linear 0s;
	}
	.nm-bubble2 {
		animation: nmBubble 2s infinite linear 1s;
	}
	@keyframes nmBubble {
		0% { transform: translateY(0); opacity: 1; }
		100% { transform: translateY(-8px); opacity: 0; }
	}

	/* Heartbeat */
	.nm-heartbeat {
		animation: nmHeartBeat 1s infinite;
		transform-origin: 14px 14px;
	}
	@keyframes nmHeartBeat {
		0%, 100% { transform: scale(1); }
		15% { transform: scale(1.1); }
		30% { transform: scale(1); }
	}

	/* Bone float */
	.nm-float1 { animation: nmFloat 2s infinite ease-in-out 0s; }
	.nm-float2 { animation: nmFloat 2.2s infinite ease-in-out 0.5s; }
	.nm-float3 { animation: nmFloat 1.8s infinite ease-in-out 1s; }
	@keyframes nmFloat {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-2px); }
	}

	/* Ash drift */
	.nm-ash-drift1 { animation: nmAshDrift 2s infinite linear 0s; }
	.nm-ash-drift2 { animation: nmAshDrift 2s infinite linear 0.7s; }
	@keyframes nmAshDrift {
		0% { transform: translate(0, 0); opacity: 1; }
		100% { transform: translate(-4px, -8px) scale(2); opacity: 0; }
	}

	/* Dagger glint */
	.nm-glint {
		animation: nmGlint 2s infinite linear;
	}
	@keyframes nmGlint {
		0%, 80% { opacity: 0; }
		90% { opacity: 1; }
		100% { opacity: 0; }
	}

	/* Dagger blood drip */
	.nm-dagger-drip {
		animation: nmDaggerDrip 1.5s infinite ease-in;
	}
	@keyframes nmDaggerDrip {
		0% { transform: translateY(0); opacity: 1; }
		100% { transform: translateY(6px); opacity: 0; }
	}

	/* Severed hand finger twitch */
	.nm-finger-twitch {
		animation: nmFingerTwitch 2.5s infinite;
		transform-origin: 18px 20px;
	}
	@keyframes nmFingerTwitch {
		0%, 90% { transform: rotate(0); }
		95% { transform: rotate(-4deg); }
		100% { transform: rotate(2deg); }
	}

	/* Pentagram talisman aura */
	.nm-aura-pulse {
		animation: nmAuraPulse 3s infinite ease-in-out;
		transform-origin: center;
	}
	@keyframes nmAuraPulse {
		0%, 100% { opacity: .3; transform: scale(1); }
		50% { opacity: .6; transform: scale(1.1); }
	}

	/* Pentagram talisman spin */
	.nm-penta-spin {
		animation: nmPentaSpin 10s infinite linear;
		transform-origin: center;
	}
	@keyframes nmPentaSpin {
		0% { transform: rotate(0); }
		100% { transform: rotate(360deg); }
	}

	/* Apple worm wriggle */
	.nm-wriggle {
		animation: nmWriggle 1.5s infinite ease-in-out;
		transform-origin: 10px 14px;
	}
	@keyframes nmWriggle {
		0%, 100% { transform: rotate(-5deg); }
		50% { transform: rotate(10deg); }
	}

	/* Worm glow */
	.nm-worm-glow {
		animation: nmWormGlow 2s infinite;
	}
	@keyframes nmWormGlow {
		0%, 100% { filter: drop-shadow(0 0 2px #3F0); }
		50% { filter: drop-shadow(0 0 5px #3F0); }
	}

	/* Brain throb */
	.nm-throb {
		animation: nmThrob 1.2s infinite ease-in-out;
		transform-origin: 14px 14px;
	}
	@keyframes nmThrob {
		0%, 100% { transform: scale(1); }
		50% { transform: scale(1.05); }
	}

	/* Shrunken head swing */
	.nm-swing {
		animation: nmSwing 2.5s infinite ease-in-out;
		transform-origin: 14px 0px;
	}
	@keyframes nmSwing {
		0%, 100% { transform: rotate(-5deg); }
		50% { transform: rotate(5deg); }
	}

	/* Demon heart flame flicker */
	.nm-flame-flicker {
		animation: nmFlameFlicker 0.8s infinite alternate;
		transform-origin: 14px 8px;
	}
	@keyframes nmFlameFlicker {
		0%, 100% { transform: scaleY(1); opacity: .8; }
		50% { transform: scaleY(1.3); opacity: 1; }
	}

	/* Demon heart beat with heat */
	.nm-beat-heat {
		animation: nmBeatHeat 1.2s infinite;
		transform-origin: 14px 16px;
	}
	@keyframes nmBeatHeat {
		0%, 100% { transform: scale(1); }
		15% { transform: scale(1.1); }
		30% { transform: scale(1); }
	}

	/* Bloodshot eye vein pulse */
	.nm-vein-pulse {
		animation: nmVeinPulse 1.5s infinite;
	}
	@keyframes nmVeinPulse {
		0%, 100% { stroke: #600; }
		50% { stroke: #F00; }
	}

	/* Bloodshot eye dart */
	.nm-eye-dart {
		animation: nmEyeDart 3s infinite steps(2, start);
	}
	@keyframes nmEyeDart {
		0%, 10%, 100% { transform: translate(0, 0); }
		20%, 40% { transform: translate(-2px, 1px); }
		50%, 70% { transform: translate(2px, -1px); }
		80%, 90% { transform: translate(0, -2px); }
	}

	/* ================================================================ */

	/* Cyber Drone bot animations */
	.bot-cyber-hover {
		/* No hover animation — instant positions for accurate tracking */
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
