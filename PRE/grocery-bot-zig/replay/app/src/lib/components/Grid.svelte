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
	} = $props();

	const ITEM_COLORS = {
		milk: '#dfe6e9', bread: '#ffeaa7', eggs: '#fab1a0', butter: '#fdcb6e',
		cheese: '#f39c12', pasta: '#e17055', rice: '#dfe6e9', juice: '#74b9ff',
		yogurt: '#a29bfe', cereal: '#e67e22', flour: '#b2bec3', sugar: '#dfe6e9',
		coffee: '#6d4c2a', tea: '#00b894', oil: '#fdcb6e', salt: '#b2bec3',
	};

	// Animation class per item type
	const ITEM_ANIM = {
		milk: 'anim-float', bread: 'anim-pulse', eggs: 'anim-wiggle', butter: 'anim-float',
		cheese: 'anim-pulse', pasta: 'anim-wiggle', rice: 'anim-float', juice: 'anim-float',
		yogurt: 'anim-pulse', cereal: 'anim-wiggle', flour: 'anim-pulse', sugar: 'anim-float',
		coffee: 'anim-float', tea: 'anim-wiggle', oil: 'anim-pulse', salt: 'anim-wiggle',
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
				// Perimeter walls: darker gray. Interior aisle walls: brownish
				if (x === 0 || x === width - 1 || y === 0 || y === height - 1) return '#4a5568';
				return '#5d4037';
			case 'shelf': return '#6d4c2a';
			case 'dropoff': return '#00b89440';
			case 'spawn': return '#0984e340';
			default: return '#1e272e';
		}
	}

	function cellStroke(type, x, y) {
		switch (type) {
			case 'wall':
				if (x === 0 || x === width - 1 || y === 0 || y === height - 1) return '#5a6577';
				return '#7b5b3a';
			case 'dropoff': return '#00b894';
			case 'spawn': return '#0984e3';
			default: return '#2a2e3d';
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
			<path d="M0 6L6 0" stroke="#00b89444" stroke-width="1"/>
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
				<rect x="1" y="1" width="26" height="26" fill="#00b894" opacity="0.15" />
				<rect x="1" y="1" width="26" height="26" fill="none" stroke="#00b894" stroke-width="1" stroke-dasharray="2 2" opacity="0.5" />
				<rect x="2" y="2" width="4" height="4" fill="#00b894" opacity="0.8" />
				<rect x="22" y="2" width="4" height="4" fill="#00b894" opacity="0.8" />
				<rect x="2" y="22" width="4" height="4" fill="#00b894" opacity="0.8" />
				<rect x="22" y="22" width="4" height="4" fill="#00b894" opacity="0.8" />
				<path d="M 11 4 L 14 6 L 17 4" fill="none" stroke="#55efc4" stroke-width="1.5" class="anim-arrow-flash-1" stroke-linejoin="round" />
				<path d="M 11 24 L 14 22 L 17 24" fill="none" stroke="#55efc4" stroke-width="1.5" class="anim-arrow-flash-1" stroke-linejoin="round" />
				<path d="M 4 11 L 6 14 L 4 17" fill="none" stroke="#55efc4" stroke-width="1.5" class="anim-arrow-flash-2" stroke-linejoin="round" />
				<path d="M 24 11 L 22 14 L 24 17" fill="none" stroke="#55efc4" stroke-width="1.5" class="anim-arrow-flash-2" stroke-linejoin="round" />
				<rect x="8" y="8" width="12" height="12" fill="#000000" opacity="0.6" rx="1" />
				<rect x="8" y="8" width="12" height="12" fill="none" stroke="#00b894" stroke-width="1" rx="1" />
				<line x1="8" y1="8" x2="20" y2="8" stroke="#55efc4" stroke-width="1.5" class="anim-hatch-scan" filter="url(#bot-glow)"/>
				<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="900" fill="#ffffff" font-family="sans-serif">D</text>
			</svg>
		{/if}
		{#if cell.type === 'spawn'}
			<svg x={cell.x * cellSize} y={cell.y * cellSize} width={cellSize} height={cellSize} viewBox="0 0 28 28">
				<rect x="1" y="1" width="26" height="26" fill="#0984e3" opacity="0.15" />
				<path d="M 0 6 V 0 H 6" fill="none" stroke="#0984e3" stroke-width="1.5" opacity="0.8"/>
				<path d="M 28 6 V 0 H 22" fill="none" stroke="#0984e3" stroke-width="1.5" opacity="0.8"/>
				<path d="M 0 22 V 28 H 6" fill="none" stroke="#0984e3" stroke-width="1.5" opacity="0.8"/>
				<path d="M 28 22 V 28 H 22" fill="none" stroke="#0984e3" stroke-width="1.5" opacity="0.8"/>
				<circle cx="14" cy="14" r="6" fill="none" stroke="#0984e3" stroke-width="1" class="anim-pad-pulse" />
				<circle cx="14" cy="14" r="10" fill="none" stroke="#0984e3" stroke-width="1.5" stroke-dasharray="4 4" class="anim-pad-rotate" opacity="0.6"/>
				<circle cx="14" cy="14" r="7" fill="none" stroke="#74b9ff" stroke-width="1" stroke-dasharray="2 2" class="anim-pad-rotate-reverse" opacity="0.8"/>
				<circle cx="14" cy="14" r="5" fill="#0984e3" opacity="0.3" />
				<circle cx="14" cy="14" r="5" fill="none" stroke="#0984e3" stroke-width="1" />
				<text x="14" y="15" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="900" fill="#ffffff" font-family="sans-serif">S</text>
			</svg>
		{/if}
	{/each}

	<!-- Animated items on shelves -->
	{#each shelfItems as si}
		<svg
			x={si.x * cellSize}
			y={si.y * cellSize}
			width={cellSize}
			height={cellSize}
			viewBox="0 0 28 28"
		>
			<ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#000" opacity="0.15" />
			{#if si.type === 'milk'}
				<g class="item-anim anim-float">
					<path d="M10 11 L14 7 L18 11 V21 C18 21.5 17.5 22 17 22 H11 C10.5 22 10 21.5 10 21 Z" fill="#dfe6e9" />
					<path d="M9 11 H19" stroke="#b2bec3" stroke-width="1.5" stroke-linecap="round"/>
					<rect x="10" y="14" width="8" height="4" fill="#74b9ff"/>
				</g>
			{:else if si.type === 'bread'}
				<g class="item-anim anim-pulse">
					<path d="M 6 15 C 6 10, 22 10, 22 15 L 21 21 C 21 21.5, 20.5 22, 20 22 H 8 C 7.5 22, 7 21.5, 7 21 Z" fill="#ffeaa7" stroke="#e17055" stroke-width="1.5"/>
					<path d="M 10 13 V 20 M 14 13 V 20 M 18 13 V 20" stroke="#e17055" stroke-width="1" opacity="0.6" stroke-linecap="round"/>
				</g>
			{:else if si.type === 'eggs'}
				<g class="item-anim anim-wiggle">
					<ellipse cx="10" cy="15" rx="3" ry="4" fill="#fab1a0"/>
					<ellipse cx="14" cy="14" rx="3" ry="4" fill="#fab1a0"/>
					<ellipse cx="18" cy="15" rx="3" ry="4" fill="#fab1a0"/>
					<path d="M 5 17 L 6 21 C 6 21.5, 6.5 22, 7 22 H 21 C 21.5 22, 22 21.5, 22 21 L 23 17 Z" fill="#b2bec3"/>
				</g>
			{:else if si.type === 'butter'}
				<g class="item-anim anim-float">
					<path d="M 7 14 L 17 11 L 21 13 L 11 16 Z" fill="#fdcb6e"/>
					<path d="M 7 14 L 11 16 L 11 20 L 7 18 Z" fill="#f39c12"/>
					<path d="M 11 16 L 21 13 L 21 17 L 11 20 Z" fill="#e67e22"/>
					<path d="M 7 14 L 13 12.5 L 13 16.5 L 7 18 Z" fill="#dfe6e9"/>
				</g>
			{:else if si.type === 'cheese'}
				<g class="item-anim anim-pulse">
					<path d="M 7 19 L 21 21 L 18 10 Z" fill="#f39c12" stroke="#e67e22" stroke-width="1.5" stroke-linejoin="round"/>
					<circle cx="12" cy="17" r="1.5" fill="#e67e22"/>
					<circle cx="17" cy="17" r="2" fill="#e67e22"/>
					<circle cx="15" cy="13" r="1" fill="#e67e22"/>
				</g>
			{:else if si.type === 'pasta'}
				<g class="item-anim anim-wiggle">
					<path d="M 5 10 L 13 13 L 13 15 L 5 18 Z" fill="#e17055"/>
					<path d="M 23 10 L 15 13 L 15 15 L 23 18 Z" fill="#e17055"/>
					<rect x="13" y="12" width="2" height="4" rx="0.5" fill="#fdcb6e"/>
				</g>
			{:else if si.type === 'rice'}
				<g class="item-anim anim-float">
					<path d="M 9 12 C 7 12, 7 22, 10 22 H 18 C 21 22, 21 12, 19 12 Z" fill="#dfe6e9"/>
					<path d="M 12 8 L 14 11 L 16 8 Z" fill="#dfe6e9"/>
					<path d="M 12 11 H 16" stroke="#b2bec3" stroke-width="2" stroke-linecap="round"/>
					<text x="14" y="19" text-anchor="middle" font-size="7" font-weight="900" fill="#b2bec3" font-family="sans-serif">R</text>
				</g>
			{:else if si.type === 'juice'}
				<g class="item-anim anim-float">
					<rect x="9" y="10" width="10" height="12" rx="1.5" fill="#74b9ff"/>
					<path d="M 15 10 L 15 6 L 17 5" fill="none" stroke="#dfe6e9" stroke-width="1.5" stroke-linecap="round"/>
					<circle cx="14" cy="16" r="3" fill="#fdcb6e"/>
				</g>
			{:else if si.type === 'yogurt'}
				<g class="item-anim anim-pulse">
					<path d="M 9 12 L 10 21 C 10 21.5, 11 22, 14 22 C 17 22, 18 21.5, 18 21 L 19 12 Z" fill="#a29bfe"/>
					<ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="#dfe6e9"/>
				</g>
			{:else if si.type === 'cereal'}
				<g class="item-anim anim-wiggle">
					<rect x="9" y="8" width="10" height="14" rx="0.5" fill="#e67e22"/>
					<circle cx="14" cy="15" r="3" fill="#ffeaa7"/>
					<circle cx="14" cy="15" r="1.5" fill="#dfe6e9"/>
					<rect x="11" y="10" width="6" height="2" fill="#dfe6e9" opacity="0.5"/>
				</g>
			{:else if si.type === 'flour'}
				<g class="item-anim anim-pulse">
					<path d="M 9 10 L 19 10 L 19 21 C 19 21.5, 18.5 22, 18 22 L 10 22 C 9.5 22, 9 21.5, 9 21 Z" fill="#b2bec3"/>
					<path d="M 9 10 L 14 13 L 19 10 L 19 7 C 19 6.5, 18.5 6, 18 6 L 10 6 C 9.5 6, 9 6.5, 9 7 Z" fill="#dfe6e9"/>
					<path d="M 14 14 V 19 M 12 15 L 14 16 M 16 15 L 14 16 M 12 17 L 14 18 M 16 17 L 14 18" stroke="#6d4c2a" stroke-width="1" stroke-linecap="round" opacity="0.5"/>
				</g>
			{:else if si.type === 'sugar'}
				<g class="item-anim anim-float">
					<rect x="9" y="15" width="6" height="6" rx="0.5" fill="#dfe6e9" stroke="#b2bec3" stroke-width="0.5"/>
					<rect x="15" y="15" width="6" height="6" rx="0.5" fill="#dfe6e9" stroke="#b2bec3" stroke-width="0.5"/>
					<rect x="12" y="10" width="6" height="6" rx="0.5" fill="#ffffff" stroke="#b2bec3" stroke-width="0.5"/>
				</g>
			{:else if si.type === 'coffee'}
				<g class="item-anim anim-float">
					<path d="M 10 12 L 11 20 C 11 21.5, 12 22, 14 22 C 16 22, 17 21.5, 17 20 L 18 12 Z" fill="#6d4c2a"/>
					<path d="M 9.5 10 C 9.5 9, 18.5 9, 18.5 10 L 18.5 12 L 9.5 12 Z" fill="#dfe6e9"/>
					<path d="M 13 8 Q 11 6, 13 4 M 16 8 Q 18 6, 16 4" fill="none" stroke="#dfe6e9" stroke-width="1.5" stroke-linecap="round" opacity="0.8"/>
				</g>
			{:else if si.type === 'tea'}
				<g class="item-anim anim-wiggle">
					<path d="M 9 13 V 17 C 9 19, 11 21, 14 21 C 17 21, 19 19, 19 17 V 13 Z" fill="#00b894"/>
					<path d="M 19 14 H 21 C 22.5 14, 22.5 17, 21 17 H 19" fill="none" stroke="#00b894" stroke-width="2"/>
					<path d="M 14 13 Q 14 9, 11 9" fill="none" stroke="#b2bec3" stroke-width="1"/>
					<rect x="9" y="7" width="4" height="4" fill="#ffeaa7"/>
				</g>
			{:else if si.type === 'oil'}
				<g class="item-anim anim-pulse">
					<path d="M 12 11 C 10 14, 10 21, 10 21 C 10 21.5, 10.5 22, 11 22 H 17 C 17.5 22, 18 21.5, 18 21 C 18 21, 18 14, 16 11 V 8 H 12 Z" fill="#fdcb6e" stroke="#b2bec3" stroke-width="1"/>
					<rect x="12" y="6" width="4" height="2" rx="0.5" fill="#00b894"/>
					<path d="M 13 15 V 19" stroke="#ffeaa7" stroke-width="1.5" stroke-linecap="round"/>
				</g>
			{:else if si.type === 'salt'}
				<g class="item-anim anim-wiggle">
					<rect x="11" y="10" width="6" height="12" rx="1" fill="#b2bec3"/>
					<path d="M 11 10 C 11 7, 17 7, 17 10 Z" fill="#2d3436"/>
					<circle cx="13" cy="8.5" r="0.75" fill="#dfe6e9"/>
					<circle cx="15" cy="8.5" r="0.75" fill="#dfe6e9"/>
					<circle cx="14" cy="7.5" r="0.75" fill="#dfe6e9"/>
					<text x="14" y="18" text-anchor="middle" font-size="7" font-weight="900" fill="#2d3436" font-family="sans-serif">S</text>
				</g>
			{:else}
				<!-- Fallback: colored dot -->
				<g class="item-anim anim-float">
					<circle cx="14" cy="14" r="5" fill={ITEM_COLORS[si.type] || '#aaa'} opacity="0.8"/>
					<text x="14" y="15" text-anchor="middle" font-size="8" font-weight="700" fill="white">{si.type.charAt(0).toUpperCase()}</text>
				</g>
			{/if}
		</svg>
	{/each}

	<!-- Bots (animated hover-bot design) -->
	{#each bots as bot (bot.id)}
		{@const bx = bot.position[0] * cellSize}
		{@const by = bot.position[1] * cellSize}
		{@const color = botColors[bot.id % botColors.length]}
		{@const isSelected = selectedBot === bot.id}
		{@const delayClass = `delay-${bot.id % 4}`}

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
			>
				<!-- Floor shadow -->
				<ellipse cx="14" cy="25" rx="8" ry="2" fill="#000" opacity="0.3" />

				<!-- Animated robot body -->
				<g class="bot-anim-hover {delayClass}">
					<!-- Hover thruster base -->
					<path d="M 10 20 L 18 20 L 16 23 L 12 23 Z" fill="#b2bec3" />

					<!-- Antenna -->
					<line x1="14" y1="8" x2="14" y2="3" stroke="#b2bec3" stroke-width="1.5" />
					<circle cx="14" cy="3" r="1.5" fill={color} filter="url(#bot-glow)" />

					<!-- Side nodes / ears -->
					<rect x="4" y="11" width="3" height="6" rx="1" fill="#b2bec3" />
					<rect x="21" y="11" width="3" height="6" rx="1" fill="#b2bec3" />

					<!-- Main colored chassis -->
					<rect x="6" y="8" width="16" height="13" rx="3" fill={color} filter="url(#bot-glow)" />

					<!-- Visor / face -->
					<rect x="8" y="10" width="12" height="9" rx="1.5" fill="#1e272e" />

					<!-- Blinking eyes -->
					<circle cx="10" cy="12" r="0.75" fill="#fff" class="bot-anim-blink" />
					<circle cx="18" cy="12" r="0.75" fill="#fff" class="bot-anim-blink" />

					<!-- Bot ID number on screen -->
					<text x="14" y="15.5" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="900" fill="#ffffff" font-family="sans-serif">{bot.id}</text>
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
	/* Item animations */
	.item-anim {
		transform-origin: 14px 14px;
	}

	.anim-float { animation: itemFloat 2.5s ease-in-out infinite; }
	.anim-pulse { animation: itemPulse 2s ease-in-out infinite; }
	.anim-wiggle { animation: itemWiggle 3s ease-in-out infinite; }

	@keyframes itemFloat {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-3px); }
	}
	@keyframes itemPulse {
		0%, 100% { transform: scale(1); }
		50% { transform: scale(1.1); }
	}
	@keyframes itemWiggle {
		0%, 100% { transform: rotate(0deg); }
		25% { transform: rotate(-8deg); }
		75% { transform: rotate(8deg); }
	}

	/* Bot hover animation */
	.bot-anim-hover {
		transform-origin: 14px 14px;
		animation: botHover 2s ease-in-out infinite;
	}

	.delay-0 { animation-delay: 0s; }
	.delay-1 { animation-delay: 0.2s; }
	.delay-2 { animation-delay: 0.4s; }
	.delay-3 { animation-delay: 0.6s; }

	.bot-anim-blink {
		animation: botBlink 4s infinite;
	}

	@keyframes botHover {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-2.5px); }
	}

	@keyframes botBlink {
		0%, 88%, 100% { opacity: 1; }
		90%, 94% { opacity: 0; }
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
