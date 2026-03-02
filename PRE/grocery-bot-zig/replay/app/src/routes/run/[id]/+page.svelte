<script>
	import Grid from '$lib/components/Grid.svelte';

	let { data } = $props();
	let run = $derived(data.run);
	let rounds = $derived(data.rounds);

	const CELL = 28;
	const BOT_COLORS = [
		'#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
		'#1abc9c', '#e67e22', '#34495e', '#e84393', '#00cec9',
	];

	// Build lookup sets
	const wallSet = new Set(run.walls.map(w => `${w[0]},${w[1]}`));
	const shelfSet = new Set(run.shelves.map(s => `${s[0]},${s[1]}`));
	const itemMap = new Map();
	for (const item of run.items) {
		const key = `${item.position[0]},${item.position[1]}`;
		if (!itemMap.has(key)) itemMap.set(key, []);
		itemMap.get(key).push(item);
	}

	// State
	let currentRound = $state(0);
	let playing = $state(false);
	let speed = $state(5);
	let selectedBot = $state(null);
	let intervalId = $state(null);

	let roundData = $derived(rounds[currentRound] || null);
	let bots = $derived(roundData?.bots || []);
	let orders = $derived(roundData?.orders || []);
	let actions = $derived(roundData?.actions || []);
	let events = $derived(roundData?.events || []);
	let score = $derived(roundData?.score || 0);

	let botPositions = $derived(new Map(bots.map(b => [`${b.position[0]},${b.position[1]}`, b])));

	// Build cumulative event history up to current round
	let eventHistory = $derived.by(() => {
		const history = [];
		for (let r = 0; r <= currentRound; r++) {
			const rd = rounds[r];
			if (!rd?.events) continue;
			for (const evt of rd.events) {
				history.push({ ...evt, round: rd.round_number });
			}
		}
		return history;
	});

	function play() {
		if (playing) return;
		playing = true;
		intervalId = setInterval(() => {
			if (currentRound < rounds.length - 1) {
				currentRound++;
			} else {
				pause();
			}
		}, 1000 / speed);
	}

	function pause() {
		playing = false;
		if (intervalId) {
			clearInterval(intervalId);
			intervalId = null;
		}
	}

	function stepForward() {
		pause();
		if (currentRound < rounds.length - 1) currentRound++;
	}

	function stepBack() {
		pause();
		if (currentRound > 0) currentRound--;
	}

	function setRound(r) {
		pause();
		currentRound = r;
	}

	function togglePlay() {
		if (playing) pause();
		else play();
	}

	// Restart play when speed changes
	$effect(() => {
		speed; // depend on speed
		if (playing) {
			if (intervalId) clearInterval(intervalId);
			intervalId = setInterval(() => {
				if (currentRound < rounds.length - 1) {
					currentRound++;
				} else {
					pause();
				}
			}, 1000 / speed);
		}
	});

	function getBotAction(botId) {
		const a = actions.find(a => a.bot === botId);
		return a ? a.action : 'wait';
	}

	function getItemTypeName(t) {
		return t.charAt(0).toUpperCase() + t.slice(1);
	}

	const diffColors = {
		easy: '#00b894',
		medium: '#fdcb6e',
		hard: '#e17055',
		expert: '#e74c3c',
	};

	// Keyboard controls
	function handleKeydown(e) {
		if (e.key === 'ArrowRight' || e.key === 'l') stepForward();
		else if (e.key === 'ArrowLeft' || e.key === 'h') stepBack();
		else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
		else if (e.key === 'Home') setRound(0);
		else if (e.key === 'End') setRound(rounds.length - 1);
	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="replay-page">
	<div class="top-bar">
		<a href="/" class="back-link">Back to runs</a>
		<div class="run-info">
			<span class="badge" style="background: {diffColors[run.difficulty]}22; color: {diffColors[run.difficulty]}; border: 1px solid {diffColors[run.difficulty]}44">
				{run.difficulty}
			</span>
			<span class="mono">Seed: {run.seed}</span>
			<span class="mono">{run.grid_width}x{run.grid_height}</span>
			<span>{run.bot_count} bots</span>
		</div>
		<div class="score-display">
			<span class="score-label">Score</span>
			<span class="score-value">{score}</span>
			<span class="score-final">/ {run.final_score}</span>
		</div>
	</div>

	<div class="main-area">
		<div class="grid-section">
			<div class="grid-container">
				<Grid
					width={run.grid_width}
					height={run.grid_height}
					cellSize={CELL}
					{wallSet}
					{shelfSet}
					{itemMap}
					dropOff={run.drop_off}
					spawn={run.spawn}
					{bots}
					{botPositions}
					botColors={BOT_COLORS}
					{selectedBot}
					onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
				/>
			</div>

			<!-- Controls -->
			<div class="controls card">
				<div class="controls-row">
					<button class="ctrl-btn" onclick={() => setRound(0)} title="Start (Home)">
						<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/></svg>
					</button>
					<button class="ctrl-btn" onclick={stepBack} title="Step back (Left/H)">
						<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M6 6h2v12H6zm3.5 6l8.5 6V6z"/></svg>
					</button>
					<button class="ctrl-btn play-btn" onclick={togglePlay} title="Play/Pause (Space)">
						{#if playing}
							<svg viewBox="0 0 24 24" width="22" height="22"><path fill="currentColor" d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
						{:else}
							<svg viewBox="0 0 24 24" width="22" height="22"><path fill="currentColor" d="M8 5v14l11-7z"/></svg>
						{/if}
					</button>
					<button class="ctrl-btn" onclick={stepForward} title="Step forward (Right/L)">
						<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/></svg>
					</button>
					<button class="ctrl-btn" onclick={() => setRound(rounds.length - 1)} title="End (End)">
						<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M6 18l8.5-6L6 6v12zM16 6v12h2V6h-2z"/></svg>
					</button>
				</div>

				<div class="round-slider">
					<span class="round-label">Round {currentRound}</span>
					<input type="range" min="0" max={rounds.length - 1} bind:value={currentRound} oninput={pause} />
					<span class="round-label">{rounds.length - 1}</span>
				</div>

				<div class="speed-control">
					<span class="speed-label">Speed:</span>
					{#each [1, 2, 5, 10, 30] as s}
						<button class="speed-btn" class:active={speed === s} onclick={() => speed = s}>{s}x</button>
					{/each}
				</div>
			</div>
		</div>

		<!-- Side panel -->
		<div class="side-panel">
			<!-- Orders -->
			<div class="panel-section card">
				<h3>Orders</h3>
				{#each orders as order}
					<div class="order" class:active-order={order.status === 'active'} class:preview-order={order.status === 'preview'}>
						<div class="order-header">
							<span class="order-status" class:active={order.status === 'active'}>{order.status}</span>
							<span class="mono order-id">{order.id}</span>
						</div>
						<div class="order-items">
							{#each order.items_required as item, i}
								<span class="order-item" class:delivered={order.items_delivered.includes(item) && order.items_delivered.filter(d => d === item).length > order.items_required.slice(0, i).filter(r => r === item && order.items_delivered.includes(r)).length}>
									{getItemTypeName(item)}
								</span>
							{/each}
						</div>
						<div class="order-progress">
							{order.items_delivered.length} / {order.items_required.length} delivered
						</div>
					</div>
				{/each}
			</div>

			<!-- Bots -->
			<div class="panel-section card">
				<h3>Bots</h3>
				{#each bots as bot}
					<button
						class="bot-row"
						class:selected={selectedBot === bot.id}
						onclick={() => selectedBot = selectedBot === bot.id ? null : bot.id}
						style="--bot-color: {BOT_COLORS[bot.id % BOT_COLORS.length]}"
					>
						<div class="bot-header">
							<span class="bot-marker" style="background: {BOT_COLORS[bot.id % BOT_COLORS.length]}">
								{bot.id}
							</span>
							<span class="bot-pos mono">({bot.position[0]}, {bot.position[1]})</span>
							<span class="bot-action">{getBotAction(bot.id)}</span>
						</div>
						<div class="bot-inv">
							{#if bot.inventory.length === 0}
								<span class="empty-inv">empty</span>
							{:else}
								{#each bot.inventory as item}
									<span class="inv-item">{getItemTypeName(item)}</span>
								{/each}
							{/if}
						</div>
					</button>
				{/each}
			</div>

			<!-- Event History -->
			<div class="panel-section card">
				<h3>Event Log ({eventHistory.length})</h3>
				<div class="event-log">
					{#each eventHistory.toReversed() as evt}
						<div class="event" class:current-round={evt.round === currentRound}>
							<span class="evt-round mono">R{evt.round}</span>
							{#if evt.type === 'pickup'}
								<span class="evt-icon pickup-icon">P</span>
								<span class="evt-text">Bot {evt.bot} picked {evt.item_type}</span>
							{:else if evt.type === 'deliver'}
								<span class="evt-icon deliver-icon">D</span>
								<span class="evt-text">Bot {evt.bot} delivered {evt.item_type}</span>
							{:else if evt.type === 'auto_deliver'}
								<span class="evt-icon auto-icon">A</span>
								<span class="evt-text">Bot {evt.bot} auto-delivered {evt.item_type}</span>
							{:else if evt.type === 'order_complete'}
								<span class="evt-icon complete-icon">!</span>
								<span class="evt-text">Order complete! (+5)</span>
							{/if}
						</div>
					{/each}
					{#if eventHistory.length === 0}
						<div class="no-events">No events yet</div>
					{/if}
				</div>
			</div>

			<!-- Map Legend -->
			<div class="panel-section card">
				<h3>Legend</h3>
				<div class="legend-grid">
					<div class="legend-item"><span class="legend-swatch" style="background: #2d3436"></span>Wall</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #6d4c2a"></span>Shelf</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #00b894"></span>Drop-off</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #0984e3"></span>Spawn</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #1e272e"></span>Floor</div>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
	.replay-page { display: flex; flex-direction: column; gap: 1rem; }

	.top-bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 1rem;
		padding: 0.75rem 1rem;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
	}
	.back-link { font-size: 0.85rem; }
	.run-info { display: flex; align-items: center; gap: 0.75rem; font-size: 0.85rem; }
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: capitalize;
	}
	.mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.8rem; }
	.score-display { display: flex; align-items: baseline; gap: 0.5rem; }
	.score-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }
	.score-value { font-size: 1.75rem; font-weight: 800; color: var(--green); }
	.score-final { font-size: 0.85rem; color: var(--text-muted); }

	.main-area {
		display: grid;
		grid-template-columns: 1fr 320px;
		gap: 1rem;
		align-items: start;
	}

	.grid-section { display: flex; flex-direction: column; gap: 1rem; }
	.grid-container {
		border: 1px solid var(--border);
		border-radius: var(--radius);
		background: var(--bg-card);
		padding: 0.5rem;
		width: 100%;
	}

	.controls {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		align-items: center;
	}
	.controls-row { display: flex; gap: 0.5rem; align-items: center; }
	.ctrl-btn {
		width: 36px;
		height: 36px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: var(--bg);
		color: var(--text-muted);
		border: 1px solid var(--border);
	}
	.ctrl-btn:hover { color: var(--text); background: var(--bg-hover); }
	.play-btn {
		width: 48px;
		height: 48px;
		background: var(--accent);
		color: white;
		border: none;
	}
	.play-btn:hover { background: var(--accent-light); }

	.round-slider {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		width: 100%;
	}
	.round-label { font-size: 0.8rem; color: var(--text-muted); white-space: nowrap; min-width: 65px; }
	input[type="range"] {
		flex: 1;
		height: 4px;
		-webkit-appearance: none;
		background: var(--border);
		border-radius: 2px;
		outline: none;
	}
	input[type="range"]::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--accent);
		cursor: pointer;
	}

	.speed-control { display: flex; align-items: center; gap: 0.35rem; }
	.speed-label { font-size: 0.75rem; color: var(--text-muted); margin-right: 0.25rem; }
	.speed-btn {
		padding: 0.2rem 0.5rem;
		background: var(--bg);
		color: var(--text-muted);
		border: 1px solid var(--border);
		font-size: 0.75rem;
	}
	.speed-btn.active {
		background: var(--accent)22;
		color: var(--accent-light);
		border-color: var(--accent);
	}

	.side-panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		max-height: calc(100vh - 150px);
		overflow-y: auto;
	}

	.panel-section h3 {
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
	}

	.order {
		padding: 0.5rem;
		border-radius: 6px;
		margin-bottom: 0.5rem;
		border: 1px solid var(--border);
	}
	.active-order { border-color: var(--green)44; background: var(--green)08; }
	.preview-order { border-color: var(--orange)44; background: var(--orange)08; }
	.order-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem; }
	.order-status {
		font-size: 0.65rem;
		text-transform: uppercase;
		font-weight: 700;
		padding: 0.1rem 0.35rem;
		border-radius: 3px;
		background: var(--orange)22;
		color: var(--orange);
	}
	.order-status.active { background: var(--green)22; color: var(--green); }
	.order-id { font-size: 0.7rem; color: var(--text-muted); }
	.order-items { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.25rem; }
	.order-item {
		font-size: 0.7rem;
		padding: 0.1rem 0.4rem;
		background: var(--bg);
		border-radius: 3px;
		border: 1px solid var(--border);
	}
	.order-item.delivered {
		background: var(--green)22;
		border-color: var(--green)44;
		color: var(--green);
		text-decoration: line-through;
	}
	.order-progress { font-size: 0.7rem; color: var(--text-muted); }

	.bot-row {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.5rem;
		background: transparent;
		border: 1px solid var(--border);
		border-radius: 6px;
		margin-bottom: 0.35rem;
		color: var(--text);
	}
	.bot-row:hover { background: var(--bg-hover); }
	.bot-row.selected { border-color: var(--bot-color); background: color-mix(in srgb, var(--bot-color) 10%, transparent); }
	.bot-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem; }
	.bot-marker {
		width: 22px;
		height: 22px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 0.7rem;
		font-weight: 700;
		color: white;
	}
	.bot-pos { font-size: 0.75rem; color: var(--text-muted); }
	.bot-action {
		margin-left: auto;
		font-size: 0.7rem;
		color: var(--accent-light);
		font-family: 'JetBrains Mono', monospace;
	}
	.bot-inv { display: flex; flex-wrap: wrap; gap: 0.2rem; margin-left: 1.75rem; }
	.empty-inv { font-size: 0.7rem; color: var(--text-muted); font-style: italic; }
	.inv-item {
		font-size: 0.65rem;
		padding: 0.05rem 0.35rem;
		background: var(--accent)18;
		border: 1px solid var(--accent)33;
		border-radius: 3px;
		color: var(--accent-light);
	}

	.event-log {
		max-height: 300px;
		overflow-y: auto;
	}
	.event {
		font-size: 0.75rem;
		padding: 0.25rem 0.35rem;
		display: flex;
		align-items: center;
		gap: 0.4rem;
		border-radius: 3px;
	}
	.event.current-round {
		background: var(--accent)15;
		border-left: 2px solid var(--accent);
	}
	.evt-round {
		font-size: 0.65rem;
		color: var(--text-muted);
		min-width: 28px;
	}
	.evt-text { flex: 1; }
	.evt-icon {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 0.6rem;
		font-weight: 700;
		color: white;
		flex-shrink: 0;
	}
	.pickup-icon { background: var(--blue); }
	.deliver-icon { background: var(--green); }
	.auto-icon { background: var(--orange); }
	.complete-icon { background: var(--red); }
	.no-events { font-size: 0.75rem; color: var(--text-muted); font-style: italic; padding: 0.5rem 0; }

	.legend-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; }
	.legend-item { display: flex; align-items: center; gap: 0.5rem; font-size: 0.75rem; color: var(--text-muted); }
	.legend-swatch { width: 16px; height: 16px; border-radius: 3px; border: 1px solid var(--border); }

	@media (max-width: 900px) {
		.main-area { grid-template-columns: 1fr; }
		.side-panel { max-height: none; }
	}
</style>
