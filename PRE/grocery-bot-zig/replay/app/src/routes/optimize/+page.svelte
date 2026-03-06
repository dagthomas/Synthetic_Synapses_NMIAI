<script>
	import Grid from '$lib/components/Grid.svelte';

	const CELL = 28;
	const BOT_COLORS = [
		'#f85149', '#58a6ff', '#39d353', '#d29922', '#bc8cff',
		'#3fb950', '#db6d28', '#8b949e', '#f778ba', '#79c0ff',
	];
	const diffColors = {
		easy: '#39d353',
		medium: '#d29922',
		hard: '#f85149',
		expert: '#da3633',
	};

	// Input state
	let wsUrl = $state('');
	let difficulty = $state('auto');
	let running = $state(false);
	let runningAction = $state(''); // 'play', 'learn', 'replay'
	let finished = $state(false);

	// Solution status
	let solutions = $state({ easy: null, medium: null, hard: null, expert: null });

	// Game state (for play/replay visualization)
	let gameInit = $state(null);
	let currentRound = $state(0);
	let maxRounds = $state(300);
	let score = $state(0);
	let finalScore = $state(null);
	let bots = $state([]);
	let orders = $state([]);
	let actions = $state([]);
	let logs = $state([]);

	// Learn state
	let learnTime = $state(120);
	let learnWorkers = $state(12);

	// Grid data
	let wallSet = $derived(gameInit ? new Set(gameInit.walls.map(w => `${w[0]},${w[1]}`)) : new Set());
	let shelfSet = $derived(gameInit ? new Set(gameInit.shelves.map(s => `${s[0]},${s[1]}`)) : new Set());
	let itemMap = $derived.by(() => {
		if (!gameInit) return new Map();
		const m = new Map();
		for (const item of gameInit.items) {
			const key = `${item.position[0]},${item.position[1]}`;
			if (!m.has(key)) m.set(key, []);
			m.get(key).push(item);
		}
		return m;
	});
	let botPositions = $derived(new Map(bots.map(b => [`${b.position[0]},${b.position[1]}`, b])));
	let selectedBot = $state(null);

	// Parse JWT from URL
	function parseToken(url) {
		try {
			const tokenMatch = url.match(/[?&]token=([^&]+)/);
			if (!tokenMatch) return null;
			const parts = tokenMatch[1].split('.');
			if (parts.length < 2) return null;
			const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
			return JSON.parse(atob(payload));
		} catch { return null; }
	}

	let tokenInfo = $derived.by(() => {
		if (!wsUrl.trim()) return null;
		return parseToken(wsUrl);
	});

	$effect(() => {
		if (tokenInfo?.difficulty) difficulty = tokenInfo.difficulty;
	});

	let abortController = $state(null);

	// Load solutions on mount
	async function loadSolutions() {
		try {
			const res = await fetch('/api/optimize/solutions');
			solutions = await res.json();
		} catch (e) {
			console.error('Failed to load solutions:', e);
		}
	}

	$effect(() => { loadSolutions(); });

	function resetGameState() {
		gameInit = null;
		currentRound = 0;
		score = 0;
		finalScore = null;
		bots = [];
		orders = [];
		actions = [];
		logs = [];
		finished = false;
	}

	// SSE reader shared by play/replay
	async function streamSSE(url, body, onEvent) {
		if (abortController) abortController.abort();
		abortController = new AbortController();

		const response = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body),
			signal: abortController.signal,
		});

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = '';

		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });
			const lines = buffer.split('\n');
			buffer = lines.pop() || '';

			for (const line of lines) {
				if (!line.startsWith('data: ')) continue;
				try {
					const event = JSON.parse(line.slice(6));
					onEvent(event);
				} catch (e) { // Stream closed — client disconnected
				}
			}
		}
	}

	function handleGameEvent(event) {
		switch (event.type) {
			case 'init':
				gameInit = event;
				maxRounds = event.max_rounds || 300;
				break;
			case 'round':
				currentRound = event.round;
				score = event.score || 0;
				bots = event.bots || [];
				orders = event.orders || [];
				break;
			case 'actions':
				actions = event.actions || [];
				break;
			case 'progress':
				currentRound = event.round;
				score = event.score || 0;
				break;
			case 'final_score':
				finalScore = event.score;
				break;
			case 'game_over':
				finalScore = event.score;
				finished = true;
				break;
			case 'log':
			case 'status':
				addLog(event.text || event.message);
				break;
			case 'done':
				running = false;
				finished = true;
				addLog(event.message || 'Done');
				loadSolutions();
				break;
			case 'error':
				addLog(`ERROR: ${event.message}`);
				break;
			case 'db':
				addLog(`DB: ${event.message}`);
				break;
		}
	}

	function handleLearnEvent(event) {
		switch (event.type) {
			case 'log':
			case 'status':
				addLog(event.text || event.message);
				break;
			case 'worker_result':
				addLog(`Worker ${event.worker} (mab=${event.mab}): planner=${event.planner} final=${event.final}`);
				break;
			case 'best':
				addLog(`BEST: Worker ${event.worker} (mab=${event.mab}) score=${event.score}`);
				break;
			case 'improved':
				addLog(`IMPROVED: ${event.old_score} -> ${event.new_score} (+${event.delta})`);
				break;
			case 'learn_done':
				addLog(`Optimization complete: score=${event.score} (was ${event.prev_score})`);
				break;
			case 'done':
				running = false;
				finished = true;
				addLog(event.message || 'Done');
				loadSolutions();
				break;
			case 'error':
				addLog(`ERROR: ${event.message}`);
				break;
		}
	}

	function addLog(text) {
		if (!text) return;
		logs = [...logs, { time: new Date().toLocaleTimeString(), text }];
	}

	// Actions
	async function playAndCapture() {
		if (!wsUrl.trim()) return;
		resetGameState();
		running = true;
		runningAction = 'play';

		try {
			const diff = difficulty === 'auto' ? (tokenInfo?.difficulty || 'medium') : difficulty;
			await streamSSE('/api/optimize/play', { url: wsUrl.trim(), difficulty: diff }, handleGameEvent);
		} catch (e) {
			if (e.name !== 'AbortError') addLog(`Error: ${e.message}`);
		}
		running = false;
	}

	async function learnFromCapture(diff) {
		resetGameState();
		running = true;
		runningAction = 'learn';
		addLog(`Starting optimization for ${diff}...`);

		try {
			await streamSSE('/api/optimize/learn', {
				difficulty: diff,
				time: learnTime,
				workers: learnWorkers,
			}, handleLearnEvent);
		} catch (e) {
			if (e.name !== 'AbortError') addLog(`Error: ${e.message}`);
		}
		running = false;
	}

	async function replayBest() {
		if (!wsUrl.trim()) return;
		resetGameState();
		running = true;
		runningAction = 'replay';

		try {
			const diff = difficulty === 'auto' ? (tokenInfo?.difficulty || null) : difficulty;
			await streamSSE('/api/optimize/replay', {
				url: wsUrl.trim(),
				difficulty: diff,
			}, handleGameEvent);
		} catch (e) {
			if (e.name !== 'AbortError') addLog(`Error: ${e.message}`);
		}
		running = false;
	}

	function stopRunning() {
		if (abortController) abortController.abort();
		running = false;
		addLog('Stopped.');
	}

	function formatDate(dateStr) {
		if (!dateStr) return '-';
		try {
			return new Date(dateStr).toLocaleString();
		} catch { return dateStr; }
	}
</script>

<div class="page stagger">
	<h1>Learn & Replay</h1>
	<p class="subtitle">Play, optimize, replay with higher scores</p>

	<!-- Solution Status Cards -->
	<div class="cards">
		{#each ['easy', 'medium', 'hard', 'expert'] as diff}
			{@const sol = solutions[diff]}
			<div class="card" class:has-solution={sol}>
				<div class="card-header">
					<span class="diff-badge" style="background: {diffColors[diff]}">{diff}</span>
					{#if sol}
						<span class="card-score">{sol.score}</span>
					{:else}
						<span class="card-score muted">-</span>
					{/if}
				</div>
				{#if sol}
					<div class="card-details">
						<div><span class="label">Bots:</span> {sol.num_bots}</div>
						<div><span class="label">Optimizations:</span> {sol.optimizations_run}</div>
						<div><span class="label">Updated:</span> {formatDate(sol.updated_at)}</div>
					</div>
					<div class="card-actions">
						<button
							class="btn btn-learn"
							disabled={running}
							onclick={() => learnFromCapture(diff)}
						>Learn</button>
					</div>
				{:else}
					<div class="card-details muted">No solution yet</div>
				{/if}
			</div>
		{/each}
	</div>

	<!-- Optimizer Settings -->
	<div class="settings-row">
		<label>
			<span>Time (s):</span>
			<input type="number" bind:value={learnTime} min="10" max="600" step="10" />
		</label>
		<label>
			<span>Workers:</span>
			<input type="number" bind:value={learnWorkers} min="1" max="24" />
		</label>
	</div>

	<!-- Action Panel -->
	<div class="action-panel">
		<div class="url-row">
			<input
				type="text"
				bind:value={wsUrl}
				placeholder="wss://game.ainm.no/ws?token=... or ws://localhost:9999"
				class="url-input"
				disabled={running}
			/>
			{#if tokenInfo}
				<span class="token-info">
					<span class="diff-badge small" style="background: {diffColors[tokenInfo.difficulty] || '#666'}">{tokenInfo.difficulty || '?'}</span>
				</span>
			{/if}
		</div>

		<div class="btn-row">
			{#if running}
				<button class="btn btn-stop" onclick={stopRunning}>Stop</button>
				<span class="running-label">
					{runningAction === 'play' ? 'Playing...' : runningAction === 'learn' ? 'Optimizing...' : 'Replaying...'}
				</span>
			{:else}
				<button
					class="btn btn-play"
					disabled={!wsUrl.trim()}
					onclick={playAndCapture}
				>Play & Capture</button>
				<button
					class="btn btn-replay"
					disabled={!wsUrl.trim()}
					onclick={replayBest}
				>Replay Best</button>
			{/if}
		</div>

		{#if running && (runningAction === 'play' || runningAction === 'replay')}
			<div class="progress-bar">
				<div class="progress-fill" style="width: {(currentRound / maxRounds) * 100}%"></div>
				<span class="progress-text">R{currentRound}/{maxRounds} Score: {score}</span>
			</div>
		{/if}

		{#if finalScore !== null}
			<div class="final-score">Final Score: {finalScore}</div>
		{/if}
	</div>

	<!-- Game Grid (for play/replay) -->
	{#if gameInit && (runningAction === 'play' || runningAction === 'replay')}
		<div class="game-section">
			<div class="grid-container">
				<Grid
					width={gameInit.width}
					height={gameInit.height}
					cellSize={CELL}
					{wallSet}
					{shelfSet}
					{itemMap}
					dropOff={gameInit.drop_off}
					dropOffZones={gameInit.drop_off_zones}
					spawn={gameInit.spawn}
					{bots}
					{botPositions}
					botColors={BOT_COLORS}
					{selectedBot}
					onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
				/>
			</div>

			<!-- Orders sidebar -->
			<div class="side-panel">
				<h3>Orders</h3>
				{#each orders as order, i}
					<div class="order" class:active={order.status === 'active'} class:preview={order.status === 'preview'}>
						<div class="order-header">
							<span class="order-status">{order.status}</span>
						</div>
						<div class="order-items">
							{#each order.items_required as item, j}
								<span class="order-item" class:delivered={order.items_delivered?.[j]}>
									{item}
								</span>
							{/each}
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Live Log -->
	<div class="log-section">
		<h3>Log</h3>
		<div class="log-box">
			{#each logs.slice(-50).reverse() as log}
				<div class="log-line">
					<span class="log-time">{log.time}</span>
					<span class="log-text">{log.text}</span>
				</div>
			{/each}
			{#if logs.length === 0}
				<div class="log-line muted">No activity yet</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.page {
		max-width: 1400px;
		margin: 0 auto;
	}
	h1 {
		font-size: 1.5rem;
		margin: 0 0 0.25rem;
		font-family: var(--font-mono);
		font-weight: 700;
	}
	.subtitle {
		color: var(--text-muted);
		font-size: 0.85rem;
		margin: 0 0 1.5rem;
	}

	/* Solution Cards */
	.cards {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 1rem;
		margin-bottom: 1rem;
	}
	.card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 1rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
		transition: all 0.25s ease;
	}
	.card.has-solution {
		border-color: rgba(57, 211, 83, 0.5);
		box-shadow: 0 0 16px rgba(57, 211, 83, 0.06), 0 2px 12px rgba(0, 0, 0, 0.4);
	}
	.card-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
	}
	.card-score {
		font-size: 1.5rem;
		font-weight: 700;
		color: var(--green);
		font-family: var(--font-mono);
	}
	.card-score.muted {
		color: var(--text-muted);
	}
	.card-details {
		font-size: 0.8rem;
		color: var(--text-muted);
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		margin-bottom: 0.75rem;
	}
	.card-details.muted {
		font-style: italic;
		margin-bottom: 0;
	}
	.card-actions {
		display: flex;
		gap: 0.5rem;
	}
	.label {
		color: var(--text);
	}

	.diff-badge {
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: uppercase;
		color: #000;
	}
	.diff-badge.small {
		font-size: 0.7rem;
		padding: 0.1rem 0.4rem;
	}

	/* Settings */
	.settings-row {
		display: flex;
		gap: 1rem;
		margin-bottom: 1rem;
		align-items: center;
	}
	.settings-row label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.settings-row input {
		width: 70px;
		padding: 0.3rem 0.5rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
		font-size: 0.8rem;
	}

	/* Action Panel */
	.action-panel {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 1rem;
		margin-bottom: 1rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}
	.url-row {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		margin-bottom: 0.75rem;
	}
	.url-input {
		flex: 1;
		padding: 0.5rem 0.75rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm, 6px);
		color: var(--text);
		font-size: 0.85rem;
		font-family: inherit;
		transition: all 0.2s ease;
	}
	.url-input:focus {
		outline: none;
		border-color: var(--accent);
		box-shadow: 0 0 0 2px rgba(57, 211, 83, 0.1);
	}
	.token-info {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}
	.btn-row {
		display: flex;
		gap: 0.5rem;
		align-items: center;
	}
	.running-label {
		color: var(--accent-light);
		font-size: 0.85rem;
		animation: pulse 1.5s infinite;
	}

	/* Buttons */
	.btn {
		padding: 0.4rem 1rem;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		font-size: 0.8rem;
		font-weight: 600;
		font-family: inherit;
		transition: all 0.15s ease;
		letter-spacing: 0.02em;
	}
	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn-play {
		background: var(--green);
		color: #0d1117;
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
	}
	.btn-learn {
		background: var(--accent);
		color: #fff;
		width: 100%;
	}
	.btn-replay {
		background: var(--blue);
		color: #0d1117;
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
	}
	.btn-stop {
		background: var(--red);
		color: #fff;
	}

	/* Progress */
	.progress-bar {
		position: relative;
		height: 24px;
		background: var(--bg);
		border-radius: 4px;
		margin-top: 0.75rem;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--green), var(--accent));
		opacity: 0.4;
		transition: width 0.15s;
	}
	.progress-text {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		font-size: 0.75rem;
		font-weight: 600;
	}
	.final-score {
		margin-top: 0.75rem;
		font-size: 1.2rem;
		font-weight: 700;
		color: var(--green);
		text-align: center;
	}

	/* Game Section */
	.game-section {
		display: flex;
		gap: 1rem;
		margin-bottom: 1rem;
	}
	.grid-container {
		flex-shrink: 0;
	}
	.side-panel {
		flex: 1;
		min-width: 200px;
	}
	.side-panel h3 {
		font-size: 0.9rem;
		margin: 0 0 0.5rem;
	}
	.order {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0.5rem;
		margin-bottom: 0.5rem;
		font-size: 0.75rem;
	}
	.order.active {
		border-color: var(--yellow);
	}
	.order.preview {
		border-color: var(--pink);
		opacity: 0.7;
	}
	.order-header {
		margin-bottom: 0.25rem;
	}
	.order-status {
		text-transform: uppercase;
		font-weight: 600;
		font-size: 0.65rem;
	}
	.order-items {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
	}
	.order-item {
		padding: 0.1rem 0.3rem;
		background: var(--bg);
		border-radius: 3px;
		font-size: 0.7rem;
	}
	.order-item.delivered {
		background: var(--green);
		color: #000;
	}

	/* Log */
	.log-section h3 {
		font-size: 0.9rem;
		margin: 0 0 0.5rem;
	}
	.log-box {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.75rem;
		max-height: 300px;
		overflow-y: auto;
		font-size: 0.75rem;
	}
	.log-line {
		display: flex;
		gap: 0.5rem;
		padding: 0.15rem 0;
		border-bottom: 1px solid var(--border);
	}
	.log-line:last-child {
		border-bottom: none;
	}
	.log-time {
		color: var(--text-muted);
		white-space: nowrap;
	}
	.log-text {
		word-break: break-all;
	}
	.muted {
		color: var(--text-muted);
	}

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}

	@media (max-width: 800px) {
		.cards {
			grid-template-columns: repeat(2, 1fr);
		}
		.game-section {
			flex-direction: column;
		}
	}
</style>
