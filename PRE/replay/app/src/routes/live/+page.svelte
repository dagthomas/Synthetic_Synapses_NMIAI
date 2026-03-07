<script>
	import Grid from '$lib/components/Grid.svelte';

	const CELL = 28;
	const BOT_COLORS = [
		'#f85149', '#58a6ff', '#39d353', '#d29922', '#bc8cff',
		'#3fb950', '#db6d28', '#8b949e', '#f778ba', '#79c0ff',
		'#ff6e6e', '#00d4aa', '#ffd700', '#ff69b4', '#00bfff',
		'#98fb98', '#dda0dd', '#f0e68c', '#87ceeb', '#ffa07a',
	];
	const diffColors = {
		easy: '#39d353',
		medium: '#d29922',
		hard: '#f85149',
		expert: '#da3633',
		nightmare: '#a855f7',
	};

	// Input state
	let wsUrl = $state('');
	let difficulty = $state('medium');
	let solver = $state('zig');
	let running = $state(false);
	let finished = $state(false);

	// Solution info for replay mode
	let solutionInfo = $state(null);
	let solutionLoading = $state(false);

	async function loadSolutionInfo(diff) {
		solutionLoading = true;
		try {
			const today = new Date().toISOString().slice(0, 10);
			const res = await fetch(`/api/solutions?date=${today}`);
			const data = await res.json();
			const solutions = data.byDifficulty?.[diff];
			solutionInfo = solutions?.[0] || null;
		} catch {
			solutionInfo = null;
		}
		solutionLoading = false;
	}

	// Load solution info when difficulty changes or solver is replay
	$effect(() => {
		if (solver === 'replay') {
			loadSolutionInfo(difficulty);
		}
	});

	// Game state
	let gameInit = $state(null);
	let currentRound = $state(0);
	let maxRounds = $state(300);
	let score = $state(0);
	let finalScore = $state(null);
	let bots = $state([]);
	let orders = $state([]);
	let actions = $state([]);
	let logs = $state([]);
	let events = $state([]);
	let dbRunId = $state(null);

	// GPU status
	let gpuConnected = $state(false);
	let gpuInfo = $state(null);
	let gpuChecking = $state(true);

	async function checkGpu() {
		gpuChecking = true;
		try {
			const res = await fetch('/api/gpu-remote');
			gpuInfo = await res.json();
			gpuConnected = gpuInfo.connected;
		} catch {
			gpuConnected = false;
		}
		gpuChecking = false;
	}

	$effect(() => { checkGpu(); });

	// Order tracking
	let ordersCompleted = $state(0);
	let lastActiveOrderId = $state(null);
	let scoreHistory = $state([]);

	// Grid data (derived from init)
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

	function getItemTypeName(t) {
		return t.charAt(0).toUpperCase() + t.slice(1);
	}

	// Parse JWT token from URL to extract difficulty and seed
	function parseToken(url) {
		try {
			const tokenMatch = url.match(/[?&]token=([^&]+)/);
			if (!tokenMatch) return null;
			const token = tokenMatch[1];
			// JWT: header.payload.signature — decode payload (base64url)
			const parts = token.split('.');
			if (parts.length < 2) return null;
			const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
			const decoded = JSON.parse(atob(payload));
			return decoded;
		} catch (e) {
			return null;
		}
	}

	// Auto-detect difficulty when URL changes
	let tokenInfo = $derived.by(() => {
		if (!wsUrl.trim()) return null;
		return parseToken(wsUrl);
	});

	$effect(() => {
		if (tokenInfo?.difficulty) difficulty = tokenInfo.difficulty;
	});

	// Reset game state when URL changes (so old grid doesn't persist)
	let prevUrl = $state('');
	$effect(() => {
		if (wsUrl !== prevUrl && !running) {
			prevUrl = wsUrl;
			if (gameInit) {
				gameInit = null;
				finished = false;
				bots = [];
				orders = [];
				actions = [];
				currentRound = 0;
				score = 0;
				finalScore = null;
				ordersCompleted = 0;
				scoreHistory = [];
			}
		} else {
			prevUrl = wsUrl;
		}
	});

	function getBotAction(botId) {
		const a = actions.find(a => a.bot === botId);
		return a ? a.action : 'wait';
	}

	let abortController = $state(null);

	async function startGame() {
		if (!wsUrl.trim()) return;

		// Abort any previous run
		if (abortController) abortController.abort();
		abortController = new AbortController();

		running = true;
		finished = false;
		gameInit = null;
		currentRound = 0;
		score = 0;
		finalScore = null;
		bots = [];
		orders = [];
		actions = [];
		logs = [];
		events = [];
		dbRunId = null;
		ordersCompleted = 0;
		lastActiveOrderId = null;
		scoreHistory = [];

		try {
			const response = await fetch('/api/run-live', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ url: wsUrl.trim(), difficulty, solver }),
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
						handleEvent(event);
					} catch (e) { // Stream closed — client disconnected
					}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				logs = [...logs, `Error: ${e.message}`];
			}
		}

		running = false;
		finished = true;
	}

	function stopGame() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		running = false;
		finished = true;
		logs = [...logs, 'Game stopped by user'];
	}

	function handleEvent(event) {
		switch (event.type) {
			case 'init':
				gameInit = event;
				maxRounds = event.max_rounds || 300;
				break;

			case 'round':
				currentRound = event.round;
				score = event.score;
				bots = event.bots || [];
				orders = event.orders || [];
				// Track order completions
				const activeOrder = orders.find(o => o.status === 'active');
				if (activeOrder && lastActiveOrderId !== null && activeOrder.id !== lastActiveOrderId) {
					ordersCompleted++;
				}
				if (activeOrder) lastActiveOrderId = activeOrder.id;
				// Track score history (sample every 10 rounds)
				if (event.round % 10 === 0) {
					scoreHistory = [...scoreHistory, { round: event.round, score: event.score }];
				}
				break;

			case 'actions':
				actions = event.actions || [];
				break;

			case 'progress':
				currentRound = event.round;
				score = event.score;
				break;

			case 'log':
				logs = [...logs.slice(-100), event.text];
				break;

			case 'final_score':
				finalScore = event.score;
				break;

			case 'game_over':
				finalScore = event.score;
				score = event.score;
				break;

			case 'done':
				finished = true;
				running = false;
				logs = [...logs, event.message];
				break;

			case 'db':
				dbRunId = event.run_id;
				logs = [...logs, `Saved to DB: run_id=${event.run_id}`];
				break;

			case 'db_error':
				logs = [...logs, `DB error: ${event.message}`];
				break;

			case 'error':
				logs = [...logs, `Error: ${event.message}`];
				break;

			case 'status':
				logs = [...logs, event.message];
				break;

			case 'gpu_event': {
				// Forward GPU solver streaming events to log
				const ge = event;  // nested event from capture_and_solve_stream.py
				if (ge.type === 'capture_done') {
					logs = [...logs, ge.message];
				} else if (ge.type === 'solver_init') {
					logs = [...logs, `GPU solver: ${ge.num_bots} bots, ${ge.device || 'cuda'}, ${ge.gpu_name || ''}`];
				} else if (ge.type === 'solver_solving') {
					logs = [...logs, ge.msg];
				} else if (ge.type === 'bot_start') {
					logs = [...logs, `Bot ${ge.bot_id}/${ge.total_bots} starting...`];
				} else if (ge.type === 'bot_done') {
					logs = [...logs, `Bot ${ge.bot_id}/${ge.total_bots} done: score=${ge.score} (${ge.time}s)`];
				} else if (ge.type === 'gpu_phase') {
					const label = ge.phase === 'pass1' ? 'Pass 1: Sequential'
						: ge.phase === 'refine' ? `Refinement iter ${ge.iteration}`
						: ge.phase === 'pass1_done' ? `Pass 1 done: score=${ge.cpu_score}`
						: ge.phase === 'refine_done' ? `Refine ${ge.iteration} done: score=${ge.cpu_score}`
						: ge.phase;
					logs = [...logs, label];
				} else if (ge.type === 'solver_result') {
					logs = [...logs, `GPU result: score=${ge.score} in ${ge.time}s`];
					finalScore = ge.score;
					score = ge.score;
				} else if (ge.type === 'solver_improved') {
					logs = [...logs, `IMPROVED: ${ge.old_score} → ${ge.new_score} (+${ge.delta})`];
				} else if (ge.type === 'solver_no_improvement') {
					logs = [...logs, `No improvement: ${ge.score} (prev best: ${ge.prev})`];
				} else if (ge.type === 'solver_prev_best') {
					logs = [...logs, `Previous best: ${ge.score}`];
				} else if (ge.type === 'solver_done') {
					logs = [...logs, `Solver done: score=${ge.score} in ${ge.time}s`];
				} else if (ge.type === 'final_score') {
					logs = [...logs, `Capture score: ${ge.score}`];
				} else if (ge.type === 'progress') {
					currentRound = ge.round;
					score = ge.score || score;
				}
				break;
			}
		}
	}

	// Count dead inventory
	let deadInventory = $derived.by(() => {
		if (!orders.length || !bots.length) return 0;
		const activeOrder = orders.find(o => o.status === 'active');
		const previewOrder = orders.find(o => o.status === 'preview');
		if (!activeOrder) return 0;

		let dead = 0;
		for (const bot of bots) {
			for (const item of (bot.inventory || [])) {
				const activeRemaining = [...activeOrder.items_required];
				for (const d of (activeOrder.items_delivered || [])) {
					const idx = activeRemaining.indexOf(d);
					if (idx >= 0) activeRemaining.splice(idx, 1);
				}
				let useful = activeRemaining.includes(item);
				if (!useful && previewOrder) {
					const previewRemaining = [...previewOrder.items_required];
					for (const d of (previewOrder.items_delivered || [])) {
						const idx = previewRemaining.indexOf(d);
						if (idx >= 0) previewRemaining.splice(idx, 1);
					}
					useful = previewRemaining.includes(item);
				}
				if (!useful) dead++;
			}
		}
		return dead;
	});

	let totalInventory = $derived(bots.reduce((sum, b) => sum + (b.inventory?.length || 0), 0));

	// Derived order stats
	let itemsDelivered = $derived(score - ordersCompleted * 5);
	let roundsPerOrder = $derived(ordersCompleted > 0 ? Math.round(currentRound / ordersCompleted * 10) / 10 : 0);
	let activeOrderProgress = $derived.by(() => {
		const ao = orders.find(o => o.status === 'active');
		if (!ao) return null;
		const delivered = ao.items_delivered?.length || 0;
		const total = ao.items_required.length;
		return { delivered, total, pct: Math.round(delivered / total * 100) };
	});
	let projectedScore = $derived.by(() => {
		if (currentRound < 30 || score === 0) return null;
		return Math.round(score / currentRound * maxRounds);
	});
</script>

<div class="live-page stagger">
	<!-- Input Bar -->
	<div class="input-bar card">
		<div class="input-row">
			<input
				type="text"
				bind:value={wsUrl}
				placeholder="wss://game.ainm.no/ws?token=..."
				class="url-input"
				disabled={running}
				/>
			<label class="select-label">
				<span class="label-text">Bot</span>
				<select bind:value={solver} class="solver-select" disabled={running}>
					<option value="replay">Replay</option>
					<option value="zig">Zig</option>
					<option value="python">Python</option>
					<option value="gpu">GPU</option>
				</select>
			</label>
			{#if tokenInfo}
				<span class="token-info" style="color: {diffColors[tokenInfo.difficulty] || 'var(--text)'}">
					{tokenInfo.difficulty?.toUpperCase() || '?'}
					{#if tokenInfo.map_seed}
						<span class="seed-badge">#{tokenInfo.map_seed}</span>
					{/if}
				</span>
			{:else}
				<select bind:value={difficulty} class="diff-select" disabled={running}>
					<option value="easy">Easy</option>
					<option value="medium">Medium</option>
					<option value="hard">Hard</option>
					<option value="expert">Expert</option>
					<option value="nightmare">Nightmare</option>
				</select>
			{/if}
			{#if running}
				<button class="stop-btn" onclick={stopGame}>Stop</button>
			{:else}
				<button
					class="run-btn"
					onclick={startGame}
					disabled={!wsUrl.trim()}
				>
					Run Game
				</button>
			{/if}
		</div>

		<!-- GPU status -->
		<div class="gpu-status" class:connected={gpuConnected} class:disconnected={!gpuConnected && !gpuChecking}>
			<span class="gpu-dot"></span>
			{#if gpuChecking}
				<span class="gpu-label">Checking GPU...</span>
			{:else if gpuConnected}
				<span class="gpu-label">{gpuInfo?.name}</span>
				<span class="gpu-vram">{gpuInfo?.vram_gb} GB</span>
				{#if gpuInfo?.source}
					<span class="gpu-source">{gpuInfo.source}</span>
				{/if}
			{:else}
				<span class="gpu-label">No GPU</span>
			{/if}
		</div>

		<!-- Solution info for replay mode -->
		{#if solver === 'replay' && !running}
			<div class="solution-info">
				{#if solutionLoading}
					<span class="sol-loading">Loading solution...</span>
				{:else if solutionInfo}
					<div class="sol-badge ready">
						<span class="sol-label">Solution ready:</span>
						<span class="sol-score" style="color: {diffColors[difficulty] || '#39d353'}">{solutionInfo.score} pts</span>
						<span class="sol-meta">{solutionInfo.num_bots} bots, {solutionInfo.num_rounds} rounds, {solutionInfo.optimizations_run} opt</span>
					</div>
				{:else}
					<div class="sol-badge empty">
						<span class="sol-label">No solution for {difficulty} today.</span>
						<span class="sol-hint">Run GPU optimization first (Stepladder or GPU page)</span>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Progress bar -->
		{#if running || finished}
			<div class="progress-section">
				<div class="progress-bar">
					<div class="progress-fill" style="width: {(currentRound / maxRounds) * 100}%"></div>
				</div>
				<div class="progress-info">
					<span class="round-info">Round {currentRound}/{maxRounds}</span>
					<span class="score-info">
						Score: <strong class="score-value">{score}</strong>
						{#if finalScore !== null}
							<span class="final-badge">Final: {finalScore}</span>
						{/if}
					</span>
					{#if ordersCompleted > 0}
						<span class="orders-info">
							Orders: <strong>{ordersCompleted}</strong>
							{#if roundsPerOrder > 0}
								<span class="rate-badge">({roundsPerOrder} r/o)</span>
							{/if}
						</span>
					{/if}
					{#if totalInventory > 0}
						<span class="inv-info">
							Inv: {totalInventory}
							{#if deadInventory > 0}
								<span class="dead-badge">({deadInventory} dead)</span>
							{/if}
						</span>
					{/if}
					{#if projectedScore !== null}
						<span class="projected-info">Proj: ~{projectedScore}</span>
					{/if}
					{#if dbRunId}
						<a href="/run/{dbRunId}" class="replay-link">View Replay</a>
					{/if}
				</div>
			</div>
		{/if}
	</div>

	{#if gameInit}
		<div class="main-area">
			<!-- Grid -->
			<div class="grid-section">
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
									<span class="order-item" class:delivered={order.items_delivered?.includes(item)}>
										{getItemTypeName(item)}
									</span>
								{/each}
							</div>
							<div class="order-progress">
								{order.items_delivered?.length || 0} / {order.items_required.length} delivered
							</div>
						</div>
					{/each}
				</div>

				<!-- Stats -->
				{#if currentRound > 0}
				<div class="panel-section card stats-panel">
					<h3>Stats</h3>
					<div class="stats-grid">
						<div class="stat">
							<span class="stat-label">Orders Done</span>
							<span class="stat-value green">{ordersCompleted}</span>
						</div>
						<div class="stat">
							<span class="stat-label">Items Delivered</span>
							<span class="stat-value">{itemsDelivered}</span>
						</div>
						<div class="stat">
							<span class="stat-label">Rounds/Order</span>
							<span class="stat-value" class:bad={roundsPerOrder > 20} class:good={roundsPerOrder > 0 && roundsPerOrder <= 15}>{roundsPerOrder || '-'}</span>
						</div>
						<div class="stat">
							<span class="stat-label">Dead Inv</span>
							<span class="stat-value" class:bad={deadInventory > 2}>{deadInventory}/{totalInventory}</span>
						</div>
						{#if projectedScore !== null}
						<div class="stat">
							<span class="stat-label">Projected</span>
							<span class="stat-value accent">~{projectedScore}</span>
						</div>
						{/if}
						{#if activeOrderProgress}
						<div class="stat">
							<span class="stat-label">Active Order</span>
							<span class="stat-value">{activeOrderProgress.delivered}/{activeOrderProgress.total} ({activeOrderProgress.pct}%)</span>
						</div>
						{/if}
					</div>
				</div>
				{/if}

				<!-- Bots -->
				<div class="panel-section card">
					<h3>Bots ({bots.length})</h3>
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
								{#if !bot.inventory?.length}
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

				<!-- Log -->
				<div class="panel-section card">
					<h3>Log ({logs.length})</h3>
					<div class="log-area">
						{#each logs.slice(-30) as line}
							<div class="log-line">{line}</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	{:else if !running && !finished}
		<div class="empty-state card">
			<h2>Live Game Runner</h2>
		</div>
	{:else if running && !gameInit}
		<div class="launch-screen">
			<!-- Animated background grid -->
			<div class="launch-grid-bg">
				{#each Array(12) as _, i}
					<div class="grid-line-h" style="top: {(i + 1) * 8}%; animation-delay: {i * 0.1}s"></div>
				{/each}
				{#each Array(16) as _, i}
					<div class="grid-line-v" style="left: {(i + 1) * 6}%; animation-delay: {i * 0.08}s"></div>
				{/each}
			</div>

			<!-- Scanning rings -->
			<div class="launch-rings">
				<div class="ring ring-1"></div>
				<div class="ring ring-2"></div>
				<div class="ring ring-3"></div>
			</div>

			<!-- Bot icon -->
			<div class="launch-bot">
				<svg viewBox="0 0 80 80" width="80" height="80">
					<g class="launch-bot-hover">
						<path d="M 30 55 L 50 55 L 46 62 L 34 62 Z" fill="#b2bec3" />
						<line x1="40" y1="22" x2="40" y2="10" stroke="#b2bec3" stroke-width="2.5" />
						<circle cx="40" cy="9" r="3" fill="#39d353" class="antenna-pulse" />
						<rect x="12" y="30" width="8" height="16" rx="3" fill="#b2bec3" />
						<rect x="60" y="30" width="8" height="16" rx="3" fill="#b2bec3" />
						<rect x="18" y="22" width="44" height="35" rx="8" fill="#39d353" class="chassis-glow" />
						<rect x="24" y="28" width="32" height="24" rx="4" fill="#1e272e" />
						<circle cx="32" cy="34" r="2" fill="#fff" class="eye-blink" />
						<circle cx="48" cy="34" r="2" fill="#fff" class="eye-blink" />
						<text x="40" y="46" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="900" fill="#39d353" font-family="monospace">GO</text>
					</g>
				</svg>
			</div>

			<!-- Status text -->
			<div class="launch-text">
				<h2 class="launch-title">INITIALIZING</h2>
				<div class="launch-subtitle">
					<span class="launch-solver">{solver.toUpperCase()}</span>
					{#if tokenInfo}
						<span class="launch-diff" style="color: {diffColors[tokenInfo.difficulty] || '#fff'}">{tokenInfo.difficulty?.toUpperCase()}</span>
						{#if tokenInfo.map_seed}
							<span class="launch-seed">SEED #{tokenInfo.map_seed}</span>
						{/if}
					{:else}
						<span class="launch-diff" style="color: {diffColors[difficulty]}">{difficulty.toUpperCase()}</span>
					{/if}
				</div>
			</div>

			<!-- Animated status steps -->
			<div class="launch-steps">
				{#each logs.slice(-4) as line, i}
					<div class="launch-step" style="animation-delay: {i * 0.15}s">
						<span class="step-dot"></span>
						<span class="step-text">{line}</span>
					</div>
				{/each}
				{#if logs.length === 0}
					<div class="launch-step">
						<span class="step-dot"></span>
						<span class="step-text">Spawning bot process...</span>
					</div>
				{/if}
			</div>

			<!-- Scanning bar -->
			<div class="scan-bar"></div>
		</div>
	{/if}
</div>

<style>
	.live-page { display: flex; flex-direction: column; gap: 1rem; }

	.input-bar { display: flex; flex-direction: column; gap: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4); }
	.input-row { display: flex; gap: 0.5rem; align-items: center; }
	.url-input {
		flex: 1;
		padding: 0.6rem 0.75rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 0.8rem;
		outline: none;
		transition: all 0.2s ease;
	}
	.url-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(57, 211, 83, 0.1); }
	.url-input:disabled { opacity: 0.5; }

	.gpu-status {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.3rem 0.6rem;
		border-radius: 4px;
		font-size: 0.7rem;
		font-family: var(--font-mono);
		border: 1px solid var(--border);
		color: var(--text-muted);
	}
	.gpu-status.connected {
		background: rgba(57, 211, 83, 0.06);
		border-color: rgba(57, 211, 83, 0.25);
	}
	.gpu-status.disconnected {
		background: rgba(248, 81, 73, 0.06);
		border-color: rgba(248, 81, 73, 0.25);
	}
	.gpu-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		flex-shrink: 0;
		background: var(--text-muted);
	}
	.connected .gpu-dot { background: var(--accent); }
	.disconnected .gpu-dot { background: #f85149; }
	.gpu-label { color: inherit; }
	.connected .gpu-label { color: var(--accent); }
	.disconnected .gpu-label { color: #f85149; }
	.gpu-vram { color: var(--text-muted); }
	.gpu-source {
		padding: 0.05rem 0.3rem;
		border-radius: 3px;
		font-size: 0.6rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		background: rgba(255,255,255,0.06);
		border: 1px solid rgba(255,255,255,0.1);
		color: var(--text-muted);
	}

	.select-label {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}
	.label-text {
		font-size: 0.6rem;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-muted);
		font-weight: 600;
	}
	.diff-select, .solver-select {
		padding: 0.6rem 0.75rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		color: var(--text);
		font-size: 0.85rem;
		outline: none;
		cursor: pointer;
	}
	.diff-select:disabled, .solver-select:disabled { opacity: 0.5; }
	.solver-select { border-color: var(--accent); }

	.solution-info {
		padding: 0.5rem 0 0;
	}
	.sol-badge {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.5rem 0.75rem;
		border-radius: var(--radius-sm);
		font-size: 0.8rem;
	}
	.sol-badge.ready {
		background: rgba(57, 211, 83, 0.08);
		border: 1px solid rgba(57, 211, 83, 0.2);
	}
	.sol-badge.empty {
		background: rgba(248, 81, 73, 0.08);
		border: 1px solid rgba(248, 81, 73, 0.2);
	}
	.sol-label { color: var(--text-muted); font-size: 0.75rem; }
	.sol-score { font-weight: 800; font-size: 1rem; }
	.sol-meta { color: var(--text-muted); font-size: 0.7rem; font-family: var(--font-mono); }
	.sol-hint { color: var(--text-muted); font-size: 0.7rem; font-style: italic; }
	.sol-loading { color: var(--text-muted); font-size: 0.75rem; }
	.token-info {
		font-weight: 700;
		font-size: 0.85rem;
		padding: 0.6rem 0.75rem;
		white-space: nowrap;
	}
	.seed-badge {
		font-weight: 400;
		font-size: 0.75rem;
		color: var(--text-muted);
		font-family: var(--font-mono);
		margin-left: 0.25rem;
	}

	.run-btn {
		padding: 0.6rem 1.5rem;
		background: var(--accent);
		color: #0d1117;
		font-weight: 600;
		border: none;
		border-radius: var(--radius);
		white-space: nowrap;
		letter-spacing: 0.03em;
	}
	.run-btn:hover:not(:disabled) { background: var(--accent-light); box-shadow: 0 0 12px rgba(57, 211, 83, 0.2); }
	.run-btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.stop-btn {
		padding: 0.6rem 1.5rem;
		background: var(--red, #e74c3c);
		color: white;
		font-weight: 600;
		border: none;
		border-radius: var(--radius);
		white-space: nowrap;
		cursor: pointer;
		letter-spacing: 0.03em;
	}
	.stop-btn:hover { opacity: 0.85; }

	.progress-section { display: flex; flex-direction: column; gap: 0.35rem; }
	.progress-bar {
		width: 100%;
		height: 6px;
		background: var(--bg);
		border-radius: 3px;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent), var(--green));
		border-radius: 3px;
		transition: width 0.2s ease;
	}
	.progress-info {
		display: flex;
		align-items: center;
		gap: 1rem;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.round-info { font-family: var(--font-mono); }
	.score-info { display: flex; align-items: center; gap: 0.35rem; }
	.score-value { color: var(--green); font-size: 1.1rem; }
	.final-badge {
		padding: 0.1rem 0.4rem;
		background: var(--green)22;
		color: var(--green);
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 600;
	}
	.dead-badge { color: var(--red); font-weight: 600; }
	.inv-info { font-family: var(--font-mono); }
	.replay-link {
		margin-left: auto;
		padding: 0.2rem 0.6rem;
		background: var(--accent)22;
		border-radius: 4px;
		font-size: 0.75rem;
	}

	.main-area {
		display: grid;
		grid-template-columns: 1fr 420px;
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
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}

	.side-panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		max-height: calc(100vh - 200px);
		overflow-y: auto;
	}

	.panel-section h3 {
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
		padding-bottom: 0.35rem;
		border-bottom: 1px solid rgba(48, 54, 61, 0.4);
	}

	.order {
		padding: 0.5rem;
		border-radius: 6px;
		margin-bottom: 0.5rem;
		border: 1px solid var(--border);
		transition: border-color 0.15s ease;
	}
	.active-order { border-color: #facc1544; background: #facc1508; }
	.preview-order { border-color: #f472b644; background: #f472b608; }
	.order-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.35rem; }
	.order-status {
		font-size: 0.65rem;
		text-transform: uppercase;
		font-weight: 700;
		padding: 0.1rem 0.35rem;
		border-radius: 3px;
		background: #f472b622;
		color: var(--pink);
	}
	.order-status.active { background: #facc1522; color: var(--yellow); }
	.order-id { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); }
	.order-items { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.25rem; }
	.order-item {
		font-size: 0.62rem;
		padding: 0.1rem 0.3rem;
		background: var(--bg);
		border-radius: 3px;
		border: 1px solid var(--border);
	}
	.order-item.delivered {
		background: #39d35322;
		border-color: #39d35344;
		color: var(--green);
		text-decoration: line-through;
	}
	.order-progress { font-size: 0.7rem; color: var(--text-muted); }

	.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem 0.75rem; }
	.stat { display: flex; justify-content: space-between; align-items: center; padding: 0.15rem 0; }
	.stat-label { font-size: 0.7rem; color: var(--text-muted); }
	.stat-value { font-size: 0.8rem; font-weight: 600; font-family: var(--font-mono); }
	.stat-value.green { color: var(--green); }
	.stat-value.bad { color: var(--red); }
	.stat-value.good { color: var(--green); }
	.stat-value.accent { color: var(--accent-light); }
	.orders-info { font-family: var(--font-mono); }
	.rate-badge { font-size: 0.7rem; color: var(--text-muted); }
	.projected-info { font-family: var(--font-mono); color: var(--accent-light); }

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
		transition: all 0.15s ease;
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
	.bot-pos { font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); }
	.bot-action {
		margin-left: auto;
		font-size: 0.7rem;
		color: var(--accent-light);
		font-family: var(--font-mono);
	}
	.bot-inv { display: flex; flex-wrap: wrap; gap: 0.2rem; margin-left: 1.75rem; }
	.empty-inv { font-size: 0.7rem; color: var(--text-muted); font-style: italic; }
	.inv-item {
		font-size: 0.65rem;
		padding: 0.05rem 0.35rem;
		background: #39d35318;
		border: 1px solid #39d35333;
		border-radius: 3px;
		color: var(--accent-light);
	}

	.log-area {
		max-height: 200px;
		overflow-y: auto;
		font-family: var(--font-mono);
		font-size: 0.7rem;
		background: rgba(1, 4, 9, 0.5);
		border-radius: 6px;
		padding: 0.5rem;
	}
	.log-line {
		padding: 0.15rem 0;
		color: var(--text-muted);
		border-bottom: 1px solid rgba(48, 54, 61, 0.3);
	}
	.mini-log { max-height: 100px; margin-top: 0.5rem; }

	.empty-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 3rem;
		text-align: center;
		gap: 0.75rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}
	.empty-state h2 { font-size: 1.3rem; font-family: var(--font-mono); }
	.empty-state p { color: var(--text-muted); max-width: 500px; }

	.spinner {
		width: 32px;
		height: 32px;
		border: 3px solid var(--border);
		border-top-color: var(--accent);
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}

	.mono { font-family: var(--font-mono); font-size: 0.8rem; }

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	/* ── Launch Screen ── */
	.launch-screen {
		position: relative;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 500px;
		background: radial-gradient(ellipse at center, #161b22 0%, #0d1117 70%);
		border: 1px solid #39d35333;
		border-radius: var(--radius);
		overflow: hidden;
		gap: 1.5rem;
	}

	/* Background grid */
	.launch-grid-bg {
		position: absolute;
		inset: 0;
		overflow: hidden;
		opacity: 0.15;
	}
	.grid-line-h {
		position: absolute;
		left: 0; right: 0;
		height: 1px;
		background: linear-gradient(90deg, transparent, #39d353, transparent);
		animation: gridFadeIn 1.5s ease-out forwards;
		opacity: 0;
	}
	.grid-line-v {
		position: absolute;
		top: 0; bottom: 0;
		width: 1px;
		background: linear-gradient(180deg, transparent, #39d353, transparent);
		animation: gridFadeIn 1.5s ease-out forwards;
		opacity: 0;
	}

	/* Scanning rings */
	.launch-rings {
		position: absolute;
		width: 300px;
		height: 300px;
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.ring {
		position: absolute;
		border-radius: 50%;
		border: 1px solid;
	}
	.ring-1 {
		width: 120px; height: 120px;
		border-color: #39d35344;
		animation: ringPulse 3s ease-out infinite;
	}
	.ring-2 {
		width: 200px; height: 200px;
		border-color: #39d35333;
		animation: ringPulse 3s ease-out infinite 0.5s;
	}
	.ring-3 {
		width: 280px; height: 280px;
		border-color: #39d35322;
		animation: ringPulse 3s ease-out infinite 1s;
	}

	/* Bot icon */
	.launch-bot {
		z-index: 2;
		filter: drop-shadow(0 0 20px #39d35344);
	}
	.launch-bot-hover {
		transform-origin: 40px 40px;
		animation: botHoverLaunch 2s ease-in-out infinite;
	}
	.antenna-pulse {
		animation: antennaBlink 1.2s infinite;
	}
	.chassis-glow {
		animation: chassisGlow 2s ease-in-out infinite;
	}
	.eye-blink {
		animation: eyeBlink 3s infinite;
	}

	/* Status text */
	.launch-text {
		z-index: 2;
		text-align: center;
	}
	.launch-title {
		font-size: 1.8rem;
		font-weight: 900;
		letter-spacing: 0.15em;
		color: #fff;
		text-shadow: 0 0 30px #39d35366;
		animation: titlePulse 2s ease-in-out infinite;
		font-family: var(--font-mono);
	}
	.launch-subtitle {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 1rem;
		margin-top: 0.5rem;
	}
	.launch-diff {
		font-size: 1.1rem;
		font-weight: 800;
		letter-spacing: 0.15em;
		text-shadow: 0 0 15px currentColor;
	}
	.launch-seed {
		font-family: var(--font-mono);
		font-size: 0.85rem;
		color: var(--text-muted);
		padding: 0.2rem 0.6rem;
		background: #ffffff08;
		border: 1px solid #ffffff15;
		border-radius: 4px;
	}
	.launch-solver {
		font-family: var(--font-mono);
		font-size: 0.85rem;
		font-weight: 700;
		color: var(--accent-light);
		padding: 0.2rem 0.6rem;
		background: #39d35318;
		border: 1px solid #39d35344;
		border-radius: 4px;
	}

	/* Status steps */
	.launch-steps {
		z-index: 2;
		display: flex;
		flex-direction: column;
		gap: 0.4rem;
		min-width: 300px;
	}
	.launch-step {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		opacity: 0;
		animation: stepSlideIn 0.4s ease-out forwards;
		font-family: var(--font-mono);
		font-size: 0.75rem;
	}
	.step-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: #39d353;
		box-shadow: 0 0 8px #39d35366;
		flex-shrink: 0;
		animation: dotPulse 1.5s infinite;
	}
	.step-text {
		color: var(--text-muted);
	}

	/* Scanning bar */
	.scan-bar {
		position: absolute;
		left: 0;
		right: 0;
		height: 2px;
		background: linear-gradient(90deg, transparent, #39d353, transparent);
		animation: scanMove 2.5s ease-in-out infinite;
		box-shadow: 0 0 15px #39d353, 0 0 30px #39d35344;
	}

	/* ── Launch Animations ── */
	@keyframes gridFadeIn {
		0% { opacity: 0; transform: scaleX(0); }
		100% { opacity: 1; transform: scaleX(1); }
	}
	@keyframes ringPulse {
		0% { transform: scale(0.8); opacity: 0.6; }
		50% { opacity: 0.2; }
		100% { transform: scale(1.3); opacity: 0; }
	}
	@keyframes botHoverLaunch {
		0%, 100% { transform: translateY(0); }
		50% { transform: translateY(-6px); }
	}
	@keyframes antennaBlink {
		0%, 100% { opacity: 1; fill: #39d353; }
		50% { opacity: 0.3; fill: #56d364; }
	}
	@keyframes chassisGlow {
		0%, 100% { filter: brightness(1); }
		50% { filter: brightness(1.3); }
	}
	@keyframes eyeBlink {
		0%, 85%, 100% { opacity: 1; }
		90%, 94% { opacity: 0; }
	}
	@keyframes titlePulse {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.7; }
	}
	@keyframes stepSlideIn {
		0% { opacity: 0; transform: translateX(-20px); }
		100% { opacity: 1; transform: translateX(0); }
	}
	@keyframes dotPulse {
		0%, 100% { box-shadow: 0 0 4px #39d35344; }
		50% { box-shadow: 0 0 12px #39d353; }
	}
	@keyframes scanMove {
		0% { top: 0; }
		50% { top: 100%; }
		100% { top: 0; }
	}

	@media (max-width: 900px) {
		.main-area { grid-template-columns: 1fr; }
		.side-panel { max-height: none; }
	}
</style>
