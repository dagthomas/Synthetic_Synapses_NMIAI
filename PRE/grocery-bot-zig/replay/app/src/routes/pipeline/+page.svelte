<script>
	import Grid from '$lib/components/Grid.svelte';

	const CELL = 24;
	const BOT_COLORS = [
		'#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
		'#1abc9c', '#e67e22', '#34495e', '#e84393', '#00cec9',
	];
	const SOURCE_COLORS = {
		greedy: '#8b8fa3',
		mapf:   '#fdcb6e',
		gpu_pass_0: '#74b9ff',
		gpu_pass_1: '#a29bfe',
		gpu_pass_2: '#6c5ce7',
		gpu_pass_3: '#e84393',
		gpu_refine: '#00b894',
		none:   '#444',
	};
	function sourceColor(s) {
		return SOURCE_COLORS[s] || '#74b9ff';
	}

	// ── Input State ──────────────────────────────────────────────────────────────
	let wsUrl          = $state('');
	let replayUrl      = $state('');
	let postOptTime    = $state(1800);
	let running        = $state(false);
	let phase          = $state('idle'); // idle | playing | post_optimizing | replaying | done
	let abortCtrl      = $state(null);

	// ── Game State ───────────────────────────────────────────────────────────────
	let gameInit       = $state(null);  // init event data
	let currentRound   = $state(0);
	let maxRounds      = $state(300);
	let liveScore      = $state(0);
	let planScore      = $state(0);
	let planSource     = $state('none');
	let bots           = $state([]);
	let orders         = $state([]);
	let finalScore     = $state(null);

	// ── Score Chart ──────────────────────────────────────────────────────────────
	let scoreHistory   = $state([]);  // [{round, score, planScore}]
	const CHART_W = 480, CHART_H = 100;

	let chartMaxScore = $derived(
		scoreHistory.length > 0 ? Math.max(...scoreHistory.map(p => Math.max(p.score, p.planScore || 0)), 10) : 10
	);
	let chartPolyline = $derived.by(() => {
		if (scoreHistory.length < 2) return '';
		const mr = maxRounds || 300;
		return scoreHistory.map(p =>
			`${(p.round / mr) * CHART_W},${CHART_H - (p.score / chartMaxScore) * CHART_H}`
		).join(' ');
	});
	let chartPlanLine = $derived.by(() => {
		if (scoreHistory.length < 2) return '';
		const mr = maxRounds || 300;
		return scoreHistory
			.filter(p => p.planScore)
			.map(p =>
				`${(p.round / mr) * CHART_W},${CHART_H - (p.planScore / chartMaxScore) * CHART_H}`
			).join(' ');
	});
	let chartLastPt = $derived(scoreHistory.length > 0 ? scoreHistory[scoreHistory.length - 1] : null);

	// ── GPU Passes ───────────────────────────────────────────────────────────────
	let gpuPasses      = $state([]);   // [{pass, max_states, score, elapsed}]
	let bestGpuScore   = $derived(gpuPasses.length ? Math.max(...gpuPasses.map(p => p.score)) : 0);

	// ── Plan Upgrades ────────────────────────────────────────────────────────────
	let planUpgrades   = $state([]);   // [{from, to, score, round}]

	// ── Post-optimize ────────────────────────────────────────────────────────────
	let postOptStart   = $state(null);  // { duration, current_score }
	let postOptProgress = $state(null); // { elapsed, remaining, score, source }
	let postOptFinal   = $state(null);  // final score after post-opt

	// ── Pipeline done ─────────────────────────────────────────────────────────────
	let pipelineDone   = $state(null);  // { final_score, difficulty, replay_ready }

	// ── Log ──────────────────────────────────────────────────────────────────────
	let logs           = $state([]);

	// ── Grid derived ─────────────────────────────────────────────────────────────
	let wallSet = $derived(gameInit
		? new Set((gameInit.walls || []).map(w => `${w[0]},${w[1]}`))
		: new Set());
	let shelfSet = $derived(gameInit
		? new Set((gameInit.shelves || []).map(s => `${s[0]},${s[1]}`))
		: new Set());
	let itemMap = $derived.by(() => {
		if (!gameInit) return new Map();
		const m = new Map();
		for (const item of gameInit.items || []) {
			const key = `${item.position[0]},${item.position[1]}`;
			if (!m.has(key)) m.set(key, []);
			m.get(key).push(item);
		}
		return m;
	});
	// item ID → type name map for inventory display
	let itemIdToType = $derived.by(() => {
		if (!gameInit) return {};
		const m = {};
		for (const item of gameInit.items || []) m[item.id] = item.type;
		return m;
	});
	// Bots with inventory converted from IDs to type names
	let botsTyped = $derived(bots.map(b => ({
		...b,
		inventory: (b.inventory || []).map(id => itemIdToType[id] || id),
	})));
	let botPositions = $derived(new Map(botsTyped.map(b => [`${b.position[0]},${b.position[1]}`, b])));
	let selectedBot  = $state(null);

	// ── JWT parse ────────────────────────────────────────────────────────────────
	function parseToken(url) {
		try {
			const m = url.match(/[?&]token=([^&]+)/);
			if (!m) return null;
			const parts = m[1].split('.');
			if (parts.length < 2) return null;
			return JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));
		} catch { return null; }
	}
	let tokenInfo = $derived(wsUrl.trim() ? parseToken(wsUrl) : null);

	// ── Helpers ──────────────────────────────────────────────────────────────────
	function addLog(text) {
		if (!text) return;
		logs = [...logs.slice(-200), { t: new Date().toLocaleTimeString(), text }];
	}

	function resetAll() {
		gameInit = null; currentRound = 0; maxRounds = 300; liveScore = 0;
		planScore = 0; planSource = 'none'; bots = []; orders = [];
		finalScore = null; scoreHistory = []; gpuPasses = [];
		planUpgrades = []; postOptStart = null; postOptProgress = null;
		postOptFinal = null; pipelineDone = null; logs = [];
	}

	// ── Event Handler ─────────────────────────────────────────────────────────────
	function handleEvent(event) {
		switch (event.type) {
			case 'init':
				gameInit = event;
				maxRounds = event.max_rounds || 300;
				phase = 'playing';
				break;

			case 'round':
				currentRound = event.round;
				liveScore = event.score || 0;
				planScore = event.plan_score || 0;
				planSource = event.plan_source || 'none';
				bots = event.bots || [];
				orders = event.orders || [];
				scoreHistory = [...scoreHistory, {
					round: event.round,
					score: event.score || 0,
					planScore: event.plan_score || 0,
				}];
				break;

			case 'game_over':
				finalScore = event.score;
				if (phase === 'playing') phase = 'post_optimizing';
				break;

			case 'plan_upgrade':
				planUpgrades = [...planUpgrades, {
					from: event.from_source,
					to: event.to_source,
					score: event.score_estimate,
					round: currentRound,
				}];
				planSource = event.to_source;
				addLog(`Plan: ${event.from_source} → ${event.to_source} (est. ${event.score_estimate})`);
				break;

			case 'gpu_pass_done':
				gpuPasses = [...gpuPasses, event];
				addLog(`GPU Pass ${event.pass}: ${event.score} pts  (${event.max_states.toLocaleString()} states, ${event.elapsed}s)`);
				break;

			case 'post_optimize_start':
				postOptStart = event;
				phase = 'post_optimizing';
				addLog(`Post-optimize: ${event.duration}s  current score: ${event.current_score}`);
				break;

			case 'post_optimize_progress':
				postOptProgress = event;
				planScore = event.score || planScore;
				break;

			case 'post_optimize_done':
				postOptFinal = event.final_score;
				addLog(`Post-optimize done: score=${event.final_score}`);
				break;

			case 'pipeline_done':
				pipelineDone = event;
				phase = 'done';
				running = false;
				addLog(`Pipeline complete! Final score: ${event.final_score}`);
				break;

			case 'log':
				addLog(event.text);
				break;

			case 'status':
				addLog(event.message);
				break;

			case 'error':
				addLog(`ERROR: ${event.message}`);
				running = false;
				break;
		}
	}

	// ── SSE Stream ───────────────────────────────────────────────────────────────
	async function startPipeline() {
		if (!wsUrl.trim()) return;
		if (abortCtrl) abortCtrl.abort();
		resetAll();
		running = true;
		phase = 'playing';
		abortCtrl = new AbortController();

		try {
			const res = await fetch('/api/pipeline/run', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ url: wsUrl.trim(), postOptimizeTime: postOptTime }),
				signal: abortCtrl.signal,
			});

			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				const lines = buf.split('\n');
				buf = lines.pop() || '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try { handleEvent(JSON.parse(line.slice(6))); } catch (e) {}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') addLog(`Connection error: ${e.message}`);
		}
		if (phase !== 'done') phase = 'done';
		running = false;
	}

	async function startReplay() {
		if (!replayUrl.trim() && !wsUrl.trim()) return;
		const url = replayUrl.trim() || wsUrl.trim();
		const diff = pipelineDone?.difficulty || tokenInfo?.difficulty || null;

		// Reset game view
		bots = []; orders = []; currentRound = 0; liveScore = 0;
		phase = 'replaying';
		running = true;
		if (abortCtrl) abortCtrl.abort();
		abortCtrl = new AbortController();

		try {
			const res = await fetch('/api/optimize/replay', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ url, difficulty: diff }),
				signal: abortCtrl.signal,
			});

			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				const lines = buf.split('\n');
				buf = lines.pop() || '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try { handleEvent(JSON.parse(line.slice(6))); } catch (e) {}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') addLog(`Replay error: ${e.message}`);
		}
		phase = 'done';
		running = false;
	}

	function stop() {
		if (abortCtrl) abortCtrl.abort();
		running = false;
		addLog('Stopped.');
	}

	function fmtSecs(s) {
		if (!s && s !== 0) return '--';
		const m = Math.floor(s / 60);
		const ss = Math.floor(s % 60);
		return `${m}:${ss.toString().padStart(2, '0')}`;
	}

	let diffColors = {
		easy: '#00b894', medium: '#fdcb6e', hard: '#e17055', expert: '#e74c3c',
	};
</script>

<div class="page">
	<div class="header">
		<div>
			<h1>GPU Pipeline</h1>
			<p class="sub">Live game → GPU optimization → Best replay</p>
		</div>
		<div class="phase-badge" class:playing={phase === 'playing'}
			class:post={phase === 'post_optimizing'} class:done={phase === 'done'}
			class:replaying={phase === 'replaying'}>
			{#if phase === 'idle'}Idle
			{:else if phase === 'playing'}● Playing
			{:else if phase === 'post_optimizing'}⚡ Optimizing
			{:else if phase === 'replaying'}▶ Replaying
			{:else}✓ Done
			{/if}
		</div>
	</div>

	<!-- ── Input Panel ──────────────────────────────────────────────────────── -->
	<div class="panel">
		<div class="url-row">
			<input
				type="text"
				bind:value={wsUrl}
				placeholder="wss://game.ainm.no/ws?token=..."
				class="url-input"
				disabled={running}
			/>
			{#if tokenInfo?.difficulty}
				<span class="badge" style="background:{diffColors[tokenInfo.difficulty]||'#666'}">{tokenInfo.difficulty}</span>
			{/if}
		</div>

		<div class="controls-row">
			<label class="ctrl-label">
				Post-optimize:
				<input type="number" bind:value={postOptTime} min="0" max="7200" step="60"
					class="num-input" disabled={running} />s
			</label>

			{#if running}
				<button class="btn btn-stop" onclick={stop}>Stop</button>
				<span class="running-pill">
					{phase === 'playing' ? 'Playing game...' :
					 phase === 'post_optimizing' ? 'GPU optimizing...' :
					 phase === 'replaying' ? 'Replaying...' : 'Running...'}
				</span>
			{:else}
				<button class="btn btn-start" disabled={!wsUrl.trim()} onclick={startPipeline}>
					Start Pipeline
				</button>
			{/if}
		</div>
	</div>

	<!-- ── Stats Bar ────────────────────────────────────────────────────────── -->
	{#if phase !== 'idle'}
	<div class="stats-bar">
		<div class="stat">
			<span class="stat-label">Score</span>
			<span class="stat-val green">{liveScore}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Plan</span>
			<span class="stat-val" style="color:{sourceColor(planSource)}">{planScore || '—'}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Source</span>
			<span class="stat-src" style="color:{sourceColor(planSource)}">{planSource}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Round</span>
			<span class="stat-val">{currentRound}/{maxRounds}</span>
		</div>
		{#if bestGpuScore}
		<div class="stat">
			<span class="stat-label">GPU Best</span>
			<span class="stat-val accent">{bestGpuScore}</span>
		</div>
		{/if}
		{#if finalScore !== null}
		<div class="stat">
			<span class="stat-label">Final</span>
			<span class="stat-val gold">{finalScore}</span>
		</div>
		{/if}
	</div>
	{/if}

	<!-- ── Progress Bar ─────────────────────────────────────────────────────── -->
	{#if phase === 'playing' || phase === 'replaying'}
	<div class="progress-bar">
		<div class="progress-fill" style="width:{(currentRound/maxRounds)*100}%"></div>
		<span class="progress-txt">R{currentRound} / {maxRounds}</span>
	</div>
	{/if}

	<!-- ── Main Grid + Sidebar ───────────────────────────────────────────────── -->
	{#if gameInit}
	<div class="game-area">
		<div class="grid-wrap">
			<Grid
				width={gameInit.width}
				height={gameInit.height}
				cellSize={CELL}
				{wallSet}
				{shelfSet}
				{itemMap}
				dropOff={gameInit.drop_off}
				spawn={gameInit.spawn}
				bots={botsTyped}
				{botPositions}
				botColors={BOT_COLORS}
				{selectedBot}
				onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
			/>
		</div>

		<div class="sidebar">
			<!-- Orders -->
			<div class="sidebar-section">
				<h3>Orders</h3>
				{#each orders as order}
					<div class="order" class:active={order.status === 'active'} class:preview={order.status === 'preview'}>
						<span class="order-tag">{order.status}</span>
						<div class="order-items">
							{#each order.items_required as item, j}
								<span class="oitem" class:del={order.items_delivered?.[j]}>{item}</span>
							{/each}
						</div>
					</div>
				{/each}
				{#if orders.length === 0}
					<div class="muted">Waiting for orders...</div>
				{/if}
			</div>

			<!-- GPU Passes -->
			{#if gpuPasses.length > 0}
			<div class="sidebar-section">
				<h3>GPU Passes</h3>
				{#each gpuPasses as p}
					<div class="gpu-pass" class:best={p.score === bestGpuScore}>
						<span class="pass-num">P{p.pass}</span>
						<span class="pass-states">{(p.max_states/1000).toFixed(0)}K</span>
						<span class="pass-score" class:best={p.score === bestGpuScore}>{p.score}</span>
						<span class="pass-time muted">{p.elapsed}s</span>
					</div>
				{/each}
			</div>
			{/if}

			<!-- Selected bot detail -->
			{#if selectedBot !== null}
				{@const bot = botsTyped.find(b => b.id === selectedBot)}
				{#if bot}
				<div class="sidebar-section">
					<h3>Bot {bot.id}</h3>
					<div class="bot-detail">
						<div>Pos: {bot.position[0]},{bot.position[1]}</div>
						<div>Inv: {bot.inventory.join(', ') || 'empty'}</div>
					</div>
				</div>
				{/if}
			{/if}
		</div>
	</div>
	{/if}

	<!-- ── Score Chart ───────────────────────────────────────────────────────── -->
	{#if scoreHistory.length > 1}
	<div class="panel chart-panel">
		<h3>Score over Rounds</h3>
		<svg viewBox="0 0 {CHART_W} {CHART_H}" class="chart-svg" preserveAspectRatio="none">
			<!-- Grid lines -->
			{#each [0.25, 0.5, 0.75] as frac}
				<line x1="0" y1={CHART_H * frac} x2={CHART_W} y2={CHART_H * frac}
					stroke="#2a2e3d" stroke-width="1"/>
			{/each}
			<!-- Plan score line (dashed) -->
			{#if chartPlanLine}
				<polyline points={chartPlanLine} fill="none" stroke="#a29bfe"
					stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7"/>
			{/if}
			<!-- Live score line -->
			<polyline points={chartPolyline} fill="none" stroke="#00b894" stroke-width="2"/>
			<!-- Current point -->
			{#if chartLastPt}
				<circle
					cx={(chartLastPt.round / maxRounds) * CHART_W}
					cy={CHART_H - (chartLastPt.score / chartMaxScore) * CHART_H}
					r="3" fill="#00b894"/>
			{/if}
		</svg>
		<div class="chart-legend">
			<span style="color:#00b894">— live score</span>
			<span style="color:#a29bfe; margin-left:1rem">- - plan score</span>
		</div>
	</div>
	{/if}

	<!-- ── Plan Upgrades Timeline ────────────────────────────────────────────── -->
	{#if planUpgrades.length > 0}
	<div class="panel">
		<h3>Plan Upgrades</h3>
		<div class="timeline">
			{#each planUpgrades as u, i}
				<div class="tl-item">
					<div class="tl-dot" style="background:{sourceColor(u.to)}"></div>
					<div class="tl-content">
						<span class="tl-src" style="color:{sourceColor(u.to)}">{u.to}</span>
						{#if u.score}
							<span class="tl-score">{u.score} pts</span>
						{/if}
						<span class="tl-round muted">R{u.round}</span>
					</div>
					{#if i < planUpgrades.length - 1}
						<div class="tl-line"></div>
					{/if}
				</div>
			{/each}
		</div>
	</div>
	{/if}

	<!-- ── Post-Optimize Progress ────────────────────────────────────────────── -->
	{#if postOptStart && phase === 'post_optimizing'}
	<div class="panel opt-panel">
		<h3>Post-Game Optimization</h3>
		<div class="opt-row">
			<div class="opt-stat">
				<span class="opt-label">Current Best</span>
				<span class="opt-val green">{postOptProgress?.score || postOptStart.current_score}</span>
			</div>
			<div class="opt-stat">
				<span class="opt-label">Elapsed</span>
				<span class="opt-val">{fmtSecs(postOptProgress?.elapsed)}</span>
			</div>
			<div class="opt-stat">
				<span class="opt-label">Remaining</span>
				<span class="opt-val accent">{fmtSecs(postOptProgress?.remaining)}</span>
			</div>
			{#if postOptProgress?.source}
			<div class="opt-stat">
				<span class="opt-label">Source</span>
				<span class="opt-val" style="color:{sourceColor(postOptProgress.source)}">{postOptProgress.source}</span>
			</div>
			{/if}
		</div>
		{#if postOptProgress}
		<div class="progress-bar" style="margin-top:0.5rem">
			<div class="progress-fill accent"
				style="width:{(postOptProgress.elapsed / postOptStart.duration) * 100}%"></div>
		</div>
		{/if}
	</div>
	{/if}

	<!-- ── Pipeline Done ─────────────────────────────────────────────────────── -->
	{#if pipelineDone}
	<div class="panel done-panel">
		<div class="done-score">
			<span class="done-label">Final Score</span>
			<span class="done-val">{pipelineDone.final_score}</span>
			{#if pipelineDone.difficulty}
				<span class="badge" style="background:{diffColors[pipelineDone.difficulty]||'#666'}">{pipelineDone.difficulty}</span>
			{/if}
		</div>
		{#if pipelineDone.replay_ready}
		<div class="replay-section">
			<p class="muted">Solution saved. Replay with a new token:</p>
			<div class="url-row" style="margin-top:0.5rem">
				<input
					type="text"
					bind:value={replayUrl}
					placeholder="wss://... (new token for replay)"
					class="url-input"
					disabled={running}
				/>
			</div>
			<button class="btn btn-replay" disabled={running} onclick={startReplay}>
				▶ Replay Best Score
			</button>
		</div>
		{/if}
	</div>
	{/if}

	<!-- ── Log ───────────────────────────────────────────────────────────────── -->
	<div class="panel log-panel">
		<h3>Log <span class="muted" style="font-weight:400; font-size:0.75rem">({logs.length} entries)</span></h3>
		<div class="log-box">
			{#each logs.slice().reverse() as entry}
				<div class="log-line">
					<span class="log-t">{entry.t}</span>
					<span>{entry.text}</span>
				</div>
			{/each}
			{#if logs.length === 0}
				<div class="muted">No activity yet</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.page {
		max-width: 1400px;
		margin: 0 auto;
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.header {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
	}
	h1 { font-size: 1.5rem; margin: 0 0 0.2rem; }
	.sub { color: var(--text-muted); font-size: 0.82rem; margin: 0; }

	/* Phase badge */
	.phase-badge {
		padding: 0.3rem 0.8rem;
		border-radius: 20px;
		font-size: 0.8rem;
		font-weight: 600;
		background: var(--bg-card);
		border: 1px solid var(--border);
		color: var(--text-muted);
	}
	.phase-badge.playing    { border-color: #00b894; color: #00b894; }
	.phase-badge.post       { border-color: #a29bfe; color: #a29bfe; }
	.phase-badge.done       { border-color: #fdcb6e; color: #fdcb6e; }
	.phase-badge.replaying  { border-color: #74b9ff; color: #74b9ff; }

	/* Panel */
	.panel {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 1rem;
	}

	/* Input */
	.url-row {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		margin-bottom: 0.5rem;
	}
	.url-input {
		flex: 1;
		padding: 0.45rem 0.75rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
		font-size: 0.82rem;
		font-family: inherit;
	}
	.url-input:focus { outline: none; border-color: var(--accent); }
	.controls-row {
		display: flex;
		gap: 0.75rem;
		align-items: center;
		flex-wrap: wrap;
	}
	.ctrl-label {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		font-size: 0.8rem;
		color: var(--text-muted);
	}
	.num-input {
		width: 70px;
		padding: 0.3rem 0.5rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
		font-size: 0.8rem;
	}
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		font-size: 0.72rem;
		font-weight: 600;
		color: #000;
		white-space: nowrap;
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
	}
	.btn:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-start  { background: #00b894; color: #000; }
	.btn-stop   { background: var(--red); color: #fff; }
	.btn-replay { background: #74b9ff; color: #000; margin-top: 0.5rem; }
	.running-pill {
		font-size: 0.8rem;
		color: var(--accent-light);
		animation: pulse 1.5s ease-in-out infinite;
	}

	/* Stats bar */
	.stats-bar {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}
	.stat {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.4rem 0.75rem;
		display: flex;
		flex-direction: column;
		align-items: center;
		min-width: 70px;
	}
	.stat-label {
		font-size: 0.65rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.stat-val { font-size: 1.1rem; font-weight: 700; }
	.stat-src { font-size: 0.78rem; font-weight: 600; }
	.green  { color: #00b894; }
	.accent { color: var(--accent-light); }
	.gold   { color: #fdcb6e; }

	/* Progress */
	.progress-bar {
		height: 6px;
		background: var(--bg-card);
		border-radius: 3px;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: var(--green);
		transition: width 0.2s;
	}
	.progress-fill.accent { background: var(--accent-light); }
	.progress-txt {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-align: center;
		margin-top: 0.2rem;
	}

	/* Game area */
	.game-area {
		display: flex;
		gap: 1rem;
		align-items: flex-start;
	}
	.grid-wrap { flex-shrink: 0; }
	.sidebar {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		min-width: 180px;
		max-width: 280px;
	}
	.sidebar-section h3 {
		font-size: 0.82rem;
		font-weight: 600;
		margin: 0 0 0.4rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	/* Orders */
	.order {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0.35rem 0.5rem;
		margin-bottom: 0.3rem;
	}
	.order.active  { border-color: #00b894; }
	.order.preview { border-color: #fdcb6e; opacity: 0.8; }
	.order-tag {
		font-size: 0.6rem;
		font-weight: 700;
		text-transform: uppercase;
		color: var(--text-muted);
		display: block;
		margin-bottom: 0.2rem;
	}
	.order-items { display: flex; flex-wrap: wrap; gap: 0.2rem; }
	.oitem {
		font-size: 0.68rem;
		padding: 0.1rem 0.3rem;
		background: var(--bg-card);
		border-radius: 3px;
	}
	.oitem.del { background: #00b894; color: #000; }

	/* GPU passes */
	.gpu-pass {
		display: flex;
		gap: 0.5rem;
		align-items: center;
		padding: 0.25rem 0.4rem;
		border-radius: 4px;
		font-size: 0.78rem;
		margin-bottom: 0.2rem;
	}
	.gpu-pass.best { background: rgba(0,184,148,0.1); border: 1px solid #00b894; }
	.pass-num   { font-weight: 700; color: var(--text-muted); min-width: 20px; }
	.pass-states { color: var(--text-muted); font-size: 0.72rem; min-width: 30px; }
	.pass-score { font-weight: 700; }
	.pass-score.best { color: #00b894; }
	.pass-time  { margin-left: auto; }

	/* Bot detail */
	.bot-detail { font-size: 0.78rem; color: var(--text-muted); }

	/* Score chart */
	.chart-panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; color: var(--text-muted); }
	.chart-svg {
		width: 100%;
		height: 100px;
		background: var(--bg);
		border-radius: 4px;
		display: block;
	}
	.chart-legend { font-size: 0.72rem; margin-top: 0.3rem; color: var(--text-muted); }

	/* Timeline */
	.panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; color: var(--text-muted); }
	.timeline {
		display: flex;
		flex-direction: row;
		flex-wrap: wrap;
		gap: 0.4rem;
		align-items: center;
	}
	.tl-item {
		display: flex;
		align-items: center;
		gap: 0.3rem;
	}
	.tl-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
	.tl-content { display: flex; gap: 0.3rem; align-items: baseline; flex-wrap: wrap; }
	.tl-src  { font-size: 0.78rem; font-weight: 600; }
	.tl-score { font-size: 0.72rem; color: #00b894; }
	.tl-round { font-size: 0.68rem; }
	.tl-line {
		width: 16px;
		height: 1px;
		background: var(--border);
	}

	/* Post-optimize */
	.opt-panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; }
	.opt-row {
		display: flex;
		gap: 1rem;
		flex-wrap: wrap;
	}
	.opt-stat {
		display: flex;
		flex-direction: column;
		align-items: center;
		min-width: 80px;
	}
	.opt-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }
	.opt-val   { font-size: 1.2rem; font-weight: 700; }

	/* Done panel */
	.done-panel { border-color: #fdcb6e; }
	.done-score {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.75rem;
	}
	.done-label { color: var(--text-muted); font-size: 0.85rem; }
	.done-val   { font-size: 2.5rem; font-weight: 800; color: #fdcb6e; }
	.replay-section {}

	/* Log */
	.log-panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; }
	.log-box {
		background: var(--bg);
		border-radius: 4px;
		padding: 0.5rem;
		max-height: 200px;
		overflow-y: auto;
		font-size: 0.72rem;
	}
	.log-line {
		display: flex;
		gap: 0.5rem;
		padding: 0.1rem 0;
		border-bottom: 1px solid var(--border);
	}
	.log-line:last-child { border-bottom: none; }
	.log-t { color: var(--text-muted); white-space: nowrap; flex-shrink: 0; }
	.muted { color: var(--text-muted); }

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50%       { opacity: 0.4; }
	}

	@media (max-width: 900px) {
		.game-area { flex-direction: column; }
		.sidebar { max-width: 100%; }
	}
</style>
