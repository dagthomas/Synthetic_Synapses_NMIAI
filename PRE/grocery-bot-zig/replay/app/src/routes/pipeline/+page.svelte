<script>
	import Grid from '$lib/components/Grid.svelte';

	const CELL = 24;
	const BOT_COLORS = [
		'#f85149', '#58a6ff', '#39d353', '#d29922', '#bc8cff',
		'#3fb950', '#db6d28', '#8b949e', '#f778ba', '#79c0ff',
	];
	const SOURCE_COLORS = {
		greedy: '#8b949e',
		mapf:   '#d29922',
		gpu_pass_0: '#58a6ff',
		gpu_pass_1: '#56d364',
		gpu_pass_2: '#39d353',
		gpu_pass_3: '#f85149',
		gpu_refine: '#39d353',
		none:   '#484f58',
	};
	const TARGETS = { easy: 142, medium: 214, hard: 252, expert: 303 };
	function sourceColor(s) {
		return SOURCE_COLORS[s] || '#74b9ff';
	}

	// ── Mode ─────────────────────────────────────────────────────────────────────
	let mode = $state('single');  // 'single' | 'iterate'

	// ── Input State ──────────────────────────────────────────────────────────────
	let wsUrl          = $state('');
	let replayUrl      = $state('');
	let postOptTime    = $state(60);
	let running        = $state(false);
	let phase          = $state('idle'); // idle | playing | post_optimizing | replaying | done
	let abortCtrl      = $state(null);
	let inputCollapsed = $state(false);

	// ── Iteration config ─────────────────────────────────────────────────────────
	let maxIters       = $state(99);   // safety cap; time budget is the real limit
	let timeBudget     = $state(280);  // seconds
	let gpuOptTime     = $state(45);   // seconds per offline GPU optimize pass

	// ── Timer ────────────────────────────────────────────────────────────────────
	let timerStart     = $state(null);
	let timerNow       = $state(null);
	let timerInterval  = $state(null);

	let timerElapsed = $derived(timerStart && timerNow ? (timerNow - timerStart) / 1000 : 0);
	let timerRemaining = $derived(Math.max(0, timeBudget - timerElapsed));
	let timerPct = $derived(timerElapsed / timeBudget * 100);

	function startTimer() {
		timerStart = Date.now();
		timerNow = Date.now();
		if (timerInterval) clearInterval(timerInterval);
		timerInterval = setInterval(() => { timerNow = Date.now(); }, 250);
	}
	function stopTimer() {
		if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
	}

	// ── Game State ───────────────────────────────────────────────────────────────
	let gameInit       = $state(null);
	let currentRound   = $state(0);
	let maxRounds      = $state(300);
	let liveScore      = $state(0);
	let planScore      = $state(0);
	let planSource     = $state('none');
	let bots           = $state([]);
	let orders         = $state([]);
	let finalScore     = $state(null);

	// ── Score Chart ──────────────────────────────────────────────────────────────
	let scoreHistory   = $state([]);
	const CHART_W = 480, CHART_H = 120;

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
	let gpuPasses      = $state([]);
	let bestGpuScore   = $derived(gpuPasses.length ? Math.max(...gpuPasses.map(p => p.score)) : 0);

	// ── Plan Upgrades ────────────────────────────────────────────────────────────
	let planUpgrades   = $state([]);

	// ── Post-optimize ────────────────────────────────────────────────────────────
	let postOptStart   = $state(null);
	let postOptProgress = $state(null);
	let postOptFinal   = $state(null);

	// ── Pipeline done ─────────────────────────────────────────────────────────────
	let pipelineDone   = $state(null);

	// ── Iteration tracking ───────────────────────────────────────────────────────
	let iterations     = $state([]);
	let activeIter     = $state(0);
	let currentIterData = $state(null); // current iteration's accumulated data
	let iterBestScore  = $derived(iterations.length ? Math.max(...iterations.map(i => i.score || 0)) : 0);
	let iterTotalTime  = $derived(iterations.reduce((s, i) => s + (i.elapsed || 0), 0));
	let detectedDiff   = $state(null);

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
	let itemIdToType = $derived.by(() => {
		if (!gameInit) return {};
		const m = {};
		for (const item of gameInit.items || []) m[item.id] = item.type;
		return m;
	});
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
	let targetScore = $derived(detectedDiff ? (TARGETS[detectedDiff] || 175) : (tokenInfo?.difficulty ? (TARGETS[tokenInfo.difficulty] || 175) : 175));

	// ── Helpers ──────────────────────────────────────────────────────────────────
	function addLog(text, iter = null) {
		if (!text) return;
		logs = [...logs.slice(-300), { t: new Date().toLocaleTimeString(), text, iter }];
	}

	function resetAll() {
		gameInit = null; currentRound = 0; maxRounds = 300; liveScore = 0;
		planScore = 0; planSource = 'none'; bots = []; orders = [];
		finalScore = null; scoreHistory = []; gpuPasses = [];
		planUpgrades = []; postOptStart = null; postOptProgress = null;
		postOptFinal = null; pipelineDone = null; logs = [];
		iterations = []; activeIter = 0; currentIterData = null;
		detectedDiff = null;
	}

	function resetIterState() {
		// Reset per-iteration state but keep iterations array
		gameInit = null; currentRound = 0; liveScore = 0;
		planScore = 0; planSource = 'none'; bots = []; orders = [];
		finalScore = null; scoreHistory = []; gpuPasses = [];
		planUpgrades = []; postOptStart = null; postOptProgress = null;
		postOptFinal = null; pipelineDone = null;
	}

	// ── Event Handler ─────────────────────────────────────────────────────────────
	function handleEvent(event) {
		const iter = event._iter ?? null;

		switch (event.type) {
			case 'init':
				gameInit = event;
				maxRounds = event.max_rounds || 300;
				phase = 'playing';
				if (event.difficulty) detectedDiff = event.difficulty;
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
				addLog(`Plan: ${event.from_source} -> ${event.to_source} (est. ${event.score_estimate})`, iter);
				break;

			case 'gpu_pass_done':
				gpuPasses = [...gpuPasses, event];
				addLog(`GPU Pass ${event.pass}: ${event.score} pts (${(event.max_states||0).toLocaleString()} states, ${event.elapsed}s)`, iter);
				break;

			case 'post_optimize_start':
				postOptStart = event;
				phase = 'post_optimizing';
				addLog(`Post-optimize: ${event.duration}s, current=${event.current_score}`, iter);
				break;

			case 'post_optimize_progress':
				postOptProgress = event;
				planScore = event.score || planScore;
				break;

			case 'post_optimize_done':
				postOptFinal = event.final_score;
				addLog(`Post-optimize done: score=${event.final_score}`, iter);
				break;

			case 'pipeline_done':
				pipelineDone = event;
				if (mode === 'single') {
					phase = 'done';
					running = false;
					stopTimer();
				}
				addLog(`Pipeline done! Final=${event.final_score}`, iter);
				break;

			// ── Iteration events ──────────────────────────────────────────────
			case 'pipeline_start':
				addLog(`Iterate pipeline: ${event.time_budget}s budget`);
				break;

			case 'iter_start':
				activeIter = event.iter;
				resetIterState();
				iterations = [...iterations, {
					iter: event.iter,
					phase: event.phase || 'playing',
					score: 0,
					gameScore: 0,
					optScore: 0,
					elapsed: 0,
					orders: 0,
					startTime: event.elapsed,
				}];
				phase = event.phase === 'optimize_replay' ? 'optimizing' : 'playing';
				addLog(`--- Iter ${event.iter + 1} (${event.phase || 'live'}) ${Math.floor(event.remaining)}s left ---`, event.iter);
				break;

			case 'optimize_phase_start':
				phase = 'optimizing';
				addLog(`GPU optimize: ${event.difficulty}, budget=${event.max_time}s`, iter);
				break;

			case 'optimize_start':
				addLog(`Optimizing: ${event.orders} orders, prev=${event.prev_score}`, iter);
				break;

			case 'optimize_done':
				addLog(`Optimize: ${event.score} (was ${event.prev_score}) in ${event.elapsed}s`, iter);
				if (event.score > planScore) planScore = event.score;
				break;

			case 'optimize_error':
				addLog(`Optimize error: ${event.message}`, iter);
				break;

			case 'replay_phase_start':
				phase = 'replaying';
				addLog(`Replaying optimized solution...`, iter);
				break;

			case 'replay_phase_done':
				addLog(`Replay score: ${event.score}`, iter);
				break;

			case 'iter_done': {
				const idx = iterations.findIndex(i => i.iter === event.iter);
				if (idx >= 0) {
					const updated = [...iterations];
					updated[idx] = {
						...updated[idx],
						phase: 'done',
						score: event.score || 0,
						gameScore: event.game_score || 0,
						optScore: event.opt_score || 0,
						orders: event.orders || 0,
						capturedOrders: event.captured_orders || 0,
						elapsed: event.elapsed - (updated[idx].startTime || 0),
						difficulty: event.difficulty,
					};
					iterations = updated;
				}
				addLog(`Iter ${event.iter + 1}: score=${event.score} (${Math.floor(event.remaining)}s left)`, event.iter);
				break;
			}

			case 'iter_skip':
				addLog(`Iter ${event.iter + 1} skipped: ${event.reason}`, event.iter);
				break;

			case 'iter_summary':
				addLog(`Best: ${event.best_score} after ${event.iterations_done} iters`, event.iter);
				break;

			case 'pipeline_complete':
				phase = 'done';
				running = false;
				stopTimer();
				addLog(`Complete! Best=${event.best_score} in ${event.iterations} iters, ${Math.floor(event.total_elapsed)}s`);
				break;

			case 'seed_cracked':
				addLog(`Seed cracked: ${event.seed}`, iter);
				break;

			case 'log':
				addLog(event.text, iter);
				break;

			case 'status':
				addLog(event.message, iter);
				break;

			case 'error':
				addLog(`ERROR: ${event.message}`, iter);
				if (mode === 'single') {
					running = false;
					stopTimer();
				}
				break;
		}
	}

	// ── SSE Stream (single mode) ─────────────────────────────────────────────────
	async function startPipeline() {
		if (!wsUrl.trim()) return;
		if (abortCtrl) abortCtrl.abort();
		resetAll();
		running = true;
		phase = 'playing';
		mode = 'single';
		inputCollapsed = true;
		abortCtrl = new AbortController();
		startTimer();

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
		stopTimer();
	}

	// ── SSE Stream (iterate mode) ────────────────────────────────────────────────
	async function startIterate() {
		if (!wsUrl.trim()) return;
		if (abortCtrl) abortCtrl.abort();
		resetAll();
		running = true;
		phase = 'playing';
		mode = 'iterate';
		inputCollapsed = true;
		abortCtrl = new AbortController();
		startTimer();

		try {
			const res = await fetch('/api/pipeline/iterate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					url: wsUrl.trim(),
					timeBudget: timeBudget,
					postOptimizeTime: postOptTime,
					gpuOptimizeTime: gpuOptTime,
				}),
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
		stopTimer();
	}

	async function startReplay() {
		if (!replayUrl.trim() && !wsUrl.trim()) return;
		const url = replayUrl.trim() || wsUrl.trim();
		const diff = pipelineDone?.difficulty || detectedDiff || tokenInfo?.difficulty || null;

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
		stopTimer();
		addLog('Stopped.');
	}

	function fmtSecs(s) {
		if (!s && s !== 0) return '--';
		const m = Math.floor(s / 60);
		const ss = Math.floor(s % 60);
		return `${m}:${ss.toString().padStart(2, '0')}`;
	}

	let diffColors = {
		easy: '#39d353', medium: '#d29922', hard: '#f85149', expert: '#da3633',
	};
</script>

<div class="page stagger">
	<div class="header">
		<div>
			<h1>GPU Pipeline</h1>
			<p class="sub">Live game -> GPU optimization -> Iterative replay</p>
		</div>
		<div class="header-right">
			{#if detectedDiff}
				<span class="diff-badge" style="background:{diffColors[detectedDiff]||'#666'}">{detectedDiff}</span>
				<span class="target-label">Target: {targetScore}</span>
			{/if}
			<div class="phase-badge" class:playing={phase === 'playing'}
				class:post={phase === 'post_optimizing' || phase === 'optimizing'}
				class:done={phase === 'done'}
				class:replaying={phase === 'replaying'}>
				{#if phase === 'idle'}Idle
				{:else if phase === 'playing'}🎮 Playing
				{:else if phase === 'optimizing'}🔥 GPU Optimizing
				{:else if phase === 'post_optimizing'}🔥 Post-Optimizing
				{:else if phase === 'replaying'}▶ Replaying
				{:else}✓ Done
				{/if}
			</div>
		</div>
	</div>

	<!-- ── Timer Bar ──────────────────────────────────────────────────────── -->
	{#if running && timerStart}
	<div class="timer-bar">
		<div class="timer-track">
			<div class="timer-fill" class:warn={timerRemaining < 60} class:danger={timerRemaining < 30}
				style="width:{Math.min(100, timerPct)}%"></div>
		</div>
		<div class="timer-labels">
			<span class="timer-elapsed">{fmtSecs(timerElapsed)}</span>
			<span class="timer-remaining" class:warn={timerRemaining < 60} class:danger={timerRemaining < 30}>
				{fmtSecs(timerRemaining)} remaining
			</span>
			<span class="timer-budget">{fmtSecs(timeBudget)}</span>
		</div>
	</div>
	{/if}

	<!-- ── Input Panel ──────────────────────────────────────────────────────── -->
	<div class="panel input-panel" class:collapsed={inputCollapsed && phase !== 'idle'}>
		{#if inputCollapsed && phase !== 'idle'}
			<button class="expand-btn" onclick={() => inputCollapsed = false}>
				Show controls ({wsUrl.slice(-20)}...)
			</button>
		{:else}
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
			<!-- Mode toggle -->
			<div class="mode-toggle">
				<button class="mode-btn" class:active={mode === 'single'} disabled={running}
					onclick={() => { mode = 'single'; postOptTime = 120; }}>Single</button>
				<button class="mode-btn" class:active={mode === 'iterate'} disabled={running}
					onclick={() => { mode = 'iterate'; postOptTime = 60; }}>Iterate</button>
			</div>

			<label class="ctrl-label" title="GPU optimization time after game ends (per iteration)">
				Post-opt:
				<input type="number" bind:value={postOptTime} min="0" max="3600" step="10"
					class="num-input" disabled={running} />s
			</label>

			{#if mode === 'iterate'}
				<label class="ctrl-label" title="Total time budget — loops until this expires">
					Budget:
					<input type="number" bind:value={timeBudget} min="60" max="600" step="10"
						class="num-input" disabled={running} />s
				</label>
				<label class="ctrl-label" title="GPU optimize time per iteration (offline DP solver)">
					GPU:
					<input type="number" bind:value={gpuOptTime} min="10" max="120" step="5"
						class="num-input sm" disabled={running} />s
				</label>
			{/if}

			{#if running}
				<button class="btn btn-stop" onclick={stop}>Stop</button>
				<span class="running-pill">
					{phase === 'playing' ? '🎮 Playing...' :
					 phase === 'optimizing' ? '🔥 GPU optimizing...' :
					 phase === 'post_optimizing' ? '🔥 Post-optimizing...' :
					 phase === 'replaying' ? '▶ Replaying...' : 'Running...'}
					{#if mode === 'iterate' && iterations.length > 0}
						(iter {activeIter + 1}, {fmtSecs(timerRemaining)} left)
					{/if}
				</span>
			{:else}
				{#if mode === 'single'}
					<button class="btn btn-start" disabled={!wsUrl.trim()} onclick={startPipeline}>
						Start Pipeline
					</button>
				{:else}
					<button class="btn btn-iterate" disabled={!wsUrl.trim()} onclick={startIterate}>
						Start Iterative
					</button>
				{/if}
			{/if}
		</div>
		{/if}
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
		{#if mode === 'iterate' && iterBestScore > 0}
		<div class="stat best-stat">
			<span class="stat-label">Best</span>
			<span class="stat-val" class:green={iterBestScore >= targetScore}
				class:gold={iterBestScore > 0 && iterBestScore < targetScore}>{iterBestScore}</span>
		</div>
		{/if}
		{#if targetScore}
		<div class="stat">
			<span class="stat-label">Target</span>
			<span class="stat-val red">{targetScore}</span>
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

	<!-- ── Iteration Tabs ──────────────────────────────────────────────────── -->
	{#if iterations.length > 0}
	<div class="iter-tabs">
		{#each iterations as it, i}
			<button
				class="iter-tab"
				class:active={activeIter === i}
				class:done={it.phase === 'done'}
				class:best={it.score === iterBestScore && it.score > 0}
				onclick={() => activeIter = i}
			>
				<span class="tab-num">#{i + 1}</span>
				{#if it.score > 0}
					<span class="tab-score" class:best={it.score === iterBestScore}>{it.score}</span>
				{:else}
					<span class="tab-dots">...</span>
				{/if}
			</button>
		{/each}
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
			<!-- Score Chart (inline) -->
			{#if scoreHistory.length > 1}
			<div class="sidebar-section chart-section">
				<h3>Score</h3>
				<svg viewBox="0 0 {CHART_W} {CHART_H}" class="chart-svg-sm" preserveAspectRatio="none">
					{#each [0.25, 0.5, 0.75] as frac}
						<line x1="0" y1={CHART_H * frac} x2={CHART_W} y2={CHART_H * frac}
							stroke="#30363d" stroke-width="1"/>
					{/each}
					{#if targetScore && chartMaxScore > 0}
						<line x1="0" y1={CHART_H - (targetScore / chartMaxScore) * CHART_H}
							x2={CHART_W} y2={CHART_H - (targetScore / chartMaxScore) * CHART_H}
							stroke="#f85149" stroke-width="1" stroke-dasharray="6 4" opacity="0.5"/>
					{/if}
					{#if chartPlanLine}
						<polyline points={chartPlanLine} fill="none" stroke="#56d364"
							stroke-width="1.5" stroke-dasharray="4 3" opacity="0.7"/>
					{/if}
					<polyline points={chartPolyline} fill="none" stroke="#39d353" stroke-width="2"/>
					{#if chartLastPt}
						<circle
							cx={(chartLastPt.round / maxRounds) * CHART_W}
							cy={CHART_H - (chartLastPt.score / chartMaxScore) * CHART_H}
							r="3" fill="#39d353"/>
					{/if}
				</svg>
				<div class="chart-legend">
					<span style="color:#39d353">— live</span>
					<span style="color:#56d364">-- plan</span>
					{#if targetScore}<span style="color:#f85149">-- {targetScore}</span>{/if}
				</div>
			</div>
			{/if}

			<!-- Orders -->
			<div class="sidebar-section">
				<h3>Orders ({orders.length})</h3>
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
				<h3>GPU Passes ({gpuPasses.length})</h3>
				{#each gpuPasses as p}
					<div class="gpu-pass" class:best={p.score === bestGpuScore}>
						<span class="pass-num">P{p.pass}</span>
						<span class="pass-states">{((p.max_states||0)/1000).toFixed(0)}K</span>
						<span class="pass-score" class:best={p.score === bestGpuScore}>{p.score}</span>
						<span class="pass-time muted">{p.elapsed}s</span>
					</div>
				{/each}
			</div>
			{/if}

			<!-- Bot inventory -->
			{#if botsTyped.length > 0}
			<div class="sidebar-section">
				<h3>Bots ({botsTyped.length})</h3>
				{#each botsTyped as bot}
					<div class="bot-row" class:selected={selectedBot === bot.id}
						onclick={() => selectedBot = selectedBot === bot.id ? null : bot.id}>
						<span class="bot-dot" style="background:{BOT_COLORS[bot.id % BOT_COLORS.length]}"></span>
						<span class="bot-pos">{bot.position[0]},{bot.position[1]}</span>
						<span class="bot-inv">{bot.inventory.length > 0 ? bot.inventory.join(', ') : '-'}</span>
					</div>
				{/each}
			</div>
			{/if}
		</div>
	</div>
	{/if}

	<!-- ── Plan Upgrades ────────────────────────────────────────────────────── -->
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

	<!-- ── Pipeline Done (single mode) ───────────────────────────────────────── -->
	{#if pipelineDone && mode === 'single'}
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

	<!-- ── Iteration History (iterate mode) ─────────────────────────────────── -->
	{#if iterations.length > 0}
	<div class="panel iter-panel">
		<h3>Iterations
			<span class="iter-meta">
				Best: <span class="green">{iterBestScore}</span>
				| {iterations.length} runs
				| {fmtSecs(timerElapsed)}
				{#if targetScore}| Target: <span class="red">{targetScore}</span>{/if}
			</span>
		</h3>
		<!-- Iteration bar chart -->
		<div class="iter-chart">
			<svg viewBox="0 0 {Math.max(400, iterations.length * 80)} 90" class="chart-svg" preserveAspectRatio="none">
				{#each iterations as it, i}
					{@const maxV = Math.max(targetScore || 200, iterBestScore, 1)}
					{@const h = Math.max(4, (it.score / maxV) * 75)}
					{@const w = Math.max(20, Math.min(60, 380 / Math.max(iterations.length, 1)))}
					{@const x = 10 + i * (w + 4)}
					<rect
						{x} y={80 - h} width={w} height={h}
						fill={it.score >= targetScore ? '#39d353' :
							  it.score === iterBestScore && it.score > 0 ? '#58a6ff' : '#d29922'}
						rx="3" opacity={activeIter === i ? 1 : 0.7}
					/>
					<text x={x + w/2} y={75 - h} fill="var(--text)" font-size="9"
						text-anchor="middle" font-weight="600">{it.score || '...'}</text>
					<text x={x + w/2} y={88} fill="var(--text-muted)" font-size="7"
						text-anchor="middle">#{i+1}</text>
				{/each}
				<!-- Target line -->
				{#if targetScore}
					{@const maxV = Math.max(targetScore || 200, iterBestScore, 1)}
					<line x1="0" y1={80 - (targetScore / maxV) * 75}
						x2={Math.max(400, iterations.length * 80)} y2={80 - (targetScore / maxV) * 75}
						stroke="#f85149" stroke-width="1" stroke-dasharray="4 3" opacity="0.5"/>
				{/if}
			</svg>
		</div>
		<!-- Iteration table -->
		<div class="iter-table">
			{#each iterations as it, i}
			<div class="iter-row" class:active={activeIter === i}
				onclick={() => activeIter = i}>
				<span class="iter-num">#{i+1}</span>
				<span class="iter-phase" class:done={it.phase === 'done'}
					class:playing={it.phase === 'playing' || it.phase === 'live_play'}
					class:optimizing={it.phase === 'optimize_replay'}>
					{it.phase === 'live_play' || it.phase === 'playing' ? '🎮 live' :
					 it.phase === 'optimize_replay' ? '🔥 opt+replay' :
					 it.phase === 'done' ? '✓' : it.phase}
				</span>
				<span class="iter-score" class:best={it.score === iterBestScore && it.score > 0}
					class:hit={it.score >= targetScore}>{it.score || '...'}</span>
				{#if it.optScore > 0}
					<span class="iter-opt muted">opt:{it.optScore}</span>
				{/if}
				{#if it.gameScore > 0}
					<span class="iter-game muted">game:{it.gameScore}</span>
				{/if}
				<span class="iter-time muted">{it.elapsed ? it.elapsed.toFixed(0) + 's' : '...'}</span>
				{#if it.capturedOrders}<span class="iter-captured">+{it.capturedOrders} ord</span>{/if}
			</div>
			{/each}
		</div>
	</div>
	{/if}

	<!-- ── Done summary (iterate mode) ────────────────────────────────────── -->
	{#if phase === 'done' && mode === 'iterate' && iterations.length > 0}
	<div class="panel done-panel">
		<div class="done-score">
			<span class="done-label">Best Score</span>
			<span class="done-val" class:hit={iterBestScore >= targetScore}>{iterBestScore}</span>
			{#if detectedDiff}
				<span class="badge" style="background:{diffColors[detectedDiff]||'#666'}">{detectedDiff}</span>
			{/if}
			{#if iterBestScore >= targetScore}
				<span class="hit-badge">TARGET HIT</span>
			{/if}
		</div>
		<div class="done-meta">
			<span>{iterations.length} iterations in {fmtSecs(timerElapsed)}</span>
			<span>|</span>
			<span>Target: {targetScore}</span>
			<span>|</span>
			<span class:green={iterBestScore >= targetScore}
				class:red={iterBestScore < targetScore}>
				{iterBestScore >= targetScore ? 'PASSED' : `${targetScore - iterBestScore} short`}
			</span>
		</div>
		</div>
	{/if}

	<!-- ── Log ───────────────────────────────────────────────────────────────── -->
	<div class="panel log-panel">
		<h3>Log <span class="muted" style="font-weight:400; font-size:0.75rem">({logs.length})</span></h3>
		<div class="log-box">
			{#each logs.slice().reverse() as entry}
				<div class="log-line" class:error={entry.text?.startsWith('ERROR')}>
					<span class="log-t">{entry.t}</span>
					{#if entry.iter !== null && entry.iter !== undefined}
						<span class="log-iter">#{entry.iter + 1}</span>
					{/if}
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
	.header-right {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	h1 { font-size: 1.5rem; margin: 0 0 0.2rem; font-family: var(--font-mono); font-weight: 700; }
	.sub { color: var(--text-muted); font-size: 0.82rem; margin: 0; }

	.diff-badge {
		padding: 0.2rem 0.6rem;
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 700;
		color: #000;
		text-transform: uppercase;
	}
	.target-label {
		font-size: 0.75rem;
		color: var(--text-muted);
		font-weight: 600;
	}

	/* Phase badge */
	.phase-badge {
		padding: 0.3rem 0.8rem;
		border-radius: 20px;
		font-size: 0.8rem;
		font-weight: 600;
		background: var(--bg-card);
		border: 1px solid var(--border);
		color: var(--text-muted);
		backdrop-filter: blur(4px);
		box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
	}
	.phase-badge.playing    { border-color: #39d353; color: #39d353; }
	.phase-badge.post       { border-color: #56d364; color: #56d364; }
	.phase-badge.done       { border-color: #39d353; color: #39d353; }
	.phase-badge.replaying  { border-color: #58a6ff; color: #58a6ff; }

	/* Timer bar */
	.timer-bar {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 0.5rem 0.75rem;
	}
	.timer-track {
		height: 6px;
		background: var(--bg);
		border-radius: 3px;
		overflow: hidden;
		margin-bottom: 0.3rem;
	}
	.timer-fill {
		height: 100%;
		background: linear-gradient(90deg, #39d353, #56d364);
		transition: width 0.3s;
		border-radius: 3px;
	}
	.timer-fill.warn { background: linear-gradient(90deg, #d29922, #facc15); }
	.timer-fill.danger { background: linear-gradient(90deg, #f85149, #da3633); }
	.timer-labels {
		display: flex;
		justify-content: space-between;
		font-size: 0.72rem;
		color: var(--text-muted);
	}
	.timer-remaining { font-weight: 600; }
	.timer-remaining.warn { color: #d29922; }
	.timer-remaining.danger { color: #f85149; animation: pulse 1s ease-in-out infinite; }

	/* Panel */
	.panel {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 1rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}
	.input-panel.collapsed {
		padding: 0.4rem 0.75rem;
	}
	.expand-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		font-size: 0.75rem;
		cursor: pointer;
		font-family: inherit;
		padding: 0;
	}
	.expand-btn:hover { color: var(--accent); }

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
	.url-input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(57, 211, 83, 0.1); }
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
	.num-input.sm { width: 45px; }
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		font-size: 0.72rem;
		font-weight: 600;
		color: #000;
		white-space: nowrap;
	}

	/* Mode toggle */
	.mode-toggle {
		display: flex;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		overflow: hidden;
	}
	.mode-btn {
		padding: 0.3rem 0.7rem;
		border: none;
		background: transparent;
		color: var(--text-muted);
		font-size: 0.78rem;
		font-weight: 600;
		cursor: pointer;
		font-family: inherit;
		transition: all 0.15s;
	}
	.mode-btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.mode-btn.active {
		background: var(--accent);
		color: #0d1117;
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
		letter-spacing: 0.02em;
	}
	.btn:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-start   { background: #39d353; color: #0d1117; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); }
	.btn-iterate { background: #58a6ff; color: #0d1117; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); }
	.btn-stop    { background: var(--red); color: #fff; }
	.btn-replay  { background: #58a6ff; color: #0d1117; margin-top: 0.5rem; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3); }
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
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
		transition: all 0.15s ease;
	}
	.best-stat { border-color: #39d353; }
	.stat-label {
		font-size: 0.65rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.stat-val { font-size: 1.1rem; font-weight: 700; font-family: var(--font-mono); }
	.stat-src { font-size: 0.78rem; font-weight: 600; }
	.green  { color: #39d353; }
	.accent { color: var(--accent-light); }
	.gold   { color: #d29922; }
	.red    { color: #f85149; }

	/* Progress */
	.progress-bar {
		height: 6px;
		background: var(--bg-card);
		border-radius: 3px;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--green), var(--accent));
		transition: width 0.2s;
	}
	.progress-fill.accent { background: var(--accent-light); }
	.progress-txt {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-align: center;
		margin-top: 0.2rem;
	}

	/* Iteration tabs */
	.iter-tabs {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}
	.iter-tab {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.35rem 0.65rem;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 6px 6px 0 0;
		cursor: pointer;
		font-family: inherit;
		font-size: 0.78rem;
		color: var(--text-muted);
		transition: all 0.15s;
	}
	.iter-tab.active {
		background: var(--bg);
		border-bottom-color: var(--bg);
		color: var(--text);
	}
	.iter-tab.best {
		border-color: #39d353;
	}
	.tab-num { font-weight: 600; }
	.tab-score { font-weight: 700; font-family: var(--font-mono); }
	.tab-score.best { color: #39d353; }
	.tab-dots { color: var(--text-muted); animation: pulse 1s ease-in-out infinite; }

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
		min-width: 220px;
		max-width: 340px;
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
	.order.active  { border-color: #facc15; }
	.order.preview { border-color: #f472b6; opacity: 0.8; }
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
	.oitem.del { background: #39d353; color: #0d1117; }

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
	.gpu-pass.best { background: rgba(57,211,83,0.1); border: 1px solid #39d353; }
	.pass-num   { font-weight: 700; color: var(--text-muted); min-width: 20px; }
	.pass-states { color: var(--text-muted); font-size: 0.72rem; min-width: 30px; }
	.pass-score { font-weight: 700; }
	.pass-score.best { color: #39d353; }
	.pass-time  { margin-left: auto; }

	/* Bot rows */
	.bot-row {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.2rem 0.35rem;
		font-size: 0.75rem;
		cursor: pointer;
		border-radius: 3px;
		transition: background 0.1s;
	}
	.bot-row:hover { background: rgba(255,255,255,0.04); }
	.bot-row.selected { background: rgba(57,211,83,0.1); }
	.bot-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
	.bot-pos { color: var(--text-muted); font-family: var(--font-mono); min-width: 40px; }
	.bot-inv { color: var(--text); }

	/* Score chart */
	.chart-panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; color: var(--text-muted); }
	.chart-svg {
		width: 100%;
		height: 120px;
		background: var(--bg);
		border-radius: 4px;
		display: block;
	}
	.chart-svg-sm {
		width: 100%;
		height: 80px;
		background: var(--bg);
		border-radius: 4px;
		display: block;
	}
	.chart-section { margin-bottom: 0.25rem; }
	.chart-legend {
		font-size: 0.72rem;
		margin-top: 0.3rem;
		color: var(--text-muted);
		display: flex;
		gap: 1rem;
	}

	/* Timeline */
	.panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; color: var(--text-muted); }
	.timeline {
		display: flex;
		flex-direction: row;
		flex-wrap: wrap;
		gap: 0.4rem;
		align-items: center;
	}
	.tl-item { display: flex; align-items: center; gap: 0.3rem; }
	.tl-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
	.tl-content { display: flex; gap: 0.3rem; align-items: baseline; flex-wrap: wrap; }
	.tl-src  { font-size: 0.78rem; font-weight: 600; }
	.tl-score { font-size: 0.72rem; color: #39d353; }
	.tl-round { font-size: 0.68rem; }
	.tl-line { width: 16px; height: 1px; background: var(--border); }

	/* Post-optimize */
	.opt-panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; }
	.opt-row { display: flex; gap: 1rem; flex-wrap: wrap; }
	.opt-stat { display: flex; flex-direction: column; align-items: center; min-width: 80px; }
	.opt-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }
	.opt-val   { font-size: 1.2rem; font-weight: 700; font-family: var(--font-mono); }

	/* Done panel */
	.done-panel { border-color: #39d353; }
	.done-score {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.75rem;
	}
	.done-label { color: var(--text-muted); font-size: 0.85rem; }
	.done-val   { font-size: 2.5rem; font-weight: 800; color: #39d353; font-family: var(--font-mono); }
	.done-val.hit { color: #39d353; text-shadow: 0 0 20px rgba(57,211,83,0.3); }
	.done-meta {
		display: flex;
		gap: 0.5rem;
		font-size: 0.8rem;
		color: var(--text-muted);
		margin-bottom: 0.75rem;
	}
	.hit-badge {
		background: #39d353;
		color: #0d1117;
		padding: 0.2rem 0.6rem;
		border-radius: 4px;
		font-size: 0.72rem;
		font-weight: 700;
		animation: pulse 2s ease-in-out infinite;
	}

	/* Iteration panel */
	.iter-panel h3 { margin-bottom: 0.5rem; }
	.iter-meta {
		font-weight: 400;
		font-size: 0.75rem;
		color: var(--text-muted);
	}
	.iter-chart { margin-bottom: 0.5rem; }
	.iter-table { display: flex; flex-direction: column; gap: 2px; max-height: 250px; overflow-y: auto; }
	.iter-row {
		display: flex; align-items: center; gap: 0.5rem;
		padding: 0.3rem 0.5rem; font-size: 0.8rem;
		background: rgba(255,255,255,0.02); border-radius: 3px;
		cursor: pointer;
		transition: background 0.1s;
	}
	.iter-row:hover { background: rgba(255,255,255,0.05); }
	.iter-row.active { background: rgba(57,211,83,0.08); border-left: 2px solid #39d353; }
	.iter-num { font-family: var(--font-mono); color: var(--text-muted); min-width: 2rem; }
	.iter-phase {
		font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
		padding: 0.1rem 0.4rem; border-radius: 3px; min-width: 3.5rem; text-align: center;
	}
	.iter-phase.playing { background: #58a6ff20; color: #58a6ff; }
	.iter-phase.optimizing { background: #f8514920; color: #f85149; }
	.iter-phase.done { background: #39d35320; color: #39d353; }
	.iter-score { font-family: var(--font-mono); font-weight: 600; min-width: 3rem; }
	.iter-score.best { color: #39d353; }
	.iter-score.hit { color: #39d353; font-weight: 800; }
	.iter-game { font-size: 0.72rem; }
	.iter-opt { font-size: 0.72rem; }
	.iter-time { min-width: 3rem; }
	.iter-orders { font-size: 0.72rem; }
	.iter-captured { font-size: 0.72rem; color: #39d353; font-weight: 600; }

	/* Log */
	.log-panel h3 { font-size: 0.82rem; font-weight: 600; margin: 0 0 0.5rem; }
	.log-box {
		background: var(--bg);
		border-radius: 4px;
		padding: 0.5rem;
		max-height: 250px;
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
	.log-line.error { color: #f85149; }
	.log-t { color: var(--text-muted); white-space: nowrap; flex-shrink: 0; }
	.log-iter {
		color: #58a6ff;
		font-weight: 600;
		font-size: 0.68rem;
		white-space: nowrap;
		flex-shrink: 0;
	}
	.muted { color: var(--text-muted); }

	.replay-section { margin-top: 0.5rem; }

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50%       { opacity: 0.4; }
	}

	@media (max-width: 900px) {
		.game-area { flex-direction: column; }
		.sidebar { max-width: 100%; }
	}
</style>
