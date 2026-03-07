<script>
	import Grid from '$lib/components/Grid.svelte';

	// Adaptive cell size — bigger for small grids, smaller for large ones
	let adaptiveCell = $derived(
		gameInit ? Math.max(18, Math.min(32, Math.floor(560 / Math.max(gameInit.width, gameInit.height)))) : 24
	);
	const CELL = 24; // fallback
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
	const TARGETS = { easy: 142, medium: 200, hard: 250, expert: 305, nightmare: 350 };
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
	let gpuOptTime     = $state(60);   // seconds per offline GPU optimize pass

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
	let greedyMode     = $state(false);  // true when DP plan exhausted, bot thinking for itself

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

	// ── Per-iteration snapshots (for tab switching) ──────────────────────────
	let iterSnapshots  = $state({});  // { [iter]: { scoreHistory, gpuPasses, planUpgrades, finalScore, bots, orders } }

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

	// ── Multi-key continuation ───────────────────────────────────────────────────
	let waitingForKey  = $state(false);
	let keyCount       = $state(0);
	let cumulativeBest = $state(0);
	let totalOrders    = $state(0);

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
		detectedDiff = null; iterSnapshots = {};
		waitingForKey = false; keyCount = 0; cumulativeBest = 0; totalOrders = 0;
	}

	function resetForContinuation() {
		// Keep cumulative state and grid, reset per-key state
		waitingForKey = false;
		keyCount++;
		currentRound = 0; liveScore = 0;
		planScore = 0; planSource = 'none';
		finalScore = null; scoreHistory = []; gpuPasses = [];
		planUpgrades = []; postOptStart = null; postOptProgress = null;
		postOptFinal = null; pipelineDone = null;
		iterations = []; activeIter = 0; currentIterData = null;
		iterSnapshots = {};
	}

	function snapshotIter(iter) {
		// Save current iteration's display state for tab recall
		iterSnapshots = { ...iterSnapshots, [iter]: {
			scoreHistory: [...scoreHistory],
			gpuPasses: [...gpuPasses],
			planUpgrades: [...planUpgrades],
			finalScore,
			bots: [...bots],
			orders: [...orders],
			liveScore,
			planScore,
			planSource,
			currentRound,
		}};
	}

	function restoreIterSnapshot(iter) {
		const snap = iterSnapshots[iter];
		if (!snap) return;
		scoreHistory = snap.scoreHistory;
		gpuPasses = snap.gpuPasses;
		planUpgrades = snap.planUpgrades;
		finalScore = snap.finalScore;
		bots = snap.bots;
		orders = snap.orders;
		liveScore = snap.liveScore;
		planScore = snap.planScore;
		planSource = snap.planSource;
		currentRound = snap.currentRound;
	}

	function resetIterState() {
		// Reset per-iteration state but keep iterations array and grid
		// Don't clear gameInit — keep showing last game grid during optimize phase
		currentRound = 0; liveScore = 0;
		planScore = 0; planSource = 'none';
		finalScore = null; scoreHistory = []; gpuPasses = [];
		planUpgrades = []; postOptStart = null; postOptProgress = null;
		postOptFinal = null; pipelineDone = null; greedyMode = false;
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
				// Snapshot previous iteration's data before resetting
				if (iterations.length > 0) snapshotIter(activeIter);
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
				if (event.phase === 'optimize_replay' || event.phase === 'deep_training' || event.phase === 'initial_optimize') {
					phase = 'optimizing';
				} else if (event.phase === 'replay_discover') {
					phase = 'replaying';
				} else {
					phase = 'playing';
				}
				addLog(`--- Iter ${event.iter + 1} (${event.phase || 'live'}) ${Math.floor(event.remaining)}s left ---`, event.iter);
				break;

			case 'optimize_phase_start':
				phase = 'optimizing';
				addLog(`GPU optimize: ${event.difficulty}, budget=${event.max_time}s`, iter);
				break;

			case 'optimize_start':
				gpuPasses = [];  // Reset for this optimize phase
				addLog(`Optimizing: ${event.orders} orders, prev=${event.prev_score}`, iter);
				break;

			case 'gpu_bot_done':
				gpuPasses = [...gpuPasses, {
					pass: `B${event.bot}`, score: event.score,
					elapsed: event.elapsed, max_states: 0,
				}];
				addLog(`Bot ${event.bot}/${event.num_bots}: score=${event.score} (${event.elapsed}s)`, iter);
				break;

			case 'gpu_phase':
				gpuPasses = [...gpuPasses, {
					pass: event.phase, score: event.score,
					elapsed: event.elapsed, max_states: 0,
				}];
				addLog(`${event.phase} iter ${event.iteration}: score=${event.score}`, iter);
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
						ordersTotal: event.orders_total || 0,
						isDeep: event.is_deep || false,
						elapsed: event.elapsed - (updated[idx].startTime || 0),
						difficulty: event.difficulty,
					};
					iterations = updated;
				}
				if ((event.score || 0) > cumulativeBest) cumulativeBest = event.score;
				if (event.orders_total) totalOrders = event.orders_total;
				addLog(`Iter ${event.iter + 1}: score=${event.score}${event.is_deep ? ' [DEEP]' : ''} (${Math.floor(event.remaining)}s left)`, event.iter);
				snapshotIter(event.iter);
				break;
			}

			case 'iter_skip':
				addLog(`Iter ${event.iter + 1} skipped: ${event.reason}`, event.iter);
				break;

			case 'iter_summary':
				if (event.best_score > cumulativeBest) cumulativeBest = event.best_score;
				if (event.orders) totalOrders = event.orders;
				addLog(`Best: ${event.best_score} after ${event.iterations_done} iters (${event.orders || '?'} orders)`, event.iter);
				break;

			case 'orders_update':
				totalOrders = event.count || totalOrders;
				addLog(`Orders: ${event.count} total (+${event.new_orders || 0} new)`, iter);
				break;

			case 'mode_change':
				addLog(`Mode: ${event.mode} (${event.reason})`, iter);
				if (event.mode === 'deep') phase = 'optimizing';
				break;

			case 'pipeline_complete':
				snapshotIter(activeIter);
				if (event.best_score > cumulativeBest) cumulativeBest = event.best_score;
				if (event.orders) totalOrders = event.orders;
				// Don't set phase=done yet — wait for need_new_key
				running = false;
				stopTimer();
				addLog(`Complete! Best=${event.best_score} in ${event.iterations} iters, ${Math.floor(event.total_elapsed)}s (${event.orders || '?'} orders)`);
				break;

			case 'need_new_key':
				waitingForKey = true;
				phase = 'waiting_for_key';
				if (event.best_score > cumulativeBest) cumulativeBest = event.best_score;
				if (event.orders) totalOrders = event.orders;
				addLog(`Token depleted. Paste new key to continue. Best: ${cumulativeBest}, Orders: ${totalOrders}`);
				break;

			case 'seed_cracked':
				addLog(`Seed cracked: ${event.seed}`, iter);
				break;

			case 'log':
				if (event.text?.includes('GREEDY mode') || event.text?.includes('greedy')) {
					greedyMode = true;
				}
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
					try { handleEvent(JSON.parse(line.slice(6))); } catch (e) { // Stream closed — client disconnected
					}
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
		if (!waitingForKey) {
			resetAll();
		} else {
			resetForContinuation();
		}
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
					try { handleEvent(JSON.parse(line.slice(6))); } catch (e) { // Stream closed — client disconnected
					}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') addLog(`Connection error: ${e.message}`);
		}
		if (phase !== 'done' && phase !== 'waiting_for_key') phase = 'done';
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
					try { handleEvent(JSON.parse(line.slice(6))); } catch (e) { // Stream closed — client disconnected
					}
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
		waitingForKey = false;
		stopTimer();
		phase = 'done';
		addLog('Stopped.');
	}

	function fmtSecs(s) {
		if (!s && s !== 0) return '--';
		const m = Math.floor(s / 60);
		const ss = Math.floor(s % 60);
		return `${m}:${ss.toString().padStart(2, '0')}`;
	}

	let diffColors = {
		easy: '#39d353', medium: '#d29922', hard: '#f85149', expert: '#da3633', nightmare: '#a855f7',
	};
</script>

<div class="page stagger">
	<div class="header">
		<div>
			<h1>GPU Pipeline {#if keyCount > 0}<span style="font-size:0.6em;color:#58a6ff">Key #{keyCount + 1}</span>{/if}</h1>
			<p class="sub">Replay -> Discover orders -> Optimize -> Repeat {#if totalOrders > 0}<span style="color:#bc8cff">| {totalOrders} orders</span>{/if}</p>
		</div>
		<div class="header-right">
			{#if detectedDiff}
				<span class="diff-badge" style="background:{diffColors[detectedDiff]||'#666'}">{detectedDiff}</span>
				<span class="target-label">Target: {targetScore}</span>
			{/if}
			<div class="phase-badge" class:playing={phase === 'playing'}
				class:post={phase === 'post_optimizing' || phase === 'optimizing'}
				class:done={phase === 'done'}
				class:replaying={phase === 'replaying'}
				class:waiting={phase === 'waiting_for_key'}>
				{#if phase === 'idle'}Idle
				{:else if phase === 'playing'}Playing
				{:else if phase === 'optimizing'}GPU Optimizing
				{:else if phase === 'post_optimizing'}Post-Optimizing
				{:else if phase === 'replaying' && greedyMode}Thinking
				{:else if phase === 'replaying'}Replaying
				{:else if phase === 'waiting_for_key'}Waiting for Key
				{:else}Done
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
					{phase === 'playing' ? 'Playing...' :
					 phase === 'optimizing' ? 'GPU optimizing...' :
					 phase === 'post_optimizing' ? 'Post-optimizing...' :
					 phase === 'replaying' && greedyMode ? 'Thinking...' :
					 phase === 'replaying' ? 'Replaying...' : 'Running...'}
					{#if mode === 'iterate' && iterations.length > 0}
						(iter {activeIter + 1}, {fmtSecs(timerRemaining)} left)
					{/if}
				</span>
			{:else if waitingForKey}
				<div class="new-key-prompt">
					<span class="new-key-label">Token depleted — paste new key to continue ({totalOrders} orders, best: {cumulativeBest})</span>
					<button class="btn btn-iterate" disabled={!wsUrl.trim()} onclick={startIterate}>
						Continue
					</button>
				</div>
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
			<span class="stat-src" style="color:{sourceColor(planSource)}">
				{#if greedyMode}🧠 {/if}{planSource}
			</span>
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
		{#if mode === 'iterate' && (iterBestScore > 0 || cumulativeBest > 0)}
		<div class="stat best-stat">
			<span class="stat-label">Best</span>
			<span class="stat-val" class:green={Math.max(iterBestScore, cumulativeBest) >= targetScore}
				class:gold={Math.max(iterBestScore, cumulativeBest) > 0 && Math.max(iterBestScore, cumulativeBest) < targetScore}>{Math.max(iterBestScore, cumulativeBest)}</span>
		</div>
		{/if}
		{#if totalOrders > 0}
		<div class="stat">
			<span class="stat-label">Orders</span>
			<span class="stat-val" style="color:#bc8cff">{totalOrders}</span>
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
		<span class="iter-tabs-label">Iter:</span>
		{#each iterations as it, i}
			<button
				class="iter-tab"
				class:active={activeIter === i}
				class:done={it.phase === 'done'}
				class:best={it.score === iterBestScore && it.score > 0}
				onclick={() => { if (it.phase === 'done') restoreIterSnapshot(i); activeIter = i; }}
			>
				<span class="tab-num">#{i + 1}</span>
				{#if it.score > 0}
					<span class="tab-score" class:best={it.score === iterBestScore}>{it.score}pts</span>
				{:else}
					<span class="tab-dots">...</span>
				{/if}
			</button>
		{/each}
	</div>
	{/if}

	<!-- ── Row 1: Logs | GPU Progress | Score Chart ────────────────────────── -->
	{#if gameInit || phase !== 'idle'}
	<div class="grid-row">
		<!-- Logs -->
		<div class="grid-cell">
			<div class="sidebar-section log-inline">
				<h3>Log <span class="log-count">({logs.length})</span></h3>
				<div class="log-box">
					{#each logs.slice().reverse() as entry}
						<div class="log-line"
							class:error={entry.text?.startsWith('ERROR')}
							class:upgrade={entry.text?.includes('Plan:') || entry.text?.includes('->')}
							class:desync={entry.text?.toLowerCase().includes('desync')}
							class:gpu-log={entry.text?.includes('GPU') || entry.text?.includes('gpu')}
							class:score-log={entry.text?.includes('score=') || entry.text?.includes('Final=')}
							class:iter-start={entry.text?.startsWith('---')}
							class:seed-log={entry.text?.includes('Seed cracked') || entry.text?.includes('seed')}
							class:warn-log={entry.text?.includes('skip') || entry.text?.includes('timeout') || entry.text?.toLowerCase().includes('warn')}
							class:greedy-log={entry.text?.includes('GREEDY') || entry.text?.includes('greedy')}>
							<span class="log-t">{entry.t}</span>
							{#if entry.iter !== null && entry.iter !== undefined}
								<span class="log-iter">#{entry.iter + 1}</span>
							{/if}
							<span>{entry.text}</span>
						</div>
					{/each}
					{#if logs.length === 0}
						<div class="muted">Awaiting signal...</div>
					{/if}
				</div>
			</div>
		</div>

		<!-- GPU Progress -->
		<div class="grid-cell">
			<div class="sidebar-section">
				{#if gpuPasses.length > 0}
				<h3>GPU Progress ({gpuPasses.length})</h3>
				{#each gpuPasses as p}
					<div class="gpu-pass" class:best={p.score === bestGpuScore}>
						<span class="pass-num">{typeof p.pass === 'number' ? `P${p.pass}` : p.pass}</span>
						{#if p.max_states > 0}
							<span class="pass-states">{((p.max_states||0)/1000).toFixed(0)}K</span>
						{/if}
						<span class="pass-score" class:best={p.score === bestGpuScore}>{p.score}</span>
						<span class="pass-time muted">{p.elapsed}s</span>
					</div>
				{/each}
				{:else}
				<h3>GPU Progress</h3>
				<div class="muted">Waiting for GPU...</div>
				{/if}
			</div>
		</div>

		<!-- Score Chart -->
		<div class="grid-cell">
			<div class="sidebar-section chart-section">
				<h3>Score</h3>
				{#if scoreHistory.length > 1}
				<svg viewBox="0 0 {CHART_W} {CHART_H}" class="chart-svg-col" preserveAspectRatio="none">
					{#each [0.25, 0.5, 0.75] as frac}
						<line x1="0" y1={CHART_H * frac} x2={CHART_W} y2={CHART_H * frac}
							stroke="#1a1f28" stroke-width="1"/>
					{/each}
					{#if targetScore && chartMaxScore > 0}
						<line x1="0" y1={CHART_H - (targetScore / chartMaxScore) * CHART_H}
							x2={CHART_W} y2={CHART_H - (targetScore / chartMaxScore) * CHART_H}
							stroke="#f85149" stroke-width="1" stroke-dasharray="5 3" opacity="0.5"/>
					{/if}
					{#if chartPlanLine}
						<polyline points={chartPlanLine} fill="none" stroke="#39d353"
							stroke-width="1.5" stroke-dasharray="4 3" opacity="0.6"/>
					{/if}
					<polyline points={chartPolyline} fill="none" stroke="#56d364" stroke-width="2"/>
					{#if chartLastPt}
						<circle
							cx={(chartLastPt.round / maxRounds) * CHART_W}
							cy={CHART_H - (chartLastPt.score / chartMaxScore) * CHART_H}
							r="3" fill="#56d364"/>
					{/if}
				</svg>
				<div class="chart-legend">
					<span style="color:#56d364">/// live</span>
					<span style="color:#39d353">--- plan</span>
					{#if targetScore}<span style="color:#f85149">--- {targetScore}</span>{/if}
				</div>
				{:else}
				<div class="muted chart-empty">Waiting for data...</div>
				{/if}
			</div>
		</div>
	</div>

	<!-- ── Row 2: Grid | Orders | Bots ─────────────────────────────────────── -->
	<div class="grid-row">
		<!-- Grid -->
		<div class="grid-cell">
			{#if gameInit}
			<div class="grid-wrap">
				<Grid
					width={gameInit.width}
					height={gameInit.height}
					cellSize={adaptiveCell}
					{wallSet}
					{shelfSet}
					{itemMap}
					dropOff={gameInit.drop_off}
					dropOffZones={gameInit.drop_off_zones}
					spawn={gameInit.spawn}
					bots={botsTyped}
					{botPositions}
					botColors={BOT_COLORS}
					{selectedBot}
					onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
				/>
			</div>
			{:else}
			<div class="muted" style="padding:1rem">Waiting for game...</div>
			{/if}
		</div>

		<!-- Orders -->
		<div class="grid-cell">
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
		</div>

		<!-- Bots -->
		<div class="grid-cell">
			<div class="sidebar-section">
				{#if botsTyped.length > 0}
				<h3>Bots ({botsTyped.length})</h3>
				{#each botsTyped as bot}
					<div class="bot-row" class:selected={selectedBot === bot.id}
						onclick={() => selectedBot = selectedBot === bot.id ? null : bot.id}>
						<span class="bot-dot" style="background:{BOT_COLORS[bot.id % BOT_COLORS.length]}"></span>
						<span class="bot-pos">{bot.position[0]},{bot.position[1]}</span>
						<span class="bot-inv">{bot.inventory.length > 0 ? bot.inventory.join(', ') : '-'}</span>
					</div>
				{/each}
				{:else}
				<h3>Bots</h3>
				<div class="muted">Waiting...</div>
				{/if}
			</div>
		</div>
	</div>

	<!-- ── Iterations (full width, shown when available) ───────────────────── -->
	{#if iterations.length > 0}
	{@const IC_W = 360}
	{@const IC_H = 100}
	{@const IC_PAD = { top: 14, right: 8, bottom: 16, left: 32 }}
	{@const icMaxV = Math.max(targetScore || 200, iterBestScore, 1)}
	{@const icMinV = Math.max(0, Math.min(...iterations.filter(it => it.score > 0).map(it => it.score), icMaxV) - 10)}
	{@const icRange = Math.max(icMaxV - icMinV, 1)}
	{@const icPlotW = IC_W - IC_PAD.left - IC_PAD.right}
	{@const icPlotH = IC_H - IC_PAD.top - IC_PAD.bottom}
	<div class="sidebar-section iter-sidebar">
		<h3>Iterations
			<span class="iter-meta">
				Best: <span class="cyan">{iterBestScore}</span>
				&middot; {iterations.length} runs
			</span>
		</h3>
		<div class="iter-chart-col">
			<svg viewBox="0 0 {IC_W} {IC_H}" class="iter-chart-svg" preserveAspectRatio="xMidYMid meet">
				<!-- Y-axis gridlines + labels -->
				{#each [0, 0.25, 0.5, 0.75, 1] as frac}
					{@const val = Math.round(icMinV + icRange * (1 - frac))}
					{@const y = IC_PAD.top + frac * icPlotH}
					<line x1={IC_PAD.left} y1={y} x2={IC_W - IC_PAD.right} y2={y}
						stroke="#1a1f28" stroke-width="0.5"/>
					<text x={IC_PAD.left - 4} y={y + 3} fill="#3a3f47" font-size="7"
						text-anchor="end" font-family="var(--font-mono)">{val}</text>
				{/each}

				<!-- Target line -->
				{#if targetScore && targetScore >= icMinV && targetScore <= icMaxV}
					{@const tY = IC_PAD.top + ((icMaxV - targetScore) / icRange) * icPlotH}
					<line x1={IC_PAD.left} y1={tY} x2={IC_W - IC_PAD.right} y2={tY}
						stroke="#f85149" stroke-width="0.8" stroke-dasharray="4 2" opacity="0.5"/>
					<text x={IC_W - IC_PAD.right + 2} y={tY + 3} fill="#f85149" font-size="6"
						font-family="var(--font-mono)" opacity="0.7">tgt</text>
				{/if}

				<!-- Area fill under line -->
				{#if iterations.filter(it => it.score > 0).length >= 2}
					{@const pts = iterations.map((it, i) => {
						const x = IC_PAD.left + (i / Math.max(iterations.length - 1, 1)) * icPlotW;
						const y = it.score > 0
							? IC_PAD.top + ((icMaxV - it.score) / icRange) * icPlotH
							: IC_PAD.top + icPlotH;
						return `${x},${y}`;
					})}
					<polygon
						points="{IC_PAD.left},{IC_PAD.top + icPlotH} {pts.join(' ')} {IC_PAD.left + ((iterations.length - 1) / Math.max(iterations.length - 1, 1)) * icPlotW},{IC_PAD.top + icPlotH}"
						fill="url(#iterGrad)" opacity="0.3"/>
					<polyline points={pts.join(' ')}
						fill="none" stroke="#39d353" stroke-width="1.5" stroke-linejoin="round"/>
				{/if}

				<!-- Data points -->
				{#each iterations as it, i}
					{@const x = IC_PAD.left + (i / Math.max(iterations.length - 1, 1)) * icPlotW}
					{@const y = it.score > 0
						? IC_PAD.top + ((icMaxV - it.score) / icRange) * icPlotH
						: IC_PAD.top + icPlotH}
					<circle cx={x} cy={y} r={activeIter === i ? 4 : 2.5}
						fill={it.score >= targetScore ? '#56d364' :
							  it.score === iterBestScore && it.score > 0 ? '#39d353' : '#d29922'}
						stroke={activeIter === i ? '#fff' : 'none'} stroke-width="1"
						style="cursor:pointer" opacity={it.score > 0 ? 1 : 0.3}
						onclick={() => { if (it.phase === 'done') restoreIterSnapshot(i); activeIter = i; }}/>
					<!-- Score label on best point -->
					{#if it.score === iterBestScore && it.score > 0}
						<text x={x} y={y - 6} fill="#39d353" font-size="8" font-weight="700"
							text-anchor="middle" font-family="var(--font-mono)">{it.score}</text>
					{/if}
				{/each}

				<!-- X-axis labels -->
				{#each iterations as _, i}
					{#if iterations.length <= 6 || i === 0 || i === iterations.length - 1 || (i + 1) % Math.ceil(iterations.length / 5) === 0}
						{@const x = IC_PAD.left + (i / Math.max(iterations.length - 1, 1)) * icPlotW}
						<text x={x} y={IC_H - 2} fill="#3a3f47" font-size="6.5"
							text-anchor="middle" font-family="var(--font-mono)">{i + 1}</text>
					{/if}
				{/each}

				<!-- Gradient def -->
				<defs>
					<linearGradient id="iterGrad" x1="0" y1="0" x2="0" y2="1">
						<stop offset="0%" stop-color="#39d353" stop-opacity="0.4"/>
						<stop offset="100%" stop-color="#39d353" stop-opacity="0.02"/>
					</linearGradient>
				</defs>
			</svg>
		</div>
		<div class="iter-list-col">
			{#each iterations as it, i}
			<div class="iter-row-sm" class:active={activeIter === i}
				class:best-row={it.score === iterBestScore && it.score > 0}
				onclick={() => { if (it.phase === 'done') restoreIterSnapshot(i); activeIter = i; }}>
				<span class="ir-num">{i+1}</span>
				<span class="ir-score" class:best={it.score === iterBestScore && it.score > 0}
					class:hit={it.score >= targetScore}>{it.score || '—'}</span>
				{#if it.optScore > 0}<span class="ir-detail">opt {it.optScore}</span>{/if}
				<span class="ir-time">{it.elapsed ? it.elapsed.toFixed(0) + 's' : ''}</span>
				{#if it.capturedOrders}<span class="ir-cap">+{it.capturedOrders}</span>{/if}
			</div>
			{/each}
		</div>
	</div>
	{/if}
	{/if}

	<!-- ── Source Transitions (inline) ──────────────────────────────────────── -->
	{#if planUpgrades.length > 0}
	<div class="panel transitions-panel">
		<h3>Source Transitions</h3>
		<div class="transition-flow">
			{#each planUpgrades as u, i}
				<div class="tr-chip">
					<span class="tr-round">R{u.round}</span>
					<span class="tr-arrow-in" style="color:{sourceColor(u.from || 'none')}">&#x25C0;</span>
					<span class="tr-to" style="color:{sourceColor(u.to)}; text-shadow: 0 0 8px {sourceColor(u.to)}40">{u.to}</span>
					{#if u.score}<span class="tr-score">{u.score}</span>{/if}
				</div>
				{#if i < planUpgrades.length - 1}
					<span class="tr-sep">&#x25B8;</span>
				{/if}
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

	<!-- Log is now inside the three-col layout above -->
</div>

<style>
	/* ═══════════════════════════════════════════════════════════════════════
	   HACKER AESTHETIC — sharp edges, glowing cyan/green/magenta, scanlines
	   ═══════════════════════════════════════════════════════════════════════ */
	.page {
		max-width: 1600px;
		margin: 0 auto;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
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
	h1 {
		font-size: 1.5rem; margin: 0 0 0.2rem;
		font-family: var(--font-mono); font-weight: 700;
		color: #39d353;
		text-shadow: 0 0 12px rgba(57,211,83,0.3);
	}
	.sub { color: #5a6270; font-size: 0.82rem; margin: 0; letter-spacing: 0.04em; }

	.diff-badge {
		padding: 0.2rem 0.6rem;
		border-radius: 2px;
		font-size: 0.72rem;
		font-weight: 700;
		color: #000;
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	.target-label {
		font-size: 0.72rem;
		color: #5a6270;
		font-weight: 600;
		font-family: var(--font-mono);
	}

	/* Phase badge */
	.phase-badge {
		padding: 0.25rem 0.7rem;
		border-radius: 2px;
		font-size: 0.78rem;
		font-weight: 600;
		background: rgba(0,0,0,0.6);
		border: 1px solid #2a2f38;
		color: #5a6270;
		font-family: var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	.phase-badge.playing    { border-color: #39d353; color: #39d353; box-shadow: 0 0 8px rgba(0,255,159,0.15); }
	.phase-badge.post       { border-color: #56d364; color: #56d364; box-shadow: 0 0 8px rgba(57,255,20,0.15); }
	.phase-badge.done       { border-color: #39d353; color: #39d353; }
	.phase-badge.replaying  { border-color: #39d353; color: #39d353; box-shadow: 0 0 8px rgba(0,229,255,0.15); }
	.phase-badge.waiting    { border-color: #58a6ff; color: #58a6ff; box-shadow: 0 0 8px rgba(88,166,255,0.15); animation: pulse 2s ease-in-out infinite; }

	/* Timer bar */
	.timer-bar {
		background: rgba(13,17,23,0.8);
		border: 1px solid #1a1f28;
		border-radius: 1px;
		padding: 0.5rem 0.75rem;
	}
	.timer-track {
		height: 4px;
		background: #0d1117;
		border-radius: 0;
		overflow: hidden;
		margin-bottom: 0.3rem;
	}
	.timer-fill {
		height: 100%;
		background: linear-gradient(90deg, #39d353, #56d364);
		transition: width 0.3s;
		border-radius: 0;
		box-shadow: 0 0 6px rgba(0,229,255,0.4);
	}
	.timer-fill.warn { background: linear-gradient(90deg, #d29922, #ff6b00); box-shadow: 0 0 6px rgba(229,160,13,0.4); }
	.timer-fill.danger { background: linear-gradient(90deg, #f85149, #da3633); box-shadow: 0 0 8px rgba(255,0,60,0.5); }
	.timer-labels {
		display: flex;
		justify-content: space-between;
		font-size: 0.7rem;
		color: #5a6270;
		font-family: var(--font-mono);
	}
	.timer-remaining { font-weight: 600; }
	.timer-remaining.warn { color: #d29922; }
	.timer-remaining.danger { color: #f85149; animation: pulse 1s ease-in-out infinite; }

	/* Panel */
	.panel {
		background: rgba(13,17,23,0.85);
		border: 1px solid #1a1f28;
		border-radius: 2px;
		padding: 0.85rem;
		box-shadow: 0 2px 16px rgba(0, 0, 0, 0.6);
	}
	.input-panel.collapsed {
		padding: 0.4rem 0.75rem;
	}
	.expand-btn {
		background: none;
		border: none;
		color: #5a6270;
		font-size: 0.72rem;
		cursor: pointer;
		font-family: var(--font-mono);
		padding: 0;
	}
	.expand-btn:hover { color: #39d353; }

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
		background: #0a0e14;
		border: 1px solid #1a1f28;
		border-radius: 1px;
		color: #39d353;
		font-size: 0.82rem;
		font-family: var(--font-mono);
	}
	.url-input:focus { outline: none; border-color: #39d353; box-shadow: 0 0 0 1px rgba(0,229,255,0.2), 0 0 12px rgba(0,229,255,0.1); }
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
		font-size: 0.78rem;
		color: #5a6270;
		font-family: var(--font-mono);
	}
	.num-input {
		width: 65px;
		padding: 0.3rem 0.5rem;
		background: #0a0e14;
		border: 1px solid #1a1f28;
		border-radius: 1px;
		color: #c9d1d9;
		font-size: 0.78rem;
		font-family: var(--font-mono);
	}
	.num-input.sm { width: 45px; }
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 1px;
		font-size: 0.7rem;
		font-weight: 700;
		color: #000;
		white-space: nowrap;
		letter-spacing: 0.04em;
	}

	/* Mode toggle */
	.mode-toggle {
		display: flex;
		background: #0a0e14;
		border: 1px solid #1a1f28;
		border-radius: 1px;
		overflow: hidden;
	}
	.mode-btn {
		padding: 0.3rem 0.7rem;
		border: none;
		background: transparent;
		color: #5a6270;
		font-size: 0.76rem;
		font-weight: 600;
		cursor: pointer;
		font-family: var(--font-mono);
		transition: all 0.15s;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.mode-btn:disabled { opacity: 0.4; cursor: not-allowed; }
	.mode-btn.active {
		background: #39d353;
		color: #0a0e14;
		box-shadow: 0 0 8px rgba(0,229,255,0.3);
	}

	/* Buttons */
	.btn {
		padding: 0.4rem 1rem;
		border: 1px solid transparent;
		border-radius: 1px;
		cursor: pointer;
		font-size: 0.78rem;
		font-weight: 700;
		font-family: var(--font-mono);
		letter-spacing: 0.04em;
		text-transform: uppercase;
	}
	.btn:disabled { opacity: 0.3; cursor: not-allowed; }
	.btn-start   { background: #56d364; color: #0a0e14; border-color: #56d364; box-shadow: 0 0 10px rgba(57,255,20,0.3); }
	.btn-iterate { background: #39d353; color: #0a0e14; border-color: #39d353; box-shadow: 0 0 10px rgba(0,229,255,0.3); }
	.btn-stop    { background: transparent; color: #f85149; border-color: #f85149; box-shadow: 0 0 8px rgba(255,0,60,0.2); }
	.btn-replay  { background: #39d353; color: #0a0e14; margin-top: 0.5rem; box-shadow: 0 0 10px rgba(0,229,255,0.3); }
	.btn-start:hover { box-shadow: 0 0 16px rgba(57,255,20,0.5); }
	.btn-iterate:hover { box-shadow: 0 0 16px rgba(0,229,255,0.5); }
	.btn-stop:hover { background: #f85149; color: #0a0e14; }
	.running-pill {
		font-size: 0.78rem;
		color: #39d353;
		font-family: var(--font-mono);
		animation: pulse 1.5s ease-in-out infinite;
	}
	.new-key-prompt {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 8px 12px;
		background: #1c2d4f;
		border: 1px solid #1f6feb;
		border-radius: 6px;
		flex-wrap: wrap;
	}
	.new-key-label {
		font-size: 0.78rem;
		color: #58a6ff;
		font-family: var(--font-mono);
	}

	/* Stats bar */
	.stats-bar {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
	}
	.stat {
		background: rgba(10,14,20,0.9);
		border: 1px solid #1a1f28;
		border-radius: 1px;
		padding: 0.35rem 0.7rem;
		display: flex;
		flex-direction: column;
		align-items: center;
		flex: 1 1 0;
		min-width: 68px;
		box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
	}
	.best-stat { border-color: #39d353; box-shadow: 0 0 8px rgba(0,229,255,0.15); }
	.stat-label {
		font-size: 0.6rem;
		color: #4a5260;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		font-family: var(--font-mono);
	}
	.stat-val { font-size: 1.05rem; font-weight: 700; font-family: var(--font-mono); }
	.stat-src { font-size: 0.76rem; font-weight: 600; font-family: var(--font-mono); }
	.green  { color: #56d364; }
	.cyan   { color: #39d353; }
	.accent { color: #39d353; }
	.gold   { color: #d29922; }
	.red    { color: #f85149; }

	/* Progress */
	.progress-bar {
		height: 4px;
		background: #0a0e14;
		border-radius: 0;
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, #39d353, #56d364);
		transition: width 0.2s;
		box-shadow: 0 0 6px rgba(0,229,255,0.4);
	}
	.progress-fill.accent { background: #39d353; }
	.progress-txt {
		font-size: 0.68rem;
		color: #4a5260;
		text-align: center;
		margin-top: 0.2rem;
		font-family: var(--font-mono);
	}

	/* Iteration tabs */
	.iter-tabs {
		display: flex;
		gap: 2px;
		flex-wrap: wrap;
		align-items: center;
	}
	.iter-tabs-label {
		font-size: 0.7rem;
		color: #484f58;
		font-family: var(--font-mono);
		font-weight: 600;
		padding-right: 0.3rem;
	}
	.iter-tab {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		padding: 0.3rem 0.55rem;
		background: rgba(10,14,20,0.8);
		border: 1px solid #1a1f28;
		border-radius: 1px 1px 0 0;
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: 0.72rem;
		color: #4a5260;
		transition: all 0.1s;
	}
	.iter-tab.active {
		background: #0d1117;
		border-bottom-color: #0d1117;
		color: #c9d1d9;
		border-top-color: #39d353;
	}
	.iter-tab.best {
		border-color: #39d353;
		box-shadow: 0 0 6px rgba(0,229,255,0.15);
	}
	.tab-num { font-weight: 600; }
	.tab-score { font-weight: 700; }
	.tab-score.best { color: #39d353; text-shadow: 0 0 6px rgba(0,229,255,0.3); }
	.tab-dots { color: #4a5260; animation: pulse 1s ease-in-out infinite; }

	/* 2-row, 3-column equal-width layout */
	.grid-row {
		display: grid;
		grid-template-columns: 1fr 1fr 1fr;
		gap: 0.75rem;
		align-items: start;
	}
	.grid-cell {
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		min-width: 0;
		max-height: 500px;
		overflow-y: auto;
	}
	.grid-cell .log-box {
		max-height: 440px;
	}
	.log-inline h3 { font-size: 0.72rem; }
	.grid-cell .grid-wrap { flex-shrink: 0; }
	.chart-svg-col {
		width: 100%;
		height: 140px;
		background: #0a0e14;
		border-radius: 1px;
		display: block;
	}
	.chart-empty {
		height: 140px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: #0a0e14;
		border-radius: 1px;
	}
	.iter-chart-col { margin-bottom: 0.4rem; }
	.iter-chart-svg {
		width: 100%;
		height: 110px;
		background: #0a0e14;
		border-radius: 2px;
		display: block;
	}
	.iter-list-col {
		display: flex;
		flex-direction: column;
		gap: 1px;
		max-height: 200px;
		overflow-y: auto;
	}
	.sidebar-section h3 {
		font-size: 0.72rem;
		font-weight: 600;
		margin: 0 0 0.35rem;
		color: #4a5260;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		font-family: var(--font-mono);
	}

	/* Orders */
	.order {
		background: #0a0e14;
		border: 1px solid #1a1f28;
		border-radius: 1px;
		padding: 0.3rem 0.45rem;
		margin-bottom: 0.25rem;
	}
	.order.active  { border-color: #d29922; border-left: 2px solid #d29922; }
	.order.preview { border-color: #f472b6; opacity: 0.75; border-left: 2px solid #f472b6; }
	.order-tag {
		font-size: 0.58rem;
		font-weight: 700;
		text-transform: uppercase;
		color: #4a5260;
		display: block;
		margin-bottom: 0.15rem;
		font-family: var(--font-mono);
		letter-spacing: 0.06em;
	}
	.order-items { display: flex; flex-wrap: wrap; gap: 0.15rem; }
	.oitem {
		font-size: 0.66rem;
		padding: 0.1rem 0.25rem;
		background: #161b22;
		border-radius: 1px;
		font-family: var(--font-mono);
	}
	.oitem.del { background: #56d364; color: #0a0e14; font-weight: 600; }

	/* GPU passes */
	.gpu-pass {
		display: flex;
		gap: 0.4rem;
		align-items: center;
		padding: 0.2rem 0.35rem;
		border-radius: 1px;
		font-size: 0.74rem;
		margin-bottom: 0.15rem;
		font-family: var(--font-mono);
	}
	.gpu-pass.best { background: rgba(0,229,255,0.08); border: 1px solid #39d353; }
	.pass-num   { font-weight: 700; color: #4a5260; min-width: 20px; }
	.pass-states { color: #4a5260; font-size: 0.68rem; min-width: 30px; }
	.pass-score { font-weight: 700; }
	.pass-score.best { color: #39d353; text-shadow: 0 0 6px rgba(0,229,255,0.3); }
	.pass-time  { margin-left: auto; color: #3a3f47; }

	/* Bot rows */
	.bot-row {
		display: flex;
		align-items: center;
		gap: 0.35rem;
		padding: 0.15rem 0.3rem;
		font-size: 0.72rem;
		cursor: pointer;
		border-radius: 1px;
		transition: background 0.1s;
		font-family: var(--font-mono);
	}
	.bot-row:hover { background: rgba(0,229,255,0.04); }
	.bot-row.selected { background: rgba(0,229,255,0.08); border-left: 2px solid #39d353; }
	.bot-dot { width: 7px; height: 7px; border-radius: 1px; flex-shrink: 0; }
	.bot-pos { color: #4a5260; min-width: 40px; }
	.bot-inv { color: #c9d1d9; }

	/* Score chart */
	.chart-panel h3 { font-size: 0.72rem; font-weight: 600; margin: 0 0 0.5rem; color: #4a5260; }
	.chart-svg {
		width: 100%;
		height: 120px;
		background: #0a0e14;
		border-radius: 1px;
		display: block;
	}
	.chart-svg-sm {
		width: 100%;
		height: 80px;
		background: #0a0e14;
		border-radius: 1px;
		display: block;
	}
	.chart-section { margin-bottom: 0.2rem; }
	.chart-legend {
		font-size: 0.68rem;
		margin-top: 0.25rem;
		color: #4a5260;
		display: flex;
		gap: 0.8rem;
		font-family: var(--font-mono);
	}

	/* Source transitions */
	.transitions-panel h3 { font-size: 0.72rem; }
	.transition-flow {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		align-items: center;
	}
	.tr-chip {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		background: #0a0e14;
		border: 1px solid #1a1f28;
		border-radius: 1px;
		padding: 0.2rem 0.4rem;
		font-size: 0.72rem;
		font-family: var(--font-mono);
	}
	.tr-round { color: #3a3f47; font-size: 0.66rem; }
	.tr-arrow-in { font-size: 0.6rem; }
	.tr-to { font-weight: 700; }
	.tr-score { color: #56d364; font-weight: 600; font-size: 0.7rem; }
	.tr-sep { color: #2a2f38; font-size: 0.6rem; }

	/* Post-optimize */
	.opt-panel h3 { font-size: 0.72rem; font-weight: 600; margin: 0 0 0.5rem; font-family: var(--font-mono); }
	.opt-row { display: flex; gap: 1rem; flex-wrap: wrap; }
	.opt-stat { display: flex; flex-direction: column; align-items: center; min-width: 80px; }
	.opt-label { font-size: 0.6rem; color: #4a5260; text-transform: uppercase; font-family: var(--font-mono); letter-spacing: 0.06em; }
	.opt-val   { font-size: 1.2rem; font-weight: 700; font-family: var(--font-mono); }

	/* Done panel */
	.done-panel { border-color: #39d353; box-shadow: 0 0 20px rgba(0,229,255,0.1); }
	.done-score {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 0.75rem;
	}
	.done-label { color: #4a5260; font-size: 0.82rem; font-family: var(--font-mono); text-transform: uppercase; }
	.done-val   {
		font-size: 2.5rem; font-weight: 800; color: #39d353;
		font-family: var(--font-mono);
		text-shadow: 0 0 20px rgba(0,229,255,0.3);
	}
	.done-val.hit { color: #56d364; text-shadow: 0 0 24px rgba(57,255,20,0.4); }
	.done-meta {
		display: flex;
		gap: 0.5rem;
		font-size: 0.78rem;
		color: #4a5260;
		font-family: var(--font-mono);
	}
	.hit-badge {
		background: #56d364;
		color: #0a0e14;
		padding: 0.2rem 0.6rem;
		border-radius: 1px;
		font-size: 0.7rem;
		font-weight: 700;
		font-family: var(--font-mono);
		letter-spacing: 0.06em;
		animation: glowPulse 2s ease-in-out infinite;
		box-shadow: 0 0 12px rgba(57,255,20,0.4);
	}

	/* Sidebar iteration section */
	.iter-sidebar h3 { margin-bottom: 0.4rem; }
	.iter-meta {
		font-weight: 400;
		font-size: 0.68rem;
		color: #4a5260;
		margin-left: 0.3rem;
	}
	.iter-chart-sm { margin-bottom: 0.3rem; }
	.iter-list {
		display: flex;
		flex-direction: column;
		gap: 1px;
		max-height: 180px;
		overflow-y: auto;
	}
	.iter-row-sm {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.2rem 0.35rem;
		font-size: 0.72rem;
		background: rgba(10,14,20,0.6);
		cursor: pointer;
		transition: background 0.1s;
		font-family: var(--font-mono);
		border-left: 2px solid transparent;
	}
	.iter-row-sm:hover { background: rgba(0,229,255,0.04); }
	.iter-row-sm.active { background: rgba(0,229,255,0.06); border-left-color: #39d353; }
	.iter-row-sm.best-row { border-left-color: #56d364; }
	.ir-num { color: #3a3f47; min-width: 1.2rem; text-align: right; font-size: 0.65rem; }
	.ir-score { font-weight: 700; min-width: 2.2rem; color: #c9d1d9; }
	.ir-score.best { color: #39d353; text-shadow: 0 0 6px rgba(57,211,83,0.3); }
	.ir-score.hit { color: #56d364; text-shadow: 0 0 6px rgba(57,255,20,0.3); }
	.ir-detail { color: #484f58; font-size: 0.64rem; }
	.ir-time { color: #484f58; font-size: 0.64rem; margin-left: auto; }
	.ir-cap { color: #39d353; font-size: 0.64rem; font-weight: 600; opacity: 0.8; }

	/* Panel h3 */
	.panel h3 {
		font-size: 0.72rem; font-weight: 600; margin: 0 0 0.5rem;
		color: #4a5260; font-family: var(--font-mono);
		text-transform: uppercase; letter-spacing: 0.06em;
	}

	/* Log */
	.log-panel h3 { font-size: 0.72rem; }
	.log-count { font-weight: 400; font-size: 0.66rem; color: #3a3f47; }
	.log-box {
		background: #0a0e14;
		border-radius: 1px;
		padding: 0.4rem;
		max-height: 280px;
		overflow-y: auto;
		font-size: 0.68rem;
		font-family: var(--font-mono);
		border: 1px solid #1a1f28;
	}
	.log-line {
		display: flex;
		gap: 0.4rem;
		padding: 0.1rem 0;
		border-bottom: 1px solid #0d1117;
	}
	.log-line:last-child { border-bottom: none; }
	.log-line.error { color: #f85149; text-shadow: 0 0 4px rgba(248,81,73,0.3); background: rgba(248,81,73,0.05); }
	.log-line.desync { color: #da3633; text-shadow: 0 0 6px rgba(218,54,51,0.4); background: rgba(218,54,51,0.08); font-weight: 600; }
	.log-line.upgrade { color: #56d364; }
	.log-line.gpu-log { color: #58a6ff; }
	.log-line.score-log { color: #39d353; font-weight: 600; }
	.log-line.iter-start { color: #d29922; border-top: 1px solid #1a1f28; font-weight: 600; }
	.log-line.seed-log { color: #bc8cff; }
	.log-line.warn-log { color: #d29922; }
	.log-line.greedy-log { color: #f472b6; font-weight: 600; }
	.log-t { color: #2a2f38; white-space: nowrap; flex-shrink: 0; }
	.log-iter {
		color: #39d353;
		font-weight: 600;
		font-size: 0.64rem;
		white-space: nowrap;
		flex-shrink: 0;
	}
	.muted { color: #4a5260; }

	.replay-section { margin-top: 0.5rem; }

	@keyframes pulse {
		0%, 100% { opacity: 1; }
		50%       { opacity: 0.3; }
	}
	@keyframes glowPulse {
		0%, 100% { box-shadow: 0 0 12px rgba(57,255,20,0.4); }
		50%       { box-shadow: 0 0 24px rgba(57,255,20,0.7); }
	}

	@media (max-width: 1200px) {
		.grid-row { grid-template-columns: 1fr 1fr; }
	}
	@media (max-width: 800px) {
		.grid-row { grid-template-columns: 1fr; }
		.grid-cell { max-height: none; }
	}
</style>
