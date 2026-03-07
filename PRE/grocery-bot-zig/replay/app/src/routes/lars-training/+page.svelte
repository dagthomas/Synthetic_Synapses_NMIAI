<script>
	import { onMount, onDestroy } from 'svelte';

	// === MODE ===
	let mode = $state('live'); // 'train' | 'live'

	// === SHARED STATE ===
	let status = $state('idle'); // idle, training, complete, error
	let selectedDifficulty = $state('expert');
	let solutions = $state({ easy: null, medium: null, hard: null, expert: null });
	let terminalLines = $state([]);
	let terminalEl = $state(null);
	let abortController = null;

	// === TRAIN MODE STATE ===
	let maxStates = $state(100000);
	let refineIters = $state(5);
	let currentBotId = $state(-1);
	let totalBots = $state(1);
	let botScores = $state([]);
	let roundData = $state([]);
	let finalScore = $state(0);
	let finalTime = $state(0);
	let scoreCanvas = $state(null);

	// === LIVE MODE STATE ===
	let wsUrl = $state('');
	let liveBotType = $state('zig'); // 'zig' | 'python'
	let timeBudget = $state(280);
	let gpuOptTime = $state(20);
	let postOptTime = $state(30);
	let liveIterations = $state([]);
	let liveBestScore = $state(0);
	let liveIterCount = $state(0);
	let liveDifficulty = $state(null);
	let liveStartTime = $state(0);
	let liveElapsed = $state(0);
	let liveTimerInterval = null;

	function addTerminal(text, type = 'info') {
		terminalLines.push({ text, type, ts: Date.now() });
		if (terminalLines.length > 300) terminalLines = terminalLines.slice(-200);
		requestAnimationFrame(() => {
			if (terminalEl) terminalEl.scrollTop = terminalEl.scrollHeight;
		});
	}

	// === GRAPH ===
	function drawScoreGraph() {
		if (!scoreCanvas || roundData.length === 0) return;
		const canvas = scoreCanvas;
		const ctx = canvas.getContext('2d');
		const W = canvas.width;
		const H = canvas.height;
		const pad = { t: 20, r: 15, b: 30, l: 50 };
		const gW = W - pad.l - pad.r;
		const gH = H - pad.t - pad.b;

		ctx.clearRect(0, 0, W, H);
		ctx.fillStyle = '#0d1117';
		ctx.fillRect(0, 0, W, H);

		const maxScore = Math.max(finalScore, ...roundData.map(d => d.score), 10);
		const maxR = 300;

		// Grid
		ctx.strokeStyle = 'rgba(48, 54, 61, 0.6)';
		ctx.lineWidth = 1;
		for (let s = 0; s <= maxScore; s += 25) {
			const y = pad.t + gH - (s / maxScore) * gH;
			ctx.beginPath();
			ctx.moveTo(pad.l, y);
			ctx.lineTo(pad.l + gW, y);
			ctx.stroke();
			ctx.fillStyle = '#8b949e';
			ctx.font = '10px JetBrains Mono, monospace';
			ctx.textAlign = 'right';
			ctx.fillText(String(s), pad.l - 5, y + 3);
		}

		// Score line
		if (roundData.length > 1) {
			ctx.strokeStyle = '#39d353';
			ctx.lineWidth = 2;
			ctx.beginPath();
			for (let i = 0; i < roundData.length; i++) {
				const d = roundData[i];
				const x = pad.l + (d.r / maxR) * gW;
				const y = pad.t + gH - (d.score / maxScore) * gH;
				if (i === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.stroke();
		}

		// Target line (175)
		const targetY = pad.t + gH - (175 / maxScore) * gH;
		if (targetY > pad.t) {
			ctx.strokeStyle = 'rgba(210, 153, 34, 0.5)';
			ctx.setLineDash([5, 5]);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(pad.l, targetY);
			ctx.lineTo(pad.l + gW, targetY);
			ctx.stroke();
			ctx.setLineDash([]);
			ctx.fillStyle = '#d29922';
			ctx.font = '10px JetBrains Mono, monospace';
			ctx.textAlign = 'left';
			ctx.fillText('target: 175', pad.l + gW - 80, targetY - 5);
		}
	}

	// === TRAIN MODE ===
	async function startTraining() {
		if (status === 'training') return;

		status = 'training';
		terminalLines = [];
		roundData = [];
		finalScore = 0;
		finalTime = 0;
		currentBotId = -1;
		totalBots = 1;
		botScores = [];

		addTerminal(`[LARS] Starting ${selectedDifficulty.toUpperCase()} training on RTX 3060`, 'system');
		if (selectedDifficulty === 'expert') {
			addTerminal(`[LARS] Using cracked seed solver (solve_expert_3060.py)`, 'system');
			addTerminal(`[LARS] max_states=${maxStates}, refine_iters=${refineIters}`, 'info');
		} else {
			addTerminal(`[LARS] Using standard GPU solver (gpu_multi_solve_stream.py)`, 'system');
		}

		try {
			abortController = new AbortController();
			const res = await fetch('/api/training/solve', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					difficulty: selectedDifficulty,
					maxStates: selectedDifficulty === 'expert' ? maxStates : undefined,
					refineIters: selectedDifficulty === 'expert' ? refineIters : undefined,
				}),
				signal: abortController.signal,
			});

			const reader = res.body.getReader();
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
						handleTrainEvent(JSON.parse(line.slice(6)));
					} catch (e) { /* malformed SSE line */ }
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				addTerminal(`[ERROR] ${e.message}`, 'error');
				status = 'error';
			}
		}
		if (status === 'training') status = 'complete';
		loadSolutions();
	}

	function stopTraining() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		status = 'idle';
		addTerminal('[LARS] Training aborted', 'warn');
	}

	function handleTrainEvent(data) {
		switch (data.type) {
			case 'training_start':
				addTerminal(`[START] Script: ${data.script} | Difficulty: ${data.difficulty}`, 'system');
				break;
			case 'init':
				totalBots = data.num_bots || 1;
				addTerminal(`[GPU] ${data.gpu_name || 'GPU'} | ${data.vram_total || '?'}GB VRAM`, 'system');
				addTerminal(`[MAP] ${data.width}x${data.height} | ${data.items} items | ${data.num_bots || 1} bots`, 'info');
				break;
			case 'bot_start':
				currentBotId = data.bot_id;
				totalBots = data.total_bots;
				addTerminal(`[BOT ${data.bot_id}/${data.total_bots}] Starting DP solve...`, 'system');
				break;
			case 'bot_done':
				botScores = [...botScores, { bot_id: data.bot_id, score: data.score }];
				addTerminal(`[BOT ${data.bot_id}/${data.total_bots}] Done: score=${data.score} (${data.time}s)`, 'success');
				break;
			case 'round':
				roundData = [...roundData, {
					r: data.r, score: data.score,
					unique: data.unique, expanded: data.expanded,
					time: data.time, bot_id: data.bot_id,
				}];
				finalScore = data.score;
				finalTime = data.time;
				if (data.r < 10 || data.r % 25 === 0 || data.r === 299) {
					const bp = totalBots > 1 ? `B${data.bot_id ?? 0} ` : '';
					addTerminal(
						`[${bp}R${String(data.r).padStart(3)}] score=${String(data.score).padStart(3)} ` +
						`unique=${String(data.unique).toLocaleString().padStart(7)} ` +
						`t=${data.time.toFixed(1)}s`,
						data.score > 0 ? 'info' : 'dim'
					);
				}
				drawScoreGraph();
				break;
			case 'result':
				finalScore = data.score;
				finalTime = data.time;
				addTerminal(`[RESULT] Score: ${data.score} | Time: ${data.time}s | Bots: ${data.num_bots || 1}`, 'success');
				break;
			case 'improved':
				addTerminal(`[IMPROVED] ${data.old_score} -> ${data.new_score} (+${data.delta})`, 'success');
				break;
			case 'no_improvement':
				addTerminal(`[NO CHANGE] ${data.score} <= ${data.prev}`, 'warn');
				break;
			case 'done':
				status = 'complete';
				addTerminal(`[DONE] Training complete: score=${data.score} in ${data.time}s`, 'success');
				loadSolutions();
				break;
			case 'stderr':
			case 'log':
				addTerminal(data.text, 'dim');
				break;
			case 'error':
				addTerminal(`[ERROR] ${data.msg || data.message}`, 'error');
				status = 'error';
				break;
			case 'process_done':
				if (status === 'training') {
					status = 'complete';
					addTerminal(`[LARS] Process exited (code ${data.code})`, data.code === 0 ? 'success' : 'warn');
				}
				loadSolutions();
				break;
		}
	}

	// === LIVE MODE ===
	async function startLive() {
		if (status === 'training' || !wsUrl.trim()) return;

		status = 'training';
		terminalLines = [];
		liveIterations = [];
		liveBestScore = 0;
		liveIterCount = 0;
		liveDifficulty = null;
		liveStartTime = Date.now();
		liveElapsed = 0;

		liveTimerInterval = setInterval(() => {
			liveElapsed = (Date.now() - liveStartTime) / 1000;
		}, 250);

		addTerminal(`[LIVE] Starting pipeline: bot=${liveBotType}, budget=${timeBudget}s, gpu-opt=${gpuOptTime}s, post-opt=${postOptTime}s`, 'system');
		addTerminal(`[LIVE] URL: ${wsUrl.slice(0, 60)}...`, 'dim');

		try {
			abortController = new AbortController();
			const res = await fetch('/api/pipeline/iterate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					url: wsUrl.trim(),
					timeBudget,
					gpuOptimizeTime: gpuOptTime,
					postOptimizeTime: postOptTime,
					botType: liveBotType,
				}),
				signal: abortController.signal,
			});

			if (!res.ok) {
				const errText = await res.text();
				addTerminal(`[ERROR] Server returned ${res.status}: ${errText}`, 'error');
				status = 'error';
				return;
			}

			const reader = res.body.getReader();
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
						handleLiveEvent(JSON.parse(line.slice(6)));
					} catch (e) { /* malformed SSE line */ }
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				addTerminal(`[ERROR] ${e.message}`, 'error');
				status = 'error';
			}
		}

		if (liveTimerInterval) { clearInterval(liveTimerInterval); liveTimerInterval = null; }
		if (status === 'training') status = 'complete';
		loadSolutions();
	}

	function stopLive() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		if (liveTimerInterval) { clearInterval(liveTimerInterval); liveTimerInterval = null; }
		status = 'idle';
		addTerminal('[LIVE] Pipeline aborted', 'warn');
	}

	function handleLiveEvent(data) {
		switch (data.type) {
			case 'pipeline_start':
				addTerminal(`[PIPELINE] Started: budget=${data.time_budget}s`, 'system');
				break;

			case 'iter_start':
				addTerminal(`[ITER ${data.iter}] Phase: ${data.phase} | elapsed=${data.elapsed?.toFixed(0)}s | remaining=${data.remaining?.toFixed(0)}s`, 'system');
				break;

			case 'iter_done': {
				const iter = {
					num: data.iter,
					phase: data.phase,
					score: data.score || 0,
					gameScore: data.game_score || 0,
					optScore: data.opt_score || 0,
					orders: data.captured_orders || 0,
					elapsed: data.elapsed?.toFixed(0) || '?',
					remaining: data.remaining?.toFixed(0) || '?',
				};
				liveIterations = [...liveIterations, iter];
				liveIterCount = data.iter + 1;
				if (data.difficulty) liveDifficulty = data.difficulty;
				if (iter.score > liveBestScore) liveBestScore = iter.score;
				addTerminal(
					`[ITER ${data.iter} DONE] score=${iter.score} game=${iter.gameScore} orders=${iter.orders} remaining=${iter.remaining}s`,
					iter.score >= liveBestScore ? 'success' : 'info'
				);
				loadSolutions(); // refresh order counts
				break;
			}

			case 'iter_summary':
				if (data.best_score > liveBestScore) liveBestScore = data.best_score;
				addTerminal(`[SUMMARY] iter=${data.iter} best=${data.best_score} remaining=${data.remaining?.toFixed(0)}s`, 'info');
				break;

			case 'iter_skip':
				addTerminal(`[SKIP] iter=${data.iter} reason=${data.reason} remaining=${data.remaining?.toFixed(0)}s`, 'warn');
				break;

			case 'init':
				if (data.difficulty) liveDifficulty = data.difficulty;
				addTerminal(`[MAP] ${data.width}x${data.height} | ${data.num_bots || data.bot_count || '?'} bots | ${data.difficulty || '?'}`, 'system');
				break;

			case 'round':
				// Only log milestone rounds to avoid flooding
				if (data.round !== undefined && (data.round % 50 === 0 || data.round === 299)) {
					addTerminal(`[R${data.round}] score=${data.score}`, 'dim');
				}
				break;

			case 'game_over':
				addTerminal(`[GAME OVER] score=${data.score}`, 'success');
				break;

			case 'optimize_phase_start':
				addTerminal(`[GPU OPT] iter=${data._iter} time=${data.max_time}s warm=${data.warm_only}`, 'system');
				break;

			case 'optimize_done':
				addTerminal(`[GPU OPT DONE] score=${data.score} prev=${data.prev_score}`, data.score > (data.prev_score || 0) ? 'success' : 'info');
				if (data.score > liveBestScore) liveBestScore = data.score;
				break;

			case 'replay_phase_start':
				addTerminal(`[REPLAY] iter=${data._iter} starting replay...`, 'system');
				break;

			case 'replay_phase_done':
				addTerminal(`[REPLAY DONE] score=${data.score}`, data.score > 0 ? 'success' : 'warn');
				if (data.score > liveBestScore) liveBestScore = data.score;
				break;

			case 'pipeline_complete':
				liveBestScore = data.best_score || liveBestScore;
				liveIterCount = data.iterations || liveIterCount;
				if (data.difficulty) liveDifficulty = data.difficulty;
				status = 'complete';
				addTerminal(`[PIPELINE COMPLETE] best=${data.best_score} iters=${data.iterations} elapsed=${data.total_elapsed?.toFixed(0)}s`, 'success');
				loadSolutions();
				break;

			case 'log':
				addTerminal(data.text, 'dim');
				break;

			case 'error':
				addTerminal(`[ERROR] ${data.msg || data.message}`, 'error');
				break;

			default:
				if (data.type && !['pipeline_done', 'stderr'].includes(data.type)) {
					addTerminal(`[${data.type}] ${JSON.stringify(data).slice(0, 120)}`, 'dim');
				}
				break;
		}
	}

	// === SOLUTIONS ===
	async function loadSolutions() {
		try {
			const res = await fetch('/api/optimize/solutions');
			if (res.ok) solutions = await res.json();
		} catch (e) { /* ignore */ }
	}

	// === LIFECYCLE ===
	onMount(() => {
		loadSolutions();
	});

	onDestroy(() => {
		if (abortController) abortController.abort();
		if (liveTimerInterval) clearInterval(liveTimerInterval);
	});

	// Derived (train mode)
	let currentRound = $derived(roundData.length > 0 ? roundData[roundData.length - 1].r : 0);
	let progressPct = $derived(
		totalBots > 1
			? ((botScores.length * 300 + currentRound) / (totalBots * 300) * 100)
			: (currentRound / 300 * 100)
	);

	// Derived (live mode)
	let timerPct = $derived(timeBudget > 0 ? Math.min(100, liveElapsed / timeBudget * 100) : 0);
	let timerRemaining = $derived(Math.max(0, timeBudget - liveElapsed));
	let timerWarn = $derived(timerRemaining < 30);
	let timerCritical = $derived(timerRemaining < 10);
</script>

<svelte:head>
	<title>Lars Training | RTX 3060</title>
</svelte:head>

<div class="training-page stagger">
	<!-- Header -->
	<div class="page-header">
		<h1>Lars Training <span class="gpu-badge">RTX 3060</span></h1>
		<div class="header-meta">
			<span class="chip">12GB VRAM</span>
			<span class="chip">Zig Bot + Python GPU</span>
			<span class="chip status-{status}">
				{#if status === 'idle'}Ready
				{:else if status === 'training'}{mode === 'live' ? 'Live...' : 'Training...'}
				{:else if status === 'complete'}Complete
				{:else}Error
				{/if}
			</span>
		</div>
	</div>

	<!-- Mode toggle -->
	<div class="mode-toggle">
		<button class="mode-btn" class:active={mode === 'train'} onclick={() => { if (status === 'idle') mode = 'train'; }}>
			Train
		</button>
		<button class="mode-btn" class:active={mode === 'live'} onclick={() => { if (status === 'idle') mode = 'live'; }}>
			Live
		</button>
	</div>

	{#if mode === 'train'}
		<!-- ==================== TRAIN MODE ==================== -->

		<!-- Bot info -->
		<div class="bots-row">
			<div class="card bot-card">
				<div class="bot-icon">Z</div>
				<div>
					<div class="bot-name">Zig Bot</div>
					<div class="bot-desc">Live decision engine, real-time play</div>
					<div class="bot-detail">Decision cascade, anti-oscillation, trip planner</div>
				</div>
			</div>
			<div class="card bot-card">
				<div class="bot-icon gpu-icon">G</div>
				<div>
					<div class="bot-name">Python GPU Bot</div>
					<div class="bot-desc">Offline GPU optimizer, sequential DP</div>
					<div class="bot-detail">PyTorch/CUDA, cracked seed (expert), beam search</div>
				</div>
			</div>
		</div>

		<!-- Solution cards -->
		<h2 class="section-title">Best Solutions</h2>
		<div class="solutions-row">
			{#each ['easy', 'medium', 'hard', 'expert'] as diff}
				{@const sol = solutions[diff]}
				{@const isStale = sol?.date && sol.date !== new Date().toISOString().slice(0, 10)}
				{@const BOT_COUNTS = { easy: 1, medium: 3, hard: 5, expert: 10 }}
				<button
					class="card solution-card"
					class:active={selectedDifficulty === diff}
					class:has-solution={sol && sol.score > 0}
					class:stale={isStale}
					onclick={() => selectedDifficulty = diff}
				>
					<div class="sol-diff">{diff}</div>
					<div class="sol-score">{sol?.score ?? '---'}</div>
					<div class="sol-meta">
						{#if sol?.score > 0}
							{sol.date || ''} | {BOT_COUNTS[diff]} bots
							{#if isStale}<span class="tag tag-red">old map</span>{/if}
						{:else}
							no solution | {BOT_COUNTS[diff]} bots
						{/if}
					</div>
					{#if sol?.orders}
						<div class="sol-orders">{sol.orders} orders</div>
					{/if}
					<div class="sol-target">target: 175</div>
				</button>
			{/each}
		</div>

		<!-- Training controls -->
		<div class="card controls-section">
			<h2 class="section-title-sm">Train: {selectedDifficulty.toUpperCase()}</h2>

			{#if selectedDifficulty === 'expert'}
				<div class="config-row">
					<label>
						<span class="config-label">Max States</span>
						<input type="number" class="config-input" bind:value={maxStates}
							min="10000" max="500000" step="10000" disabled={status === 'training'} />
						<span class="config-hint">100K safe for 12GB</span>
					</label>
					<label>
						<span class="config-label">Refine Iterations</span>
						<input type="number" class="config-input" bind:value={refineIters}
							min="1" max="20" disabled={status === 'training'} />
						<span class="config-hint">More = better but slower</span>
					</label>
				</div>
				<div class="solver-info">
					Using <strong>solve_expert_3060.py</strong> — cracked seed solver with all 50 future orders pre-generated
				</div>
			{:else}
				<div class="solver-info">
					Using <strong>gpu_multi_solve_stream.py</strong> — standard GPU sequential DP solver
				</div>
			{/if}

			<div class="controls-row">
				{#if status === 'training'}
					<button class="btn btn-danger" onclick={stopTraining}>Abort</button>
				{:else}
					<button class="btn btn-primary" onclick={startTraining}>
						Start Training ({selectedDifficulty})
					</button>
					<button class="btn btn-secondary" onclick={loadSolutions}>Refresh Solutions</button>
				{/if}
			</div>
		</div>

		<!-- Progress bar -->
		{#if status === 'training' && currentRound > 0}
			<div class="progress-container">
				<div class="progress-bar" style="width: {progressPct}%"></div>
				<div class="progress-text">
					{progressPct.toFixed(0)}% --
					{#if totalBots > 1}Bot {Math.max(0, currentBotId)}/{totalBots}, {/if}Round {currentRound}/300
				</div>
			</div>
		{/if}

		<!-- Score graph -->
		{#if roundData.length > 0}
			<div class="card graph-card">
				<h3>Score Progression</h3>
				<canvas bind:this={scoreCanvas} width="800" height="250"></canvas>
			</div>
		{/if}

		<!-- Stats row -->
		{#if status === 'training' || status === 'complete'}
			<div class="stats-row">
				<div class="card stat-card">
					<div class="stat-label">Score</div>
					<div class="stat-value score-value">{finalScore}</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Time</div>
					<div class="stat-value">{finalTime.toFixed(1)}s</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Round</div>
					<div class="stat-value">{currentRound}/300</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Bots</div>
					<div class="stat-value">
						{#if totalBots > 1}{botScores.length}/{totalBots}{:else}1{/if}
					</div>
				</div>
			</div>
		{/if}

	{:else}
		<!-- ==================== LIVE MODE ==================== -->

		<!-- URL input -->
		<div class="card live-url-section">
			<h2 class="section-title-sm">WebSocket Token URL</h2>
			<div class="live-url-row">
				<input
					type="text" class="live-url-input" bind:value={wsUrl}
					placeholder="wss://game.ainm.no/ws?token=..."
					disabled={status === 'training'}
				/>
			</div>
		</div>

		<!-- Bot type selector -->
		<div class="bot-selector">
			<span class="config-label">Initial Bot</span>
			<div class="mode-toggle bot-toggle">
				<button class="mode-btn" class:active={liveBotType === 'zig'}
					onclick={() => { if (status !== 'training') liveBotType = 'zig'; }}>
					Zig Bot
				</button>
				<button class="mode-btn" class:active={liveBotType === 'python'}
					onclick={() => { if (status !== 'training') liveBotType = 'python'; }}>
					Python GPU
				</button>
			</div>
			<span class="config-hint">
				{#if liveBotType === 'zig'}Reactive play, discovers many orders (best for expert)
				{:else}GPU-assisted decisions (fewer orders discovered)
				{/if}
			</span>
		</div>

		<!-- Live config -->
		<div class="card controls-section">
			<h2 class="section-title-sm">Pipeline Config</h2>
			<div class="config-row">
				<label>
					<span class="config-label">Time Budget (s)</span>
					<input type="number" class="config-input" bind:value={timeBudget}
						min="30" max="300" step="10" disabled={status === 'training'} />
					<span class="config-hint">Token lasts ~288s</span>
				</label>
				<label>
					<span class="config-label">GPU Opt Time (s)</span>
					<input type="number" class="config-input" bind:value={gpuOptTime}
						min="5" max="60" step="5" disabled={status === 'training'} />
					<span class="config-hint">Per iteration</span>
				</label>
				{#if liveBotType === 'python'}
				<label>
					<span class="config-label">Post-Opt Time (s)</span>
					<input type="number" class="config-input" bind:value={postOptTime}
						min="5" max="60" step="5" disabled={status === 'training'} />
					<span class="config-hint">GPU optimize after live play</span>
				</label>
				{/if}
			</div>

			<div class="solver-info">
				{#if liveBotType === 'zig'}
					Pipeline: <strong>Zig bot</strong> play -> capture orders -> GPU optimize -> replay -> repeat
				{:else}
					Pipeline: <strong>Python GPU</strong> live play (+{postOptTime}s post-opt) -> GPU optimize -> replay -> repeat
				{/if}
			</div>

			<div class="controls-row">
				{#if status === 'training'}
					<button class="btn btn-danger" onclick={stopLive}>Abort Pipeline</button>
				{:else}
					<button class="btn btn-primary" onclick={startLive} disabled={!wsUrl.trim()}>
						Start Pipeline
					</button>
					<button class="btn btn-secondary" onclick={loadSolutions}>Refresh Solutions</button>
				{/if}
			</div>
		</div>

		<!-- Timer bar -->
		{#if status === 'training'}
			<div class="progress-container" class:timer-warn={timerWarn} class:timer-critical={timerCritical}>
				<div class="progress-bar timer-bar" style="width: {timerPct}%"></div>
				<div class="progress-text">
					{timerRemaining.toFixed(0)}s remaining | {liveElapsed.toFixed(0)}s elapsed
				</div>
			</div>
		{/if}

		<!-- Live stats -->
		{#if status === 'training' || status === 'complete'}
			<div class="stats-row stats-5">
				<div class="card stat-card">
					<div class="stat-label">Best Score</div>
					<div class="stat-value score-value">{liveBestScore}</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Iterations</div>
					<div class="stat-value">{liveIterCount}</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Orders</div>
					<div class="stat-value">{liveDifficulty ? (solutions[liveDifficulty]?.orders || 0) : '---'}</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Elapsed</div>
					<div class="stat-value">{liveElapsed.toFixed(0)}s</div>
				</div>
				<div class="card stat-card">
					<div class="stat-label">Difficulty</div>
					<div class="stat-value">{liveDifficulty || '---'}</div>
				</div>
			</div>
		{/if}

		<!-- Iteration results -->
		{#if liveIterations.length > 0}
			<div class="card iterations-section">
				<h2 class="section-title-sm">Iteration Results</h2>
				<div class="iterations-list">
					{#each liveIterations as iter}
						<div class="iteration-row" class:best={iter.score === liveBestScore && iter.score > 0}>
							<span class="iter-num">#{iter.num}</span>
							<span class="iter-phase">{iter.phase}</span>
							<span class="iter-score">score: {iter.score}</span>
							{#if iter.gameScore > 0}
								<span class="iter-game">game: {iter.gameScore}</span>
							{/if}
							{#if iter.optScore > 0}
								<span class="iter-opt">opt: {iter.optScore}</span>
							{/if}
							{#if iter.orders > 0}
								<span class="iter-orders">+{iter.orders} orders</span>
							{/if}
							<span class="iter-time">{iter.remaining}s left</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}

	<!-- Terminal (shared) -->
	<div class="card terminal-section">
		<div class="terminal-header">
			<h3>{mode === 'live' ? 'Pipeline Log' : 'Training Log'}</h3>
			{#if terminalLines.length > 0}
				<button class="btn-clear" onclick={() => terminalLines = []}>Clear</button>
			{/if}
		</div>
		<div class="terminal" bind:this={terminalEl}>
			{#if terminalLines.length === 0}
				{#if mode === 'live'}
					<div class="terminal-line dim">$ Paste token URL and press Start Pipeline.</div>
					<div class="terminal-line dim">$ Pipeline: Zig bot -> capture -> GPU optimize -> replay -> repeat</div>
				{:else}
					<div class="terminal-line dim">$ Ready. Select difficulty and press Start Training.</div>
					<div class="terminal-line dim">$ Expert mode uses cracked seed solver optimized for RTX 3060 (12GB)</div>
				{/if}
			{/if}
			{#each terminalLines as line}
				<div class="terminal-line {line.type}">{line.text}</div>
			{/each}
			{#if status === 'training'}
				<div class="terminal-line dim cursor-line">_</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.training-page {
		max-width: 1200px;
		margin: 0 auto;
	}

	/* Header */
	.page-header {
		margin-bottom: 1.25rem;
	}

	.page-header h1 {
		font-size: 1.5rem;
		font-family: var(--font-display);
		font-weight: 700;
		color: var(--text);
		margin-bottom: 0.5rem;
	}

	.gpu-badge {
		font-size: 0.9rem;
		color: var(--green);
		border: 1px solid var(--green);
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		margin-left: 0.5rem;
		font-weight: 600;
	}

	.header-meta {
		display: flex;
		gap: 0.5rem;
		flex-wrap: wrap;
	}

	.chip {
		font-size: 0.75rem;
		padding: 0.2rem 0.6rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text-muted);
		background: var(--bg-card);
	}

	.chip.status-training { color: var(--green); border-color: var(--green); }
	.chip.status-complete { color: var(--orange); border-color: var(--orange); }
	.chip.status-error { color: var(--red); border-color: var(--red); }

	/* Mode toggle */
	.mode-toggle {
		display: flex;
		gap: 2px;
		margin-bottom: 1.25rem;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 3px;
		width: fit-content;
	}

	.mode-btn {
		font-size: 0.8rem;
		padding: 0.4rem 1.25rem;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		font-weight: 600;
		color: var(--text-muted);
		background: transparent;
		transition: all 0.15s ease;
		letter-spacing: 0.03em;
	}

	.mode-btn:hover:not(.active) {
		color: var(--text);
		background: var(--bg-hover);
	}

	.mode-btn.active {
		color: #0d1117;
		background: var(--accent);
	}

	/* Bots info */
	.bots-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
		margin-bottom: 1.25rem;
	}

	.bot-card {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.75rem 1rem;
	}

	.bot-icon {
		width: 36px;
		height: 36px;
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 1rem;
		font-weight: 800;
		color: var(--blue);
		border: 1px solid var(--blue);
		background: rgba(88, 166, 255, 0.08);
		flex-shrink: 0;
	}

	.bot-icon.gpu-icon {
		color: var(--green);
		border-color: var(--green);
		background: rgba(57, 211, 83, 0.08);
	}

	.bot-name {
		font-size: 0.85rem;
		font-weight: 600;
		color: var(--text);
	}

	.bot-desc {
		font-size: 0.7rem;
		color: var(--text-muted);
	}

	.bot-detail {
		font-size: 0.65rem;
		color: var(--text-muted);
		opacity: 0.7;
	}

	/* Section titles */
	.section-title {
		font-size: 0.9rem;
		font-weight: 600;
		color: var(--text-muted);
		margin-bottom: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.section-title-sm {
		font-size: 0.85rem;
		font-weight: 600;
		color: var(--text);
		margin-bottom: 0.75rem;
	}

	/* Solutions */
	.solutions-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.75rem;
		margin-bottom: 1.25rem;
	}

	.solution-card {
		text-align: center;
		padding: 0.75rem;
		cursor: pointer;
		transition: all 0.15s ease;
		font-family: inherit;
		color: inherit;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}

	.solution-card:hover {
		border-color: var(--accent-light);
		background: var(--bg-hover);
	}

	.solution-card.active {
		border-color: var(--accent);
		box-shadow: 0 0 0 1px var(--accent), 0 0 16px rgba(57, 211, 83, 0.08);
	}

	.solution-card.stale { opacity: 0.6; }

	.sol-diff {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-transform: capitalize;
		margin-bottom: 0.15rem;
	}

	.sol-score {
		font-size: 1.4rem;
		font-family: var(--font-mono);
		font-weight: 700;
		color: var(--text);
		line-height: 1.3;
	}

	.solution-card.has-solution .sol-score {
		color: var(--green);
	}

	.sol-meta {
		font-size: 0.65rem;
		color: var(--text-muted);
	}

	.sol-orders {
		font-size: 0.6rem;
		color: var(--blue);
		margin-top: 0.1rem;
	}

	.sol-target {
		font-size: 0.6rem;
		color: var(--orange);
		margin-top: 0.25rem;
	}

	.tag {
		display: inline-block;
		font-size: 0.65rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		font-weight: 600;
	}

	.tag-red {
		background: rgba(248, 81, 73, 0.15);
		color: var(--red);
	}

	/* Controls */
	.controls-section {
		margin-bottom: 1rem;
	}

	.config-row {
		display: flex;
		gap: 1.5rem;
		margin-bottom: 0.75rem;
	}

	.config-row label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.config-label {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.03em;
	}

	.config-input {
		width: 140px;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 0.8rem;
		padding: 0.4rem 0.6rem;
		outline: none;
	}

	.config-input:focus {
		border-color: var(--accent);
	}

	.config-hint {
		font-size: 0.65rem;
		color: var(--text-muted);
	}

	.solver-info {
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-bottom: 0.75rem;
		padding: 0.4rem 0.6rem;
		background: rgba(88, 166, 255, 0.05);
		border: 1px solid rgba(88, 166, 255, 0.15);
		border-radius: var(--radius);
	}

	.solver-info strong {
		color: var(--blue);
	}

	.controls-row {
		display: flex;
		gap: 0.75rem;
		align-items: center;
	}

	.btn {
		font-size: 0.8rem;
		padding: 0.5rem 1.25rem;
		border-radius: var(--radius);
		border: 1px solid;
		cursor: pointer;
		font-weight: 500;
		transition: all 0.15s ease;
		letter-spacing: 0.02em;
	}

	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn-primary {
		color: #0d1117;
		border-color: var(--accent);
		background: var(--accent);
	}

	.btn-primary:hover:not(:disabled) { opacity: 0.85; }

	.btn-secondary {
		color: var(--text);
		border-color: var(--border);
		background: var(--bg-hover);
	}

	.btn-secondary:hover { border-color: var(--accent-light); }

	.btn-danger {
		color: white;
		border-color: var(--red);
		background: var(--red);
	}

	.btn-danger:hover { opacity: 0.85; }

	/* Bot selector */
	.bot-selector {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.bot-toggle {
		margin-bottom: 0;
	}

	/* Live URL */
	.live-url-section {
		margin-bottom: 1rem;
	}

	.live-url-row {
		display: flex;
		gap: 0.75rem;
	}

	.live-url-input {
		flex: 1;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 0.8rem;
		padding: 0.5rem 0.75rem;
		outline: none;
	}

	.live-url-input:focus {
		border-color: var(--accent);
	}

	.live-url-input:disabled {
		opacity: 0.5;
	}

	/* Progress / Timer */
	.progress-container {
		position: relative;
		height: 24px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		margin-bottom: 1rem;
		overflow: hidden;
		box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
	}

	.progress-bar {
		height: 100%;
		background: var(--accent);
		transition: width 0.3s ease;
	}

	.timer-bar {
		background: var(--blue);
	}

	.progress-container.timer-warn .timer-bar {
		background: var(--orange);
	}

	.progress-container.timer-critical .timer-bar {
		background: var(--red);
	}

	.progress-container.timer-warn {
		border-color: var(--orange);
	}

	.progress-container.timer-critical {
		border-color: var(--red);
	}

	.progress-text {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		font-size: 0.7rem;
		color: var(--text);
	}

	/* Graph */
	.graph-card {
		margin-bottom: 1rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}

	.graph-card h3 {
		font-size: 0.8rem;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
		font-weight: 500;
	}

	.graph-card canvas {
		width: 100%;
		height: auto;
		display: block;
		border-radius: 4px;
		box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.3);
	}

	/* Stats */
	.stats-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.stats-5 {
		grid-template-columns: repeat(5, 1fr);
	}

	.stat-card {
		text-align: center;
		padding: 0.75rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}

	.stat-label {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		margin-bottom: 0.15rem;
	}

	.stat-value {
		font-size: 1.3rem;
		font-family: var(--font-mono);
		font-weight: 700;
		color: var(--text);
	}

	.score-value { color: var(--green); }

	/* Iterations list */
	.iterations-section {
		margin-bottom: 1rem;
	}

	.iterations-list {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.iteration-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.4rem 0.6rem;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: 4px;
		font-size: 0.75rem;
		font-family: var(--font-mono);
		color: var(--text-muted);
	}

	.iteration-row.best {
		border-color: var(--accent);
		background: rgba(57, 211, 83, 0.05);
	}

	.iter-num {
		font-weight: 700;
		color: var(--text);
		min-width: 24px;
	}

	.iter-phase {
		color: var(--blue);
		min-width: 100px;
	}

	.iter-score {
		color: var(--green);
		font-weight: 600;
	}

	.iter-game, .iter-opt {
		color: var(--text-muted);
	}

	.iter-orders {
		color: var(--orange);
	}

	.iter-time {
		margin-left: auto;
		color: var(--text-muted);
	}

	/* Terminal */
	.terminal-section {
		padding: 0;
		overflow: hidden;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
	}

	.terminal-header {
		padding: 0.6rem 1rem;
		border-bottom: 1px solid var(--border);
		background: rgba(1, 4, 9, 0.5);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.terminal-header h3 {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--text-muted);
	}

	.btn-clear {
		font-size: 0.65rem;
		color: var(--text-muted);
		background: transparent;
		border: 1px solid var(--border);
		border-radius: 3px;
		padding: 0.15rem 0.5rem;
		cursor: pointer;
	}

	.btn-clear:hover {
		color: var(--text);
		border-color: var(--text-muted);
	}

	.terminal {
		height: 300px;
		overflow-y: auto;
		padding: 0.75rem 1rem;
		font-family: var(--font-mono);
		font-size: 0.75rem;
		line-height: 1.6;
		background: var(--bg);
		box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.6);
	}

	.terminal-line {
		white-space: pre-wrap;
		word-break: break-all;
	}

	.terminal-line.system { color: var(--blue); }
	.terminal-line.info { color: var(--text); }
	.terminal-line.success { color: var(--green); }
	.terminal-line.warn { color: var(--orange); }
	.terminal-line.error { color: var(--red); }
	.terminal-line.dim { color: var(--text-muted); }

	.cursor-line {
		animation: blink 1s infinite;
	}

	@keyframes blink {
		0%, 100% { opacity: 1; }
		50% { opacity: 0; }
	}

	/* Responsive */
	@media (max-width: 900px) {
		.bots-row { grid-template-columns: 1fr; }
		.solutions-row { grid-template-columns: repeat(2, 1fr); }
		.stats-row { grid-template-columns: repeat(2, 1fr); }
		.config-row { flex-direction: column; gap: 0.75rem; }
		.iteration-row { flex-wrap: wrap; }
	}
</style>
