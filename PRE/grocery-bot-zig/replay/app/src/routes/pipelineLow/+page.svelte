<script>
	const TARGETS = { easy: 150, medium: 225, hard: 260, expert: 310 };

	// ── Input State ──────────────────────────────────────────────────────────────
	let wsUrl          = $state('');
	let running        = $state(false);
	let phase          = $state('idle');
	let abortCtrl      = $state(null);

	// ── Iteration config ─────────────────────────────────────────────────────────
	let timeBudget     = $state(280);
	let postOptTime    = $state(60);
	let gpuOptTime     = $state(20);

	// ── Timer ────────────────────────────────────────────────────────────────────
	let timerStart     = $state(null);
	let timerNow       = $state(null);
	let timerInterval  = $state(null);

	let timerElapsed = $derived(timerStart && timerNow ? (timerNow - timerStart) / 1000 : 0);
	let timerRemaining = $derived(Math.max(0, timeBudget - timerElapsed));

	function startTimer() {
		timerStart = Date.now();
		timerNow = Date.now();
		if (timerInterval) clearInterval(timerInterval);
		timerInterval = setInterval(() => { timerNow = Date.now(); }, 250);
	}
	function stopTimer() {
		if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
	}

	// ── Score State ──────────────────────────────────────────────────────────────
	let bestScore      = $state(0);
	let currentScore   = $state(0);
	let detectedDiff   = $state(null);
	let iterCount      = $state(0);
	let currentPhase   = $state('');
	let lastStatus     = $state('');

	let targetScore = $derived(detectedDiff ? (TARGETS[detectedDiff] || 175) : 175);

	// ── Logs ─────────────────────────────────────────────────────────────────────
	let logs = $state([]);
	function addLog(text) {
		if (!text) return;
		const t = new Date().toLocaleTimeString();
		logs = [...logs.slice(-80), { t, text }];
	}

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
	function fmtSecs(s) {
		if (!s && s !== 0) return '--';
		const m = Math.floor(s / 60);
		const ss = Math.floor(s % 60);
		return `${m}:${ss.toString().padStart(2, '0')}`;
	}

	function resetAll() {
		bestScore = 0; currentScore = 0; detectedDiff = null;
		iterCount = 0; currentPhase = ''; lastStatus = ''; logs = [];
	}

	// ── Event Handler ────────────────────────────────────────────────────────────
	function handleEvent(event) {
		switch (event.type) {
			case 'init':
				if (event.difficulty) detectedDiff = event.difficulty;
				lastStatus = `Connected: ${event.difficulty} ${event.width}x${event.height}`;
				addLog(lastStatus);
				break;

			case 'round':
				currentScore = event.score || 0;
				if (event.round % 50 === 0) addLog(`R${event.round} score=${event.score}`);
				break;

			case 'game_over':
				currentScore = event.score || currentScore;
				lastStatus = `Game over: ${event.score}`;
				addLog(lastStatus);
				break;

			case 'iter_start':
				currentPhase = event.phase || 'playing';
				phase = event.phase === 'optimize_replay' ? 'optimizing' : 'playing';
				lastStatus = `Iter ${(event.iter||0)+1}: ${event.phase}`;
				addLog(`--- ${lastStatus} (${Math.floor(event.remaining||0)}s left) ---`);
				break;

			case 'optimize_phase_start':
				phase = 'optimizing';
				currentPhase = 'optimizing';
				lastStatus = `GPU optimize: ${event.difficulty} ${event.max_time}s`;
				addLog(lastStatus);
				break;

			case 'optimize_done':
				currentScore = Math.max(currentScore, event.score || 0);
				lastStatus = `Optimize: ${event.score} (was ${event.prev_score})`;
				addLog(lastStatus);
				break;

			case 'replay_phase_start':
				phase = 'replaying';
				currentPhase = 'replaying';
				lastStatus = 'Replaying solution...';
				addLog(lastStatus);
				break;

			case 'replay_phase_done':
				lastStatus = `Replay done: ${event.score}`;
				addLog(lastStatus);
				break;

			case 'iter_done':
				if (event.difficulty) detectedDiff = event.difficulty;
				currentScore = event.score || 0;
				if (currentScore > bestScore) bestScore = currentScore;
				iterCount = (event.iter || 0) + 1;
				lastStatus = `Iter ${iterCount}: score=${event.score}`;
				addLog(lastStatus);
				break;

			case 'iter_skip':
				addLog(`Iter ${(event.iter||0)+1} skipped: ${event.reason}`);
				break;

			case 'iter_summary':
				bestScore = event.best_score || bestScore;
				addLog(`Best: ${event.best_score} after ${event.iterations_done} iters`);
				break;

			case 'pipeline_start':
				addLog(`Pipeline: ${event.time_budget}s budget`);
				break;

			case 'pipeline_complete':
				bestScore = event.best_score || bestScore;
				iterCount = event.iterations || iterCount;
				phase = 'done';
				running = false;
				stopTimer();
				lastStatus = `Done! Best=${event.best_score} in ${event.iterations} iters`;
				addLog(lastStatus);
				break;

			case 'log':
				// Show WebSocket and important logs
				if (event.text) {
					const t = event.text;
					if (t.includes('WebSocket') || t.includes('ERROR') || t.includes('WARNING')
						|| t.includes('GAME_OVER') || t.includes('timeout') || t.includes('stall')
						|| t.includes('closed') || t.includes('connect') || t.includes('Score:')
						|| t.includes('[pipeline]') || t.includes('[capture]')
						|| t.includes('Connecting') || t.includes('Connected')
						|| t.includes('[zig]') || t.includes('Replaying')
						|| t.includes('DESYNC') || t.includes('GREEDY')
						|| t.includes('failed') || t.includes('error')) {
						addLog(t);
					}
				}
				break;

			case 'status':
				if (event.message) addLog(event.message);
				break;

			case 'error':
				lastStatus = `ERROR: ${event.message}`;
				addLog(lastStatus);
				phase = 'done';
				running = false;
				stopTimer();
				break;
		}
	}

	// ── SSE Stream ───────────────────────────────────────────────────────────────
	async function startIterate() {
		if (!wsUrl.trim()) return;
		if (abortCtrl) abortCtrl.abort();
		resetAll();
		running = true;
		phase = 'playing';
		abortCtrl = new AbortController();
		startTimer();

		try {
			const res = await fetch('/api/pipeline/iterate', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					url: wsUrl.trim(),
					timeBudget,
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
			if (e.name !== 'AbortError') {
				addLog(`Connection error: ${e.message}`);
			}
		}
		if (phase !== 'done') phase = 'done';
		running = false;
		stopTimer();
	}

	function stop() {
		if (abortCtrl) abortCtrl.abort();
		running = false;
		stopTimer();
		addLog('Stopped.');
	}

	let diffColors = {
		easy: '#39d353', medium: '#d29922', hard: '#f85149', expert: '#da3633',
	};

	let logBox = $state(null);
	$effect(() => {
		if (logBox && logs.length) {
			logBox.scrollTop = logBox.scrollHeight;
		}
	});
</script>

<div class="page">
	<h1>Pipeline Low</h1>

	<!-- Controls -->
	<div class="controls">
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

		<div class="params">
			<label>Budget: <input type="number" bind:value={timeBudget} min="60" max="600" step="10" class="num" disabled={running} />s</label>
			<label>Post-opt: <input type="number" bind:value={postOptTime} min="0" max="3600" step="10" class="num" disabled={running} />s</label>
			<label>GPU: <input type="number" bind:value={gpuOptTime} min="10" max="120" step="5" class="num" disabled={running} />s</label>
		</div>

		{#if running}
			<button class="btn stop" onclick={stop}>Stop</button>
		{:else}
			<button class="btn start" disabled={!wsUrl.trim()} onclick={startIterate}>Start</button>
		{/if}
	</div>

	<!-- Score display -->
	<div class="score-box">
		<div class="score-main">
			<span class="label">Best Score</span>
			<span class="value" class:hit={bestScore >= targetScore}>{bestScore}</span>
			{#if detectedDiff}
				<span class="diff" style="color:{diffColors[detectedDiff]||'#aaa'}">{detectedDiff}</span>
			{/if}
			{#if targetScore}
				<span class="target">/ {targetScore}</span>
			{/if}
		</div>

		{#if running}
			<div class="status-row">
				<span class="phase-pill"
					class:playing={phase === 'playing'}
					class:optimizing={phase === 'optimizing'}
					class:replaying={phase === 'replaying'}>
					{phase === 'playing' ? 'Playing' :
					 phase === 'optimizing' ? 'Optimizing' :
					 phase === 'replaying' ? 'Replaying' : phase}
				</span>
				<span class="iter-count">Iter {iterCount + 1}</span>
				<span class="current">Current: {currentScore}</span>
				<span class="timer">{fmtSecs(timerRemaining)} left</span>
			</div>
		{:else if phase === 'done'}
			<div class="status-row done">
				Done — {iterCount} iterations in {fmtSecs(timerElapsed)}
			</div>
		{/if}
	</div>

	<!-- Log area -->
	<div class="log-box" bind:this={logBox}>
		{#each logs as log}
			<div class="log-line" class:err={log.text.includes('ERROR') || log.text.includes('failed')}
				class:warn={log.text.includes('WARNING') || log.text.includes('timeout')}
				class:ws={log.text.includes('WebSocket') || log.text.includes('connect')}>
				<span class="log-time">{log.t}</span> {log.text}
			</div>
		{/each}
		{#if logs.length === 0 && phase !== 'idle'}
			<div class="log-line dim">Waiting for events...</div>
		{/if}
	</div>
</div>

<style>
	.page {
		max-width: 600px;
		margin: 40px auto;
		padding: 20px;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
		color: #e6edf3;
	}
	h1 {
		font-size: 1.4em;
		margin: 0 0 16px;
		color: #8b949e;
	}
	.controls {
		display: flex;
		flex-direction: column;
		gap: 8px;
		margin-bottom: 24px;
	}
	.url-input {
		width: 100%;
		padding: 8px 12px;
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 6px;
		color: #e6edf3;
		font-size: 13px;
		font-family: monospace;
	}
	.params {
		display: flex;
		gap: 12px;
		font-size: 13px;
		color: #8b949e;
	}
	.params label {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.num {
		width: 50px;
		padding: 4px 6px;
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 4px;
		color: #e6edf3;
		font-size: 13px;
		text-align: center;
	}
	.badge {
		display: inline-block;
		padding: 2px 8px;
		border-radius: 12px;
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		color: #fff;
	}
	.btn {
		padding: 8px 20px;
		border: none;
		border-radius: 6px;
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		align-self: flex-start;
	}
	.btn.start {
		background: #238636;
		color: #fff;
	}
	.btn.start:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}
	.btn.stop {
		background: #da3633;
		color: #fff;
	}

	.score-box {
		background: #161b22;
		border: 1px solid #30363d;
		border-radius: 8px;
		padding: 24px;
		margin-bottom: 16px;
	}
	.score-main {
		display: flex;
		align-items: baseline;
		gap: 12px;
	}
	.score-main .label {
		font-size: 14px;
		color: #8b949e;
	}
	.score-main .value {
		font-size: 48px;
		font-weight: 700;
		color: #e6edf3;
		font-variant-numeric: tabular-nums;
	}
	.score-main .value.hit {
		color: #39d353;
	}
	.score-main .diff {
		font-size: 14px;
		font-weight: 600;
		text-transform: uppercase;
	}
	.score-main .target {
		font-size: 20px;
		color: #484f58;
	}

	.status-row {
		margin-top: 12px;
		display: flex;
		align-items: center;
		gap: 12px;
		font-size: 13px;
		color: #8b949e;
	}
	.status-row.done {
		color: #39d353;
	}
	.phase-pill {
		padding: 2px 10px;
		border-radius: 12px;
		font-size: 12px;
		font-weight: 600;
		background: #30363d;
		color: #8b949e;
	}
	.phase-pill.playing { background: #1f3a2d; color: #39d353; }
	.phase-pill.optimizing { background: #3b2a0f; color: #d29922; }
	.phase-pill.replaying { background: #1c2d4f; color: #58a6ff; }
	.iter-count { font-weight: 600; }
	.current { font-variant-numeric: tabular-nums; }
	.timer { margin-left: auto; font-variant-numeric: tabular-nums; }

	/* Log area */
	.log-box {
		background: #0d1117;
		border: 1px solid #21262d;
		border-radius: 6px;
		padding: 8px;
		max-height: 240px;
		overflow-y: auto;
		font-family: 'Cascadia Code', 'Fira Code', monospace;
		font-size: 11px;
		line-height: 1.5;
	}
	.log-line {
		color: #8b949e;
		white-space: pre-wrap;
		word-break: break-all;
	}
	.log-line.err { color: #f85149; }
	.log-line.warn { color: #d29922; }
	.log-line.ws { color: #58a6ff; }
	.log-line.dim { color: #484f58; }
	.log-time {
		color: #484f58;
		margin-right: 4px;
	}
</style>
