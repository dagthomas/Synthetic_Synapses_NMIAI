<script>
	// ── Constants ─────────────────────────────────────────────────────────
	const PHASE_LABELS = {
		capture: 'Zig Capture',
		replay: 'Replay',
		capture_orders: 'Order Capture',
		optimize: 'GPU Optimize',
		deep_train: 'Deep Training',
	};
	const PHASE_COLORS = {
		capture: '#f85149',
		replay: '#58a6ff',
		capture_orders: '#d29922',
		optimize: '#39d353',
		deep_train: '#bc8cff',
	};

	// ── Input State ──────────────────────────────────────────────────────
	let wsUrl         = $state('');
	let difficulty    = $state('expert');
	let gpu           = $state('auto');
	let deepBudget    = $state(300);     // seconds for deep training per iteration
	let maxStates     = $state('');       // empty = auto

	// ── Remote GPU ───────────────────────────────────────────────────────
	let gpuConnected  = $state(false);
	let gpuInfo       = $state(null);
	let gpuChecking   = $state(false);

	async function checkGpu() {
		gpuChecking = true;
		try {
			const res = await fetch('/api/gpu-remote');
			gpuInfo = await res.json();
			gpuConnected = gpuInfo.connected;
			if (gpuConnected) {
				addLog(`GPU connected: ${gpuInfo.name} (${gpuInfo.vram_gb} GB)`);
			} else {
				addLog(`GPU not reachable: ${gpuInfo.error || 'connection refused'}`);
			}
		} catch (e) {
			gpuConnected = false;
			addLog(`GPU check failed: ${e.message}`);
		}
		gpuChecking = false;
	}

	// Check GPU on load
	$effect(() => { checkGpu(); });

	// ── Runtime State ────────────────────────────────────────────────────
	let running       = $state(false);
	let currentPhase  = $state(null);
	let abortCtrl     = $state(null);

	// ── Iteration tracking ───────────────────────────────────────────────
	let iterations    = $state([]);
	let totalOrders   = $state(0);
	let bestScore     = $state(0);

	// ── Current iteration live data ──────────────────────────────────────
	let liveScore     = $state(0);
	let liveRound     = $state(0);
	let liveMaxRound  = $state(0);

	// ── Logs ─────────────────────────────────────────────────────────────
	let logs          = $state([]);
	let showLogs      = $state(false);
	let logEl;

	// ── Timer ────────────────────────────────────────────────────────────
	let timerStart    = $state(null);
	let timerNow      = $state(null);
	let timerInterval = $state(null);
	let elapsed = $derived(timerStart && timerNow ? (timerNow - timerStart) / 1000 : 0);

	function startTimer() {
		timerStart = Date.now();
		timerNow = Date.now();
		if (timerInterval) clearInterval(timerInterval);
		timerInterval = setInterval(() => { timerNow = Date.now(); }, 500);
	}
	function stopTimer() {
		if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
	}

	// ── Deep budget presets ──────────────────────────────────────────────
	const DEEP_PRESETS = [
		{ label: '5 min', value: 300 },
		{ label: '15 min', value: 900 },
		{ label: '30 min', value: 1800 },
		{ label: '1 hour', value: 3600 },
		{ label: '2 hours', value: 7200 },
		{ label: '4 hours', value: 14400 },
	];

	// ── Helpers ──────────────────────────────────────────────────────────
	function addLog(text) {
		const ts = new Date().toLocaleTimeString('nb-NO', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
		logs = [...logs, { ts, text }];
		if (logEl) requestAnimationFrame(() => logEl.scrollTop = logEl.scrollHeight);
	}

	function fmtTime(s) {
		if (s < 60) return `${Math.floor(s)}s`;
		if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
		return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
	}

	// ── Run one iteration ────────────────────────────────────────────────
	// Step 1: Local capture/replay (uses token, Zig bot)
	// Step 2: Remote GPU optimize (via gpu_server.py on RunPod)
	async function runIteration() {
		if (!wsUrl.trim()) return;
		running = true;
		currentPhase = null;
		liveScore = 0;
		liveRound = 0;
		liveMaxRound = 0;
		startTimer();

		const iterNum = iterations.length;
		addLog(`--- Iteration ${iterNum + 1} starting ---`);

		const ctrl = new AbortController();
		abortCtrl = ctrl;

		try {
			// Step 1: Local capture via pipeline (Zig bot + order extraction)
			currentPhase = 'capture';
			addLog('Phase: Local capture (Zig bot)...');

			const captureRes = await fetch('/api/stepladder/run', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					url: wsUrl.trim(),
					difficulty,
					iteration: iterNum,
					gpu: 'local',  // only capture locally, no deep training
					deepBudget: 0, // skip deep training on local
					maxStates: maxStates ? parseInt(maxStates) : null,
				}),
				signal: ctrl.signal,
			});

			const reader = captureRes.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';
			let captureData = null;
			let gameScore = 0;

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buf += decoder.decode(value, { stream: true });
				const lines = buf.split('\n');
				buf = lines.pop() || '';

				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try {
						const evt = JSON.parse(line.slice(6));
						handleEvent(evt);
						if (evt.type === 'iter_done') {
							gameScore = evt.game_score || 0;
						}
						if (evt.type === 'capture_data') {
							captureData = evt.data;
						}
					} catch {}
				}
			}

			// Step 2: Remote GPU optimize (if connected)
			if (gpuConnected && captureData) {
				currentPhase = 'optimize';
				addLog(`Phase: Remote GPU optimize (${fmtTime(deepBudget)} budget)...`);

				const gpuRes = await fetch('/api/gpu-remote', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						deep: deepBudget > 120,
						capture: captureData,
						params: {
							difficulty,
							gpu,
							max_states: maxStates ? parseInt(maxStates) : 200000,
							max_time: Math.min(120, deepBudget),
							budget: deepBudget,
							refine_iters: 20,
							orderings: 3,
							speed_bonus: 50,
						},
					}),
					signal: ctrl.signal,
				});

				const result = await gpuRes.json();
				if (result.error) {
					addLog(`GPU error: ${result.error}`);
				} else {
					addLog(`GPU done: score=${result.score} in ${result.elapsed}s`);
					for (const evt of (result.events || [])) {
						handleEvent(evt);
					}

					// Save solution back to local DB
					if (result.score > 0 && result.actions) {
						try {
							await fetch('/api/optimize/solutions', {
								method: 'POST',
								headers: { 'Content-Type': 'application/json' },
								body: JSON.stringify({
									difficulty,
									score: result.score,
									actions: result.actions,
								}),
							});
							addLog(`Solution saved: score=${result.score}`);
						} catch (e) {
							addLog(`Failed to save solution: ${e.message}`);
						}
					}

					if (result.score > bestScore) bestScore = result.score;
					iterations = [...iterations, {
						num: iterNum + 1,
						gameScore,
						optScore: result.score,
						deepScore: result.score,
						bestScore: Math.max(gameScore, result.score),
						orders: totalOrders,
						newOrders: 0,
						elapsed: elapsed,
					}];
				}
			} else if (!gpuConnected) {
				addLog('GPU not connected — skipping remote optimize. Start SSH tunnel + gpu_server.py');
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				addLog(`Error: ${e.message}`);
			}
		}

		running = false;
		stopTimer();
		wsUrl = '';  // clear token (used up)
		addLog(`--- Iteration ${iterNum + 1} complete ---`);
	}

	function stopIteration() {
		if (abortCtrl) abortCtrl.abort();
		running = false;
		stopTimer();
	}

	// ── Event handler ────────────────────────────────────────────────────
	function handleEvent(evt) {
		switch (evt.type) {
			case 'phase_start':
				currentPhase = evt.phase;
				addLog(`Phase: ${PHASE_LABELS[evt.phase] || evt.phase}${evt.budget ? ` (${fmtTime(evt.budget)})` : ''}`);
				break;

			case 'phase_done':
				addLog(`${PHASE_LABELS[evt.phase] || evt.phase} done: score=${evt.score || evt.captured || 0} (${fmtTime(evt.elapsed)})`);
				if (evt.phase === 'capture_orders' && evt.captured) {
					addLog(`  Captured ${evt.captured} orders from game log`);
				}
				break;

			case 'round':
				liveRound = evt.round;
				liveMaxRound = evt.max || liveMaxRound;
				liveScore = evt.score;
				break;

			case 'orders_update':
				totalOrders = evt.after;
				if (evt.new_orders > 0) {
					addLog(`Orders: ${evt.before} -> ${evt.after} (+${evt.new_orders} new)`);
				} else {
					addLog(`Orders: ${evt.after} (no new)`);
				}
				break;

			case 'deep_improvement':
				addLog(`Deep: NEW BEST ${evt.score}`);
				break;

			case 'deep_phase':
				addLog(`Deep Phase ${evt.phase}: ${evt.name} (${fmtTime(evt.budget)})`);
				break;

			case 'iter_start':
				totalOrders = evt.orders || totalOrders;
				bestScore = evt.prev_score || bestScore;
				break;

			case 'iter_done':
				if (evt.best_score > bestScore) bestScore = evt.best_score;
				totalOrders = evt.orders || totalOrders;
				iterations = [...iterations, {
					num: evt.iteration + 1,
					gameScore: evt.game_score,
					optScore: evt.opt_score,
					deepScore: evt.deep_score,
					bestScore: evt.best_score,
					orders: evt.orders,
					newOrders: evt.new_orders,
					elapsed: evt.elapsed,
				}];
				addLog(`Iteration done: game=${evt.game_score} opt=${evt.opt_score} deep=${evt.deep_score} best=${evt.best_score}`);
				break;

			case 'optimize_start':
			case 'optimize_done':
			case 'gpu_bot_done':
			case 'gpu_phase':
				// Forward GPU events to log
				if (evt.type === 'gpu_bot_done') {
					addLog(`Bot ${evt.bot}/${evt.num_bots}: score=${evt.score} (${evt.elapsed}s)`);
				} else if (evt.type === 'optimize_done') {
					addLog(`Optimize done: score=${evt.score}`);
				}
				break;

			case 'log':
				addLog(evt.text);
				break;

			case 'error':
				addLog(`ERROR: ${evt.message}`);
				break;
		}
	}
</script>

<div class="page">
	<div class="header">
		<h1>Stepladder</h1>
		<p class="subtitle">
			Multi-iteration order discovery + deep GPU optimization.
			Each iteration: replay/capture orders, GPU optimize, deep training.
			Provide a fresh token for each iteration.
		</p>
	</div>

	<!-- How it works -->
	<details class="how-it-works">
		<summary>Hvordan fungerer Stepladder?</summary>
		<div class="explanation">
			<p>Stepladder utnytter at <strong>samme dag = same map + orders</strong>. Hver iterasjon oppdager nye orders som gjor neste iterasjon bedre:</p>
			<ol>
				<li><strong>Capture/Replay</strong> — Spiller en runde (Zig-bot eller replayer best solution). Oppdager nye orders fra spillet.</li>
				<li><strong>GPU Optimize</strong> — Rask optimalisering med alle kjente orders (60s).</li>
				<li><strong>Deep Training</strong> — Lang offline optimalisering (minutter til timer). Tre faser:
					<ul>
						<li><em>Exploration</em> (30%) — Tester mange orderings, finn gode kandidater</li>
						<li><em>Intensification</em> (50%) — Dyp refinement med joint multi-bot DP</li>
						<li><em>LNS</em> (20%) — Destroy-repair cycles for a unnslippe lokale optima</li>
					</ul>
				</li>
			</ol>
			<p>Med <strong>B200</strong> (192 GB): 200K states, 4-bot joint DP, 200 orderings. Langt kraftigere enn 5090.</p>
			<p><strong>Typisk workflow:</strong> Kjor 5-10 iterasjoner over flere timer. Hver iterasjon bruker ett token (~5 min live-spill) + lang offline deep training.</p>
		</div>
	</details>

	<!-- GPU Status -->
	<div class="gpu-status" class:connected={gpuConnected} class:disconnected={!gpuConnected}>
		<span class="gpu-dot"></span>
		{#if gpuChecking}
			<span>Checking GPU...</span>
		{:else if gpuConnected}
			<span>GPU: {gpuInfo?.name} ({gpuInfo?.vram_gb} GB)</span>
		{:else}
			<span>GPU not connected</span>
		{/if}
		<button class="btn-small" onclick={checkGpu} disabled={gpuChecking}>Refresh</button>
	</div>

	<!-- Controls -->
	<div class="controls">
		<div class="input-row">
			<label>
				<span class="label">WSS Token URL</span>
				<input type="text" bind:value={wsUrl} placeholder="wss://game.ainm.no/ws?token=..."
					disabled={running} class="url-input" />
			</label>
		</div>

		<div class="input-row compact">
			<label>
				<span class="label">Difficulty</span>
				<select bind:value={difficulty} disabled={running}>
					<option value="easy">Easy</option>
					<option value="medium">Medium</option>
					<option value="hard">Hard</option>
					<option value="expert">Expert</option>
				</select>
			</label>

			<label>
				<span class="label">GPU</span>
				<select bind:value={gpu} disabled={running}>
					<option value="auto">Auto-detect</option>
					<option value="b200">B200 (192 GB)</option>
					<option value="5090">RTX 5090 (32 GB)</option>
				</select>
			</label>

			<label>
				<span class="label">Deep Budget</span>
				<select bind:value={deepBudget} disabled={running}>
					{#each DEEP_PRESETS as p}
						<option value={p.value}>{p.label}</option>
					{/each}
				</select>
			</label>

			<label>
				<span class="label">Max States</span>
				<input type="text" bind:value={maxStates} placeholder="auto" disabled={running} class="states-input" />
			</label>
		</div>

		<div class="action-row">
			{#if running}
				<button class="btn btn-stop" onclick={stopIteration}>Stop</button>
				<div class="running-status">
					{#if currentPhase}
						<span class="phase-badge" style="background: {PHASE_COLORS[currentPhase] || '#8b949e'}">
							{PHASE_LABELS[currentPhase] || currentPhase}
						</span>
					{/if}
					{#if liveRound > 0}
						<span class="round-info">R{liveRound}/{liveMaxRound} Score:{liveScore}</span>
					{/if}
					<span class="timer">{fmtTime(elapsed)}</span>
				</div>
			{:else}
				<button class="btn btn-run" onclick={runIteration} disabled={!wsUrl.trim()}>
					Run Iteration {iterations.length + 1}
				</button>
			{/if}
		</div>
	</div>

	<!-- Stats bar -->
	<div class="stats-bar">
		<div class="stat">
			<span class="stat-label">Iterations</span>
			<span class="stat-value">{iterations.length}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Orders</span>
			<span class="stat-value accent">{totalOrders}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Best Score</span>
			<span class="stat-value gold">{bestScore}</span>
		</div>
		<div class="stat">
			<span class="stat-label">GPU</span>
			<span class="stat-value">{gpu}</span>
		</div>
		<div class="stat">
			<span class="stat-label">Deep Budget</span>
			<span class="stat-value">{fmtTime(deepBudget)}</span>
		</div>
	</div>

	<!-- Iteration history -->
	{#if iterations.length > 0}
		<div class="section">
			<h2>Iterations</h2>
			<table class="iter-table">
				<thead>
					<tr>
						<th>#</th>
						<th>Game</th>
						<th>GPU Opt</th>
						<th>Deep</th>
						<th>Best</th>
						<th>Orders</th>
						<th>New</th>
						<th>Time</th>
					</tr>
				</thead>
				<tbody>
					{#each iterations as iter}
						<tr>
							<td>{iter.num}</td>
							<td>{iter.gameScore}</td>
							<td>{iter.optScore}</td>
							<td class={iter.deepScore > iter.optScore ? 'highlight' : ''}>{iter.deepScore}</td>
							<td class="best-col">{iter.bestScore}</td>
							<td>{iter.orders}</td>
							<td class={iter.newOrders > 0 ? 'new-orders' : ''}>{iter.newOrders > 0 ? `+${iter.newOrders}` : '0'}</td>
							<td>{fmtTime(iter.elapsed)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}

	<!-- Score chart -->
	{#if iterations.length > 1}
		<div class="section">
			<h2>Score Progress</h2>
			<div class="chart">
				{@const maxScore = Math.max(...iterations.map(i => i.bestScore), 1)}
				{@const barW = Math.min(60, Math.floor(600 / iterations.length))}
				<svg width={iterations.length * (barW + 4) + 40} height="140" viewBox="0 0 {iterations.length * (barW + 4) + 40} 140">
					{#each iterations as iter, idx}
						{@const h = (iter.bestScore / maxScore) * 110}
						<rect x={idx * (barW + 4) + 30} y={120 - h} width={barW} height={h}
							fill="#39d353" opacity="0.8" rx="2" />
						<text x={idx * (barW + 4) + 30 + barW/2} y={115 - h} fill="#e6edf3" font-size="10"
							text-anchor="middle">{iter.bestScore}</text>
						<text x={idx * (barW + 4) + 30 + barW/2} y={135} fill="#8b949e" font-size="9"
							text-anchor="middle">#{iter.num}</text>
					{/each}
				</svg>
			</div>
		</div>
	{/if}

	<!-- Logs -->
	<div class="section">
		<h2 class="logs-header" onclick={() => showLogs = !showLogs}>
			Logs {showLogs ? '▼' : '▶'} <span class="log-count">({logs.length})</span>
		</h2>
		{#if showLogs}
			<div class="log-area" bind:this={logEl}>
				{#each logs as log}
					<div class="log-line">
						<span class="log-ts">{log.ts}</span>
						<span class="log-text">{log.text}</span>
					</div>
				{/each}
			</div>
		{/if}
	</div>
</div>

<style>
	.page { max-width: 900px; margin: 0 auto; }

	/* GPU Status */
	.gpu-status {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.5rem 0.8rem;
		border-radius: 6px;
		font-size: 0.8rem;
		margin-bottom: 0.75rem;
		border: 1px solid var(--border);
	}
	.gpu-status.connected {
		background: rgba(57, 211, 83, 0.08);
		border-color: rgba(57, 211, 83, 0.3);
		color: var(--accent);
	}
	.gpu-status.disconnected {
		background: rgba(248, 81, 73, 0.08);
		border-color: rgba(248, 81, 73, 0.3);
		color: #f85149;
	}
	.gpu-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		flex-shrink: 0;
	}
	.connected .gpu-dot { background: var(--accent); }
	.disconnected .gpu-dot { background: #f85149; }
	.btn-small {
		padding: 0.15rem 0.5rem;
		font-size: 0.7rem;
		background: rgba(255,255,255,0.05);
		border: 1px solid var(--border);
		border-radius: 3px;
		color: var(--text-muted);
		cursor: pointer;
		margin-left: auto;
	}

	.header h1 {
		font-family: var(--font-display);
		font-size: 1.5rem;
		color: var(--accent);
		margin-bottom: 0.25rem;
	}
	.subtitle {
		color: var(--text-muted);
		font-size: 0.8rem;
		line-height: 1.4;
		margin-bottom: 1rem;
	}

	/* How it works */
	.how-it-works {
		background: rgba(30, 40, 55, 0.5);
		border: 1px solid var(--border);
		border-radius: 6px;
		margin-bottom: 1rem;
		padding: 0;
	}
	.how-it-works summary {
		padding: 0.6rem 1rem;
		cursor: pointer;
		color: var(--accent);
		font-size: 0.85rem;
		font-weight: 600;
	}
	.explanation {
		padding: 0 1rem 1rem;
		color: var(--text-muted);
		font-size: 0.78rem;
		line-height: 1.6;
	}
	.explanation ol, .explanation ul { padding-left: 1.2rem; }
	.explanation li { margin-bottom: 0.3rem; }
	.explanation strong { color: var(--text); }
	.explanation em { color: var(--accent); font-style: normal; }

	/* Controls */
	.controls {
		background: rgba(13, 17, 23, 0.7);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 1rem;
		margin-bottom: 1rem;
	}
	.input-row { margin-bottom: 0.75rem; }
	.input-row.compact {
		display: flex;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.input-row label {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.label {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.url-input {
		width: 100%;
		padding: 0.5rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 0.8rem;
	}
	.states-input {
		width: 100px;
		padding: 0.4rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 0.8rem;
	}
	select {
		padding: 0.4rem;
		background: var(--bg-secondary);
		border: 1px solid var(--border);
		border-radius: 4px;
		color: var(--text);
		font-size: 0.8rem;
	}

	.action-row {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	.btn {
		padding: 0.5rem 1.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		font-family: var(--font-display);
		font-size: 0.85rem;
		font-weight: 600;
		cursor: pointer;
		transition: all 0.2s;
	}
	.btn-run {
		background: rgba(57, 211, 83, 0.15);
		border-color: var(--accent);
		color: var(--accent);
	}
	.btn-run:hover:not(:disabled) {
		background: rgba(57, 211, 83, 0.25);
	}
	.btn-run:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-stop {
		background: rgba(248, 81, 73, 0.15);
		border-color: #f85149;
		color: #f85149;
	}
	.btn-stop:hover { background: rgba(248, 81, 73, 0.25); }

	.running-status {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		font-size: 0.8rem;
	}
	.phase-badge {
		padding: 0.2rem 0.6rem;
		border-radius: 3px;
		font-size: 0.75rem;
		font-weight: 600;
		color: #0d1117;
	}
	.round-info { color: var(--text-muted); font-family: var(--font-mono); }
	.timer { color: var(--accent); font-family: var(--font-mono); font-weight: 600; }

	/* Stats bar */
	.stats-bar {
		display: flex;
		gap: 0;
		margin-bottom: 1rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		overflow: hidden;
	}
	.stat {
		flex: 1;
		padding: 0.6rem 0.8rem;
		background: rgba(13, 17, 23, 0.5);
		border-right: 1px solid var(--border);
		text-align: center;
	}
	.stat:last-child { border-right: none; }
	.stat-label {
		display: block;
		font-size: 0.65rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
		margin-bottom: 0.15rem;
	}
	.stat-value {
		font-family: var(--font-display);
		font-size: 1.1rem;
		font-weight: 700;
		color: var(--text);
	}
	.stat-value.accent { color: var(--accent); }
	.stat-value.gold { color: #d29922; }

	/* Section */
	.section {
		margin-bottom: 1rem;
	}
	.section h2 {
		font-size: 0.85rem;
		color: var(--text-muted);
		margin-bottom: 0.5rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	/* Iteration table */
	.iter-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 0.8rem;
		font-family: var(--font-mono);
	}
	.iter-table th {
		text-align: left;
		padding: 0.4rem 0.6rem;
		border-bottom: 1px solid var(--border);
		color: var(--text-muted);
		font-size: 0.7rem;
		text-transform: uppercase;
	}
	.iter-table td {
		padding: 0.4rem 0.6rem;
		border-bottom: 1px solid rgba(48, 54, 61, 0.3);
		color: var(--text);
	}
	.best-col { color: #d29922; font-weight: 600; }
	.highlight { color: #bc8cff; font-weight: 600; }
	.new-orders { color: var(--accent); font-weight: 600; }

	/* Chart */
	.chart {
		background: rgba(13, 17, 23, 0.5);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.5rem;
		overflow-x: auto;
	}

	/* Logs */
	.logs-header {
		cursor: pointer;
		user-select: none;
	}
	.log-count { color: var(--text-muted); font-weight: 400; }
	.log-area {
		max-height: 400px;
		overflow-y: auto;
		background: rgba(13, 17, 23, 0.8);
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 0.5rem;
		font-family: var(--font-mono);
		font-size: 0.72rem;
	}
	.log-line {
		display: flex;
		gap: 0.5rem;
		padding: 0.1rem 0;
		line-height: 1.4;
	}
	.log-ts { color: var(--text-muted); flex-shrink: 0; }
	.log-text { color: var(--text); word-break: break-all; }
</style>
