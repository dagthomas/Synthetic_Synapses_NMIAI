<script>
	import { onMount, onDestroy } from 'svelte';

	// === STATE ===
	let status = $state('idle'); // idle, solving, complete, error
	let selectedDifficulty = $state('easy');
	let gpuName = $state('');
	let vramTotal = $state(0);

	// Init info
	let initInfo = $state(null);

	// Multi-bot tracking
	let currentBotId = $state(-1);
	let totalBots = $state(1);
	let botScores = $state([]); // score per completed bot

	// Round-by-round data for graphs
	let roundData = $state([]); // [{r, score, unique, expanded, time}]
	let maxUnique = $state(0);
	let maxExpanded = $state(0);

	// Final result
	let finalScore = $state(0);
	let finalTime = $state(0);
	let isOptimal = $state(false);
	let prevBest = $state(0);

	// Solutions metadata
	let solutions = $state({easy: null, medium: null, hard: null, expert: null});

	// Terminal lines
	let terminalLines = $state([]);
	let terminalEl = $state(null);

	// Graph canvas
	let scoreCanvas = $state(null);
	let stateCanvas = $state(null);

	// WebSocket URL for live games
	let wsUrl = $state('');
	let tokenInfo = $state(null); // parsed JWT payload

	function parseToken(url) {
		try {
			const match = url.match(/token=([^&]+)/);
			if (!match) return null;
			const parts = match[1].split('.');
			if (parts.length !== 3) return null;
			const payload = JSON.parse(atob(parts[1].replace(/-/g, '+').replace(/_/g, '/')));
			return payload;
		} catch (e) {
			return null;
		}
	}

	// Auto-detect difficulty from pasted URL
	$effect(() => {
		const info = parseToken(wsUrl);
		tokenInfo = info;
		if (info?.difficulty) {
			selectedDifficulty = info.difficulty;
		}
	});

	// SSE reader
	let abortController = null;

	function addTerminal(text, type = 'info') {
		terminalLines.push({ text, type, ts: Date.now() });
		if (terminalLines.length > 100) terminalLines = terminalLines.slice(-80);
		requestAnimationFrame(() => {
			if (terminalEl) terminalEl.scrollTop = terminalEl.scrollHeight;
		});
	}

	// === GRAPH DRAWING ===
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

		// Grid lines
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
		for (let r = 0; r <= maxR; r += 50) {
			const x = pad.l + (r / maxR) * gW;
			ctx.beginPath();
			ctx.moveTo(x, pad.t);
			ctx.lineTo(x, pad.t + gH);
			ctx.stroke();
			ctx.fillStyle = '#8b949e';
			ctx.font = '10px JetBrains Mono, monospace';
			ctx.textAlign = 'center';
			ctx.fillText(String(r), x, pad.t + gH + 15);
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

			// Data points
			for (let i = 0; i < roundData.length; i++) {
				const d = roundData[i];
				const x = pad.l + (d.r / maxR) * gW;
				const y = pad.t + gH - (d.score / maxScore) * gH;
				ctx.fillStyle = d.score > 0 ? '#39d353' : 'rgba(57, 211, 83, 0.3)';
				ctx.beginPath();
				ctx.arc(x, y, 2, 0, Math.PI * 2);
				ctx.fill();
			}
		}

		// Previous best line
		if (prevBest > 0) {
			const y = pad.t + gH - (prevBest / maxScore) * gH;
			ctx.strokeStyle = 'rgba(57, 211, 83, 0.6)';
			ctx.setLineDash([5, 5]);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(pad.l, y);
			ctx.lineTo(pad.l + gW, y);
			ctx.stroke();
			ctx.setLineDash([]);
			ctx.fillStyle = '#39d353';
			ctx.font = '10px JetBrains Mono, monospace';
			ctx.textAlign = 'left';
			ctx.fillText(`prev: ${prevBest}`, pad.l + gW - 60, y - 5);
		}
	}

	function drawStateGraph() {
		if (!stateCanvas || roundData.length === 0) return;
		const canvas = stateCanvas;
		const ctx = canvas.getContext('2d');
		const W = canvas.width;
		const H = canvas.height;
		const pad = { t: 20, r: 15, b: 30, l: 60 };
		const gW = W - pad.l - pad.r;
		const gH = H - pad.t - pad.b;

		ctx.clearRect(0, 0, W, H);
		ctx.fillStyle = '#0d1117';
		ctx.fillRect(0, 0, W, H);

		const maxU = Math.max(maxUnique, 100);
		const maxR = 300;

		// Grid
		ctx.strokeStyle = 'rgba(48, 54, 61, 0.6)';
		ctx.lineWidth = 1;
		const steps = [1000, 5000, 10000, 50000, 100000, 200000, 500000];
		for (const s of steps) {
			if (s > maxU * 1.2) continue;
			const y = pad.t + gH - (s / maxU) * gH;
			if (y < pad.t) continue;
			ctx.beginPath();
			ctx.moveTo(pad.l, y);
			ctx.lineTo(pad.l + gW, y);
			ctx.stroke();
			ctx.fillStyle = '#8b949e';
			ctx.font = '10px JetBrains Mono, monospace';
			ctx.textAlign = 'right';
			ctx.fillText(s >= 1000 ? `${(s/1000).toFixed(0)}K` : String(s), pad.l - 5, y + 3);
		}

		// Area fill
		if (roundData.length > 1) {
			ctx.beginPath();
			ctx.moveTo(pad.l + (roundData[0].r / maxR) * gW, pad.t + gH);
			for (const d of roundData) {
				const x = pad.l + (d.r / maxR) * gW;
				const y = pad.t + gH - (d.unique / maxU) * gH;
				ctx.lineTo(x, y);
			}
			const lastD = roundData[roundData.length - 1];
			ctx.lineTo(pad.l + (lastD.r / maxR) * gW, pad.t + gH);
			ctx.closePath();

			const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + gH);
			grad.addColorStop(0, 'rgba(88, 166, 255, 0.15)');
			grad.addColorStop(1, 'rgba(88, 166, 255, 0.02)');
			ctx.fillStyle = grad;
			ctx.fill();

			// Line
			ctx.strokeStyle = '#58a6ff';
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			for (let i = 0; i < roundData.length; i++) {
				const d = roundData[i];
				const x = pad.l + (d.r / maxR) * gW;
				const y = pad.t + gH - (d.unique / maxU) * gH;
				if (i === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.stroke();
		}

		// Label
		ctx.fillStyle = '#8b949e';
		ctx.font = '10px JetBrains Mono, monospace';
		ctx.textAlign = 'center';
		ctx.fillText('ROUND', pad.l + gW / 2, pad.t + gH + 15);
	}

	// === SOLVE ===
	async function startSolve() {
		if (status === 'solving') return;

		status = 'solving';
		roundData = [];
		maxUnique = 0;
		maxExpanded = 0;
		finalScore = 0;
		finalTime = 0;
		isOptimal = false;
		terminalLines = [];
		currentBotId = -1;
		totalBots = 1;
		botScores = [];

		addTerminal(`[SYSTEM] Initiating parallel solve: ${selectedDifficulty.toUpperCase()}`, 'system');
		addTerminal(`[SYSTEM] Target: Best solution via parallel optimizer with SA`, 'system');

		try {
			abortController = new AbortController();
			const res = await fetch('/api/gpu/solve', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ difficulty: selectedDifficulty }),
				signal: abortController.signal,
			});

			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try {
						const data = JSON.parse(line.slice(6));
						handleEvent(data);
					} catch (e) {}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				addTerminal(`[ERROR] ${e.message}`, 'error');
				status = 'error';
			}
		}
	}

	function stopSolve() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		status = 'idle';
		addTerminal('[SYSTEM] Solve aborted by user', 'warn');
	}

	// === CAPTURE & SOLVE (Python capture → Parallel solve) ===
	async function startCapture() {
		if (!wsUrl.trim()) {
			addTerminal('[ERROR] Paste a WebSocket URL first', 'error');
			return;
		}
		status = 'solving';
		terminalLines = [];
		roundData = [];
		finalScore = 0;
		addTerminal(`[PIPELINE] Python capture > Parallel solve`, 'system');
		addTerminal(`[PIPELINE] URL: ${wsUrl.slice(0, 60)}...`, 'dim');

		let currentPhase = 'capture';
		let captureScore = 0;
		let lastLoggedRound = -1;

		try {
			abortController = new AbortController();
			const res = await fetch('/api/optimize/play', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ url: wsUrl, difficulty: selectedDifficulty }),
				signal: abortController.signal,
			});
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n\n');
				buffer = lines.pop() || '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try {
						const data = JSON.parse(line.slice(6));

						if (data.type === 'phase') {
							currentPhase = data.phase;
							addTerminal(`[PHASE] ${data.phase.toUpperCase()}`, 'system');
						} else if (data.type === 'progress') {
							captureScore = data.score || 0;
							finalScore = captureScore;
							if (data.round % 10 === 0 || data.round >= 295 || data.round <= 2) {
								if (data.round !== lastLoggedRound) {
									addTerminal(`[R${data.round}/${data.max_rounds}] Score: ${data.score}`, 'info');
									lastLoggedRound = data.round;
								}
							}
						} else if (data.type === 'final_score') {
							captureScore = data.score;
							finalScore = captureScore;
							addTerminal(`[CAPTURE] Probe game finished: score=${data.score}`, 'success');
						} else if (data.type === 'capture_done') {
							addTerminal(`[CAPTURE] Game complete (score ${data.score})`, 'success');
						} else if (data.type === 'extract_done') {
							addTerminal(`[EXTRACT] Captured ${data.orders} orders, ${data.items} items (${data.grid})`, 'success');
						} else if (data.type === 'solver_init') {
							totalBots = data.num_bots || 1;
							addTerminal(`[SOLVER] ${data.difficulty} | ${data.num_workers} workers | ${data.time_limit}s budget`, 'system');
							addTerminal(`[SOLVER] ${data.num_orders} orders, ${data.num_items} items, probe=${data.probe_score}`, 'info');
						} else if (data.type === 'solver_prev_best') {
							if (data.score > 0) addTerminal(`[SOLVER] Previous best: ${data.score}`, 'info');
						} else if (data.type === 'solver_solving') {
							addTerminal(`[SOLVER] ${data.msg}`, 'system');
						} else if (data.type === 'solver_result') {
							finalScore = data.score;
							finalTime = data.time;
							addTerminal(`[SOLVER] Result: score=${data.score} in ${data.time}s`, 'success');
						} else if (data.type === 'solver_improved') {
							addTerminal(`[SOLVER] IMPROVED: ${data.old_score} > ${data.new_score} (+${data.delta})`, 'success');
						} else if (data.type === 'solver_no_improvement') {
							addTerminal(`[SOLVER] No improvement (${data.score} vs prev ${data.prev})`, 'info');
						} else if (data.type === 'solver_error') {
							addTerminal(`[SOLVER ERROR] ${data.msg}`, 'error');
						} else if (data.type === 'solver_done') {
							finalScore = data.score;
							finalTime = data.time;
							addTerminal(`[SOLVER] Complete: score=${data.score} in ${data.time}s`, 'success');
						} else if (data.type === 'done') {
							addTerminal(`[DONE] Pipeline complete! Capture=${captureScore}, Solver=${finalScore}`, 'success');
							status = 'complete';
							loadSolutions();
						} else if (data.type === 'log') {
							if (!data.text.includes('Connecting to') && !data.text.includes('Logging to')) {
								addTerminal(`[LOG] ${data.text}`, 'dim');
							}
						} else if (data.type === 'error') {
							addTerminal(`[ERROR] ${data.message || data.msg}`, 'error');
							status = 'error';
						} else if (data.type === 'status') {
							addTerminal(`[STATUS] ${data.message}`, 'system');
						}
					} catch (e) {}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				addTerminal(`[ERROR] ${e.message}`, 'error');
				status = 'error';
			}
		}
		if (status === 'solving') status = 'idle';
	}

	// === REPLAY (replay best solution on live server) ===
	async function startReplay() {
		if (!wsUrl.trim()) {
			addTerminal('[ERROR] Paste a WebSocket URL first', 'error');
			return;
		}
		status = 'solving';
		terminalLines = [];
		addTerminal(`[REPLAY] Replaying optimal solution...`, 'system');

		try {
			abortController = new AbortController();
			const res = await fetch('/api/optimize/replay', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ url: wsUrl, difficulty: selectedDifficulty }),
				signal: abortController.signal,
			});
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n\n');
				buffer = lines.pop() || '';
				for (const line of lines) {
					if (!line.startsWith('data: ')) continue;
					try {
						const data = JSON.parse(line.slice(6));
						if (data.type === 'progress') {
							addTerminal(`[R${data.round}] Score: ${data.score}`, 'info');
							finalScore = data.score || 0;
						} else if (data.type === 'final_score') {
							finalScore = data.score;
							addTerminal(`[REPLAY] Final score: ${data.score}`, 'success');
						} else if (data.type === 'done') {
							addTerminal(`[REPLAY] Complete.`, 'success');
							status = 'complete';
							loadSolutions();
						} else if (data.type === 'log') {
							addTerminal(`[LOG] ${data.text}`, 'dim');
						} else if (data.type === 'error') {
							addTerminal(`[ERROR] ${data.message || data.msg}`, 'error');
							status = 'error';
						}
					} catch (e) {}
				}
			}
		} catch (e) {
			if (e.name !== 'AbortError') {
				addTerminal(`[ERROR] ${e.message}`, 'error');
				status = 'error';
			}
		}
		if (status === 'solving') status = 'idle';
	}

	// === LIVE SOLVE (capture + parallel solve in single pipeline) ===
	async function startLiveSolve() {
		await startCapture();
	}

	function handleEvent(data) {
		switch (data.type) {
			case 'init':
				initInfo = data;
				gpuName = data.gpu_name || '';
				vramTotal = data.vram_total || 0;
				totalBots = data.num_bots || 1;
				currentBotId = -1;
				botScores = [];
				addTerminal(`[GPU] ${data.gpu_name} | ${data.vram_total}GB VRAM`, 'system');
				addTerminal(`[MAP] ${data.width}x${data.height} | ${data.items} items | ${data.types} types | ${data.num_bots || 1} bots`, 'info');
				addTerminal(`[ORDERS] ${data.orders} total orders pre-generated`, 'info');
				break;

			case 'bot_start':
				currentBotId = data.bot_id;
				totalBots = data.total_bots;
				addTerminal(``, 'info');
				addTerminal(`[BOT ${data.bot_id}/${data.total_bots}] Starting DP solve...`, 'system');
				break;

			case 'bot_done':
				botScores = [...botScores, { bot_id: data.bot_id, score: data.score }];
				addTerminal(`[BOT ${data.bot_id}/${data.total_bots}] Done: score=${data.score} (${data.time}s)`, 'success');
				break;

			case 'gpu_ready':
				addTerminal(`[GPU] Searcher initialized in ${data.time}s | ${data.actions_per_state} actions/state`, 'success');
				break;

			case 'verify':
				if (data.ok) {
					addTerminal(`[VERIFY] GPU step matches CPU step`, 'success');
				} else {
					addTerminal(`[VERIFY] MISMATCH - GPU step diverges from CPU!`, 'error');
				}
				break;

			case 'prev_best':
				prevBest = data.score;
				if (data.score > 0) {
					addTerminal(`[PREV] Previous best: ${data.score}`, 'info');
				}
				break;

			case 'solving':
				addTerminal(`[SOLVE] ${data.msg}`, 'system');
				break;

			case 'round':
				roundData = [...roundData, {
					r: data.r, score: data.score,
					unique: data.unique, expanded: data.expanded,
					time: data.time, bot_id: data.bot_id,
				}];
				if (data.unique > maxUnique) maxUnique = data.unique;
				if (data.expanded > maxExpanded) maxExpanded = data.expanded;
				finalScore = data.score;
				finalTime = data.time;

				{
					const botPrefix = totalBots > 1 ? `B${data.bot_id ?? 0} ` : '';
					if (data.r < 10 || data.r % 25 === 0 || data.r === 299) {
						addTerminal(
							`[${botPrefix}R${String(data.r).padStart(3)}] score=${String(data.score).padStart(3)} ` +
							`unique=${String(data.unique).toLocaleString().padStart(7)} ` +
							`expanded=${String(data.expanded).toLocaleString().padStart(9)} ` +
							`t=${data.time.toFixed(1)}s`,
							data.score > prevBest ? 'success' : 'info'
						);
					}
				}

				drawScoreGraph();
				drawStateGraph();
				break;

			case 'result':
				finalScore = data.score;
				finalTime = data.time;
				isOptimal = data.optimal;
				totalBots = data.num_bots || totalBots;
				addTerminal('', 'info');
				addTerminal(`[RESULT] Score: ${data.score} | Time: ${data.time}s | Bots: ${data.num_bots || 1} | Optimal: ${data.optimal ? 'YES' : 'NO'}`, 'success');
				break;

			case 'improved':
				addTerminal(`[IMPROVED] ${data.old_score} > ${data.new_score} (+${data.delta})`, 'success');
				break;

			case 'no_improvement':
				addTerminal(`[NO CHANGE] ${data.score} <= ${data.prev}`, 'warn');
				break;

			case 'done':
				status = 'complete';
				addTerminal(`[DONE] Solve complete: score=${data.score} in ${data.time}s`, 'system');
				loadSolutions();
				break;

			case 'stderr':
				addTerminal(`[DBG] ${data.text}`, 'dim');
				break;

			case 'error':
				addTerminal(`[ERROR] ${data.msg}`, 'error');
				status = 'error';
				break;

			case 'process_done':
				if (status === 'solving') status = 'complete';
				break;
		}
	}

	// === SOLUTIONS ===
	async function loadSolutions() {
		try {
			const res = await fetch('/api/optimize/solutions');
			if (res.ok) solutions = await res.json();
		} catch (e) {}
	}

	async function clearSolutions(difficulty = 'all') {
		const label = difficulty === 'all' ? 'ALL difficulties' : difficulty.toUpperCase();
		addTerminal(`[CLEAR] Clearing map data for ${label}...`, 'warn');
		try {
			const res = await fetch('/api/optimize/solutions', {
				method: 'DELETE',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ difficulty }),
			});
			if (res.ok) {
				const data = await res.json();
				addTerminal(`[CLEAR] Cleared: ${data.cleared.join(', ') || 'none'}`, 'success');
				await loadSolutions();
			} else {
				addTerminal(`[CLEAR] Failed to clear`, 'error');
			}
		} catch (e) {
			addTerminal(`[CLEAR] Error: ${e.message}`, 'error');
		}
	}

	// === LIFECYCLE ===
	onMount(() => {
		loadSolutions();
	});

	onDestroy(() => {
		if (abortController) abortController.abort();
	});

	// Derived values
	let currentRound = $derived(roundData.length > 0 ? roundData[roundData.length - 1].r : 0);
	let statesPerSec = $derived(
		finalTime > 0 && maxExpanded > 0
			? Math.round(roundData.reduce((s, d) => s + d.expanded, 0) / finalTime)
			: 0
	);
	let progressPct = $derived(
		totalBots > 1
			? ((botScores.length * 300 + currentRound) / (totalBots * 300) * 100)
			: (currentRound / 300 * 100)
	);
</script>

<svelte:head>
	<title>GPU Solver</title>
</svelte:head>

<div class="gpu-page stagger">
	<!-- Header -->
	<div class="page-header">
		<h1>GPU Solver</h1>
		<div class="header-meta">
			{#if gpuName}
				<span class="chip">{gpuName}</span>
				<span class="chip">{vramTotal}GB VRAM</span>
			{/if}
			<span class="chip status-{status}">
				{#if status === 'idle'}Ready
				{:else if status === 'solving'}Computing...
				{:else if status === 'complete'}Complete
				{:else}Error
				{/if}
			</span>
		</div>
	</div>

	<!-- Stats -->
	<div class="stats-row">
		<div class="card stat-card">
			<div class="stat-label">Score</div>
			<div class="stat-value score-value">{finalScore}</div>
			<div class="stat-sub">
				{#if isOptimal}<span class="tag tag-green">Optimal</span>{/if}
				{#if prevBest > 0}<span class="stat-dim">prev: {prevBest}</span>{/if}
			</div>
		</div>
		<div class="card stat-card">
			<div class="stat-label">{totalBots > 1 ? `Bot ${Math.max(0, currentBotId)}/${totalBots}` : 'Unique States'}</div>
			<div class="stat-value">
				{#if totalBots > 1 && currentBotId >= 0}
					{botScores.length}/{totalBots}
				{:else}
					{maxUnique.toLocaleString()}
				{/if}
			</div>
			<div class="stat-sub stat-dim">
				{#if totalBots > 1}
					{maxUnique.toLocaleString()} peak states
				{:else}
					peak reachable
				{/if}
			</div>
		</div>
		<div class="card stat-card">
			<div class="stat-label">States Explored</div>
			<div class="stat-value">{roundData.reduce((s, d) => s + d.expanded, 0).toLocaleString()}</div>
			<div class="stat-sub stat-dim">{statesPerSec > 0 ? `${(statesPerSec/1000000).toFixed(1)}M/s` : '---'}</div>
		</div>
		<div class="card stat-card">
			<div class="stat-label">Solve Time</div>
			<div class="stat-value">{finalTime.toFixed(1)}s</div>
			<div class="stat-sub stat-dim">R{currentRound}/300{totalBots > 1 ? ` (Bot ${Math.max(0, currentBotId)})` : ''}</div>
		</div>
	</div>

	<!-- Progress bar -->
	{#if status === 'solving'}
		<div class="progress-container">
			<div class="progress-bar" style="width: {progressPct}%"></div>
			<div class="progress-text">
				{progressPct.toFixed(0)}% --
				{#if totalBots > 1}Bot {Math.max(0, currentBotId)}/{totalBots}, {/if}Round {currentRound}/300
			</div>
		</div>
	{/if}

	<!-- Graphs -->
	<div class="graphs-row">
		<div class="card graph-card">
			<h3>Score Progression</h3>
			<canvas bind:this={scoreCanvas} width="600" height="250"></canvas>
		</div>
		<div class="card graph-card">
			<h3>State Space Exploration</h3>
			<canvas bind:this={stateCanvas} width="600" height="250"></canvas>
		</div>
	</div>

	<!-- Solution cards -->
	<div class="solutions-row">
		{#each ['easy', 'medium', 'hard', 'expert'] as diff}
			{@const sol = solutions[diff]}
			{@const isStale = sol?.date && sol.date !== new Date().toISOString().slice(0, 10)}
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
						{sol.date || ''}
						{#if isStale}<span class="tag tag-red">old map</span>{/if}
					{:else}
						no solution
					{/if}
				</div>
			</button>
		{/each}
	</div>

	{#if Object.values(solutions).some(s => s?.score > 0)}
		<div class="clear-row">
			<button class="btn btn-danger-outline" onclick={() => clearSolutions('all')}>Clear All</button>
			{#if solutions[selectedDifficulty]?.score > 0}
				<button class="btn btn-danger-outline btn-sm" onclick={() => clearSolutions(selectedDifficulty)}>
					Clear {selectedDifficulty}
				</button>
			{/if}
		</div>
	{/if}

	<!-- Workflow -->
	<div class="workflow-row">
		<div class="wf-step" class:done={solutions[selectedDifficulty]?.score > 10}>
			<span class="wf-num">1</span>
			<div>
				<div class="wf-label">Capture & Solve</div>
				<div class="wf-desc">Python capture, then parallel solve</div>
			</div>
		</div>
		<span class="wf-arrow">&rarr;</span>
		<div class="wf-step">
			<span class="wf-num">2</span>
			<div>
				<div class="wf-label">Replay</div>
				<div class="wf-desc">Replay optimal on new token</div>
			</div>
		</div>
	</div>

	<!-- Controls -->
	<div class="card controls-section">
		<div class="url-row">
			<label class="url-label" for="ws-url">WebSocket URL</label>
			<input
				id="ws-url"
				type="text"
				class="url-input"
				placeholder="wss://game.ainm.no/ws?token=..."
				bind:value={wsUrl}
				disabled={status === 'solving'}
			/>
		</div>
		{#if tokenInfo}
			<div class="token-info">
				<span class="chip">{tokenInfo.difficulty?.toUpperCase()}</span>
				<span class="chip">seed: {tokenInfo.map_seed}</span>
				<span class="chip" class:chip-red={tokenInfo.exp * 1000 < Date.now()}>
					{#if tokenInfo.exp * 1000 < Date.now()}
						Expired
					{:else}
						{Math.max(0, Math.round((tokenInfo.exp * 1000 - Date.now()) / 1000))}s left
					{/if}
				</span>
			</div>
		{/if}
		<div class="controls-row">
			{#if status === 'solving'}
				<button class="btn btn-danger" onclick={stopSolve}>Abort</button>
			{:else}
				{@const hasSolution = solutions[selectedDifficulty]?.score > 10}
				{@const hasCapture = solutions[selectedDifficulty]?.capture_hash}
				{#if wsUrl.trim()}
					<button class="btn btn-primary" onclick={startCapture}>Capture & Solve</button>
					{#if hasCapture}
						<button class="btn btn-secondary" onclick={startSolve}>Re-Solve (GPU)</button>
					{/if}
					{#if hasSolution}
						<button class="btn btn-accent" onclick={startReplay}>
							Replay [{solutions[selectedDifficulty]?.score}]
						</button>
					{/if}
				{:else}
					{#if hasCapture}
						<button class="btn btn-primary" onclick={startSolve}>GPU Solve from Capture</button>
					{/if}
					<span class="hint">Paste a token URL above for live capture/replay</span>
				{/if}
			{/if}
		</div>
	</div>

	<!-- Terminal -->
	<div class="card terminal-section">
		<div class="terminal-header">
			<h3>System Log</h3>
		</div>
		<div class="terminal" bind:this={terminalEl}>
			{#if terminalLines.length === 0}
				<div class="terminal-line dim">$ Awaiting input. Select difficulty and press Solve.</div>
			{/if}
			{#each terminalLines as line}
				<div class="terminal-line {line.type}">{line.text}</div>
			{/each}
			{#if status === 'solving'}
				<div class="terminal-line dim cursor-line">_</div>
			{/if}
		</div>
	</div>
</div>

<style>
	.gpu-page {
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

	.chip.status-solving { color: var(--green); border-color: var(--green); }
	.chip.status-complete { color: var(--orange); border-color: var(--orange); }
	.chip.status-error { color: var(--red); border-color: var(--red); }
	.chip-red { color: var(--red) !important; border-color: var(--red) !important; }

	/* Stats */
	.stats-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.stat-card {
		text-align: center;
		padding: 1rem;
		box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
		transition: all 0.25s ease;
	}

	.stat-card:hover {
		box-shadow: 0 0 20px rgba(57, 211, 83, 0.06), 0 2px 12px rgba(0, 0, 0, 0.4);
		border-color: #484f58;
	}

	.stat-label {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.05em;
		margin-bottom: 0.25rem;
	}

	.stat-value {
		font-size: 1.5rem;
		font-family: var(--font-mono);
		font-weight: 700;
		color: var(--text);
		line-height: 1.3;
	}

	.score-value {
		color: var(--green);
	}

	.stat-sub {
		font-size: 0.7rem;
		margin-top: 0.15rem;
	}

	.stat-dim {
		color: var(--text-muted);
	}

	.tag {
		display: inline-block;
		font-size: 0.65rem;
		padding: 0.1rem 0.4rem;
		border-radius: 3px;
		font-weight: 600;
	}

	.tag-green {
		background: rgba(57, 211, 83, 0.15);
		color: var(--green);
	}

	.tag-red {
		background: rgba(248, 81, 73, 0.15);
		color: var(--red);
	}

	/* Progress */
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

	.progress-text {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		font-size: 0.7rem;
		color: var(--text);
	}

	/* Graphs */
	.graphs-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.graph-card {
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

	/* Solutions */
	.solutions-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.75rem;
		margin-bottom: 0.75rem;
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

	.solution-card.stale {
		opacity: 0.6;
	}

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

	.clear-row {
		display: flex;
		gap: 0.5rem;
		justify-content: center;
		margin-bottom: 1rem;
	}

	/* Workflow */
	.workflow-row {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.75rem;
		margin-bottom: 1rem;
	}

	.wf-step {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		padding: 0.5rem 1rem;
		border: 1px solid var(--border);
		border-radius: var(--radius);
		background: var(--bg-card);
	}

	.wf-step.done {
		border-color: var(--green);
	}

	.wf-num {
		font-size: 1rem;
		font-weight: 700;
		color: var(--text-muted);
	}

	.wf-step.done .wf-num {
		color: var(--green);
	}

	.wf-label {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--text);
	}

	.wf-desc {
		font-size: 0.7rem;
		color: var(--text-muted);
	}

	.wf-arrow {
		color: var(--text-muted);
		font-size: 1.2rem;
	}

	/* Controls */
	.controls-section {
		margin-bottom: 1rem;
	}

	.url-row {
		margin-bottom: 0.75rem;
	}

	.url-label {
		display: block;
		font-size: 0.75rem;
		color: var(--text-muted);
		margin-bottom: 0.35rem;
	}

	.url-input {
		width: 100%;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: 0.8rem;
		padding: 0.5rem 0.75rem;
		outline: none;
	}

	.url-input:focus {
		border-color: var(--accent);
	}

	.url-input::placeholder {
		color: var(--text-muted);
		opacity: 0.5;
	}

	.token-info {
		display: flex;
		gap: 0.5rem;
		margin-bottom: 0.75rem;
	}

	.controls-row {
		display: flex;
		gap: 0.75rem;
		align-items: center;
	}

	.hint {
		font-size: 0.75rem;
		color: var(--text-muted);
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

	.btn-primary {
		color: #0d1117;
		border-color: var(--accent);
		background: var(--accent);
	}

	.btn-primary:hover {
		opacity: 0.85;
	}

	.btn-secondary {
		color: var(--text);
		border-color: var(--border);
		background: var(--bg-hover);
	}

	.btn-secondary:hover {
		border-color: var(--accent-light);
	}

	.btn-accent {
		color: var(--orange);
		border-color: var(--orange);
		background: rgba(57, 211, 83, 0.1);
	}

	.btn-accent:hover {
		background: rgba(57, 211, 83, 0.2);
	}

	.btn-danger {
		color: white;
		border-color: var(--red);
		background: var(--red);
	}

	.btn-danger:hover {
		opacity: 0.85;
	}

	.btn-danger-outline {
		color: var(--red);
		border-color: var(--red);
		background: transparent;
		font-size: 0.75rem;
		padding: 0.35rem 0.75rem;
	}

	.btn-danger-outline:hover {
		background: rgba(248, 81, 73, 0.1);
	}

	.btn-sm {
		font-size: 0.7rem;
		padding: 0.3rem 0.6rem;
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
	}

	.terminal-header h3 {
		font-size: 0.8rem;
		font-weight: 600;
		color: var(--text-muted);
	}

	.terminal {
		height: 240px;
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
		.stats-row { grid-template-columns: repeat(2, 1fr); }
		.graphs-row { grid-template-columns: 1fr; }
		.solutions-row { grid-template-columns: repeat(2, 1fr); }
	}
</style>
