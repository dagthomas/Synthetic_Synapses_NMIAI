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

	// Matrix rain
	let matrixCanvas = $state(null);
	let matrixCtx = null;
	let matrixCols = [];
	let matrixAnimId = null;

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

	// Animation
	let animFrame = $state(0);
	let animId = null;

	function addTerminal(text, type = 'info') {
		terminalLines.push({ text, type, ts: Date.now() });
		if (terminalLines.length > 100) terminalLines = terminalLines.slice(-80);
		requestAnimationFrame(() => {
			if (terminalEl) terminalEl.scrollTop = terminalEl.scrollHeight;
		});
	}

	// === MATRIX RAIN ===
	function initMatrix() {
		if (!matrixCanvas) return;
		const canvas = matrixCanvas;
		const ctx = canvas.getContext('2d');
		matrixCtx = ctx;

		function resize() {
			canvas.width = window.innerWidth;
			canvas.height = window.innerHeight;
			const colW = 18;
			const numCols = Math.ceil(canvas.width / colW);
			matrixCols = [];
			for (let i = 0; i < numCols; i++) {
				matrixCols.push({
					x: i * colW,
					y: Math.random() * canvas.height,
					speed: 1 + Math.random() * 3,
					chars: [],
					len: 8 + Math.floor(Math.random() * 20),
				});
			}
		}

		resize();
		window.addEventListener('resize', resize);

		const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEF=+<>{}[]|/\\';

		function draw() {
			ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
			ctx.fillRect(0, 0, canvas.width, canvas.height);

			const intensity = status === 'solving' ? 1.0 : 0.3;

			for (const col of matrixCols) {
				col.y += col.speed * (status === 'solving' ? 1.5 : 0.7);

				if (col.y > canvas.height + col.len * 18) {
					col.y = -col.len * 18;
					col.speed = 1 + Math.random() * 3;
				}

				for (let j = 0; j < col.len; j++) {
					const cy = col.y - j * 18;
					if (cy < -18 || cy > canvas.height + 18) continue;

					const alpha = (1 - j / col.len) * intensity;
					if (j === 0) {
						ctx.fillStyle = `rgba(180, 255, 180, ${alpha})`;
						ctx.font = 'bold 16px monospace';
					} else {
						const g = Math.floor(160 + 80 * (1 - j / col.len));
						ctx.fillStyle = `rgba(0, ${g}, 65, ${alpha * 0.7})`;
						ctx.font = '14px monospace';
					}

					const char = chars[Math.floor(Math.random() * chars.length)];
					ctx.fillText(char, col.x, cy);
				}
			}

			matrixAnimId = requestAnimationFrame(draw);
		}

		draw();
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

		// Background
		ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
		ctx.fillRect(0, 0, W, H);

		const maxScore = Math.max(finalScore, ...roundData.map(d => d.score), 10);
		const maxR = 300;

		// Grid lines
		ctx.strokeStyle = 'rgba(0, 255, 65, 0.08)';
		ctx.lineWidth = 1;
		for (let s = 0; s <= maxScore; s += 25) {
			const y = pad.t + gH - (s / maxScore) * gH;
			ctx.beginPath();
			ctx.moveTo(pad.l, y);
			ctx.lineTo(pad.l + gW, y);
			ctx.stroke();
			ctx.fillStyle = 'rgba(0, 255, 65, 0.4)';
			ctx.font = '10px monospace';
			ctx.textAlign = 'right';
			ctx.fillText(String(s), pad.l - 5, y + 3);
		}
		for (let r = 0; r <= maxR; r += 50) {
			const x = pad.l + (r / maxR) * gW;
			ctx.beginPath();
			ctx.moveTo(x, pad.t);
			ctx.lineTo(x, pad.t + gH);
			ctx.stroke();
			ctx.fillStyle = 'rgba(0, 255, 65, 0.4)';
			ctx.font = '10px monospace';
			ctx.textAlign = 'center';
			ctx.fillText(String(r), x, pad.t + gH + 15);
		}

		// Score line with glow
		if (roundData.length > 1) {
			// Glow
			ctx.shadowColor = '#00ff41';
			ctx.shadowBlur = 12;
			ctx.strokeStyle = '#00ff41';
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
			ctx.shadowBlur = 0;

			// Data points
			for (let i = 0; i < roundData.length; i++) {
				const d = roundData[i];
				const x = pad.l + (d.r / maxR) * gW;
				const y = pad.t + gH - (d.score / maxScore) * gH;
				ctx.fillStyle = d.score > 0 ? '#00ff41' : 'rgba(0, 255, 65, 0.3)';
				ctx.beginPath();
				ctx.arc(x, y, 2, 0, Math.PI * 2);
				ctx.fill();
			}
		}

		// Previous best line
		if (prevBest > 0) {
			const y = pad.t + gH - (prevBest / maxScore) * gH;
			ctx.strokeStyle = 'rgba(255, 100, 0, 0.5)';
			ctx.setLineDash([5, 5]);
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(pad.l, y);
			ctx.lineTo(pad.l + gW, y);
			ctx.stroke();
			ctx.setLineDash([]);
			ctx.fillStyle = 'rgba(255, 100, 0, 0.6)';
			ctx.font = '10px monospace';
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
		ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
		ctx.fillRect(0, 0, W, H);

		const maxU = Math.max(maxUnique, 100);
		const maxR = 300;

		// Grid
		ctx.strokeStyle = 'rgba(0, 200, 255, 0.08)';
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
			ctx.fillStyle = 'rgba(0, 200, 255, 0.4)';
			ctx.font = '10px monospace';
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
			grad.addColorStop(0, 'rgba(0, 200, 255, 0.3)');
			grad.addColorStop(1, 'rgba(0, 200, 255, 0.02)');
			ctx.fillStyle = grad;
			ctx.fill();

			// Line
			ctx.shadowColor = '#00c8ff';
			ctx.shadowBlur = 8;
			ctx.strokeStyle = '#00c8ff';
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
			ctx.shadowBlur = 0;
		}

		// Label
		ctx.fillStyle = 'rgba(0, 200, 255, 0.6)';
		ctx.font = '10px monospace';
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
		addTerminal(`[PIPELINE] Python capture → Parallel solve`, 'system');
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
							// Only log every 10th round to avoid terminal spam
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
							addTerminal(`[SOLVER] IMPROVED: ${data.old_score} → ${data.new_score} (+${data.delta})`, 'success');
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
							// Don't spam connection/protocol logs
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
		// startCapture now runs the full pipeline: Python capture → Parallel solve
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
					addTerminal(`[VERIFY] GPU step matches CPU step ✓`, 'success');
				} else {
					addTerminal(`[VERIFY] MISMATCH — GPU step diverges from CPU!`, 'error');
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
				addTerminal(`[IMPROVED] ${data.old_score} → ${data.new_score} (+${data.delta})`, 'success');
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
		initMatrix();
		loadSolutions();
		animId = setInterval(() => { animFrame++; }, 100);
	});

	onDestroy(() => {
		if (matrixAnimId) cancelAnimationFrame(matrixAnimId);
		if (animId) clearInterval(animId);
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
	let statusColor = $derived(
		status === 'solving' ? '#00ff41' :
		status === 'complete' ? '#ffd700' :
		status === 'error' ? '#ff4444' : '#00ff41'
	);
</script>

<svelte:head>
	<title>GPU Neural Solver</title>
</svelte:head>

<!-- Matrix Rain Background -->
<canvas bind:this={matrixCanvas} class="matrix-bg"></canvas>

<!-- Scanlines overlay -->
<div class="scanlines"></div>

<!-- Content -->
<div class="gpu-page">
	<!-- Header -->
	<header class="gpu-header">
		<div class="title-row">
			<div class="title-glitch" data-text="GPU NEURAL SOLVER">
				<span>GPU NEURAL SOLVER</span>
			</div>
			<div class="version">v2.0 // OPTIMAL DP</div>
		</div>
		<div class="subtitle">
			{#if totalBots > 1}Sequential Per-Bot DP via CUDA ({totalBots} bots){:else}Exhaustive State-Space Search via CUDA Parallel Dynamic Programming{/if}
		</div>
		<div class="gpu-info">
			{#if gpuName}
				<span class="chip">{gpuName}</span>
				<span class="chip">{vramTotal}GB VRAM</span>
			{:else}
				<span class="chip dim">GPU: Detecting...</span>
			{/if}
			<span class="chip" style="color: {statusColor}">
				{#if status === 'idle'}READY
				{:else if status === 'solving'}COMPUTING
				{:else if status === 'complete'}COMPLETE
				{:else}ERROR
				{/if}
			</span>
		</div>
	</header>

	<!-- Status orb + big numbers -->
	<div class="stats-row">
		<div class="stat-card glow-green">
			<div class="stat-label">SCORE</div>
			<div class="stat-value big">{finalScore}</div>
			<div class="stat-sub">
				{#if isOptimal}<span class="optimal-tag">OPTIMAL</span>{/if}
				{#if prevBest > 0}prev: {prevBest}{/if}
			</div>
		</div>
		<div class="stat-card glow-cyan">
			<div class="stat-label">{totalBots > 1 ? `BOT ${Math.max(0, currentBotId)}/${totalBots}` : 'UNIQUE STATES'}</div>
			<div class="stat-value">
				{#if totalBots > 1 && currentBotId >= 0}
					{botScores.length}/{totalBots}
				{:else}
					{maxUnique.toLocaleString()}
				{/if}
			</div>
			<div class="stat-sub">
				{#if totalBots > 1}
					{maxUnique.toLocaleString()} peak states
				{:else}
					peak reachable
				{/if}
			</div>
		</div>
		<div class="stat-card glow-cyan">
			<div class="stat-label">STATES EXPLORED</div>
			<div class="stat-value">{roundData.reduce((s, d) => s + d.expanded, 0).toLocaleString()}</div>
			<div class="stat-sub">{statesPerSec > 0 ? `${(statesPerSec/1000000).toFixed(1)}M/s` : '---'}</div>
		</div>
		<div class="stat-card glow-purple">
			<div class="stat-label">SOLVE TIME</div>
			<div class="stat-value">{finalTime.toFixed(1)}s</div>
			<div class="stat-sub">R{currentRound}/300{totalBots > 1 ? ` (Bot ${Math.max(0, currentBotId)})` : ''}</div>
		</div>
	</div>

	<!-- Progress bar -->
	{#if status === 'solving'}
		<div class="progress-bar-container">
			<div class="progress-bar" style="width: {progressPct}%"></div>
			<div class="progress-text">
				{progressPct.toFixed(0)}% —
				{#if totalBots > 1}Bot {Math.max(0, currentBotId)}/{totalBots}, {/if}Round {currentRound}/300
			</div>
		</div>
	{/if}

	<!-- Graphs -->
	<div class="graphs-row">
		<div class="graph-card">
			<div class="graph-title">SCORE PROGRESSION</div>
			<canvas bind:this={scoreCanvas} width="600" height="250"></canvas>
		</div>
		<div class="graph-card">
			<div class="graph-title">STATE SPACE EXPLORATION</div>
			<canvas bind:this={stateCanvas} width="600" height="250"></canvas>
		</div>
	</div>

	<!-- Solution cards -->
	<div class="solutions-row">
		{#each ['easy', 'medium', 'hard', 'expert'] as diff}
			{@const sol = solutions[diff]}
			{@const isStale = sol?.date && sol.date !== new Date().toISOString().slice(0, 10)}
			<button
				class="solution-card"
				class:active={selectedDifficulty === diff}
				class:has-solution={sol && sol.score > 0}
				class:stale={isStale}
				onclick={() => selectedDifficulty = diff}
			>
				<div class="sol-diff">{diff.toUpperCase()}</div>
				<div class="sol-score">{sol?.score ?? '---'}</div>
				<div class="sol-meta">
					{#if sol?.score > 0}
						{sol.date || ''}
						{#if isStale}<span class="stale-tag">OLD MAP</span>{/if}
					{:else}
						no solution
					{/if}
				</div>
			</button>
		{/each}
	</div>
	{#if Object.values(solutions).some(s => s?.score > 0)}
		<div class="clear-row">
			<button class="btn btn-clear" onclick={() => clearSolutions('all')}>
				CLEAR ALL MAP DATA
			</button>
			{#if solutions[selectedDifficulty]?.score > 0}
				<button class="btn btn-clear-sm" onclick={() => clearSolutions(selectedDifficulty)}>
					CLEAR {selectedDifficulty.toUpperCase()}
				</button>
			{/if}
		</div>
	{/if}

	<!-- Workflow guide -->
	<div class="workflow-section">
		<div class="workflow-title">WORKFLOW</div>
		<div class="workflow-steps">
			<div class="wf-step" class:wf-done={solutions[selectedDifficulty]?.score > 10}>
				<span class="wf-num">1</span>
				<span class="wf-label">CAPTURE & SOLVE</span>
				<span class="wf-desc">Python capture → parallel solve</span>
			</div>
			<div class="wf-arrow">&rarr;</div>
			<div class="wf-step">
				<span class="wf-num">2</span>
				<span class="wf-label">REPLAY</span>
				<span class="wf-desc">Replay optimal on new token</span>
			</div>
		</div>
	</div>

	<!-- Controls -->
	<div class="controls-section">
		<div class="url-row">
			<div class="url-input-wrap">
				<span class="url-prefix">URL</span>
				<input
					type="text"
					class="url-input"
					placeholder="wss://game.ainm.no/ws?token=..."
					bind:value={wsUrl}
					disabled={status === 'solving'}
				/>
			</div>
		</div>
		{#if tokenInfo}
			<div class="token-info">
				<span class="chip">{tokenInfo.difficulty?.toUpperCase()}</span>
				<span class="chip">seed: {tokenInfo.map_seed}</span>
				<span class="chip" class:expired={tokenInfo.exp * 1000 < Date.now()}>
					{#if tokenInfo.exp * 1000 < Date.now()}
						EXPIRED
					{:else}
						expires: {Math.max(0, Math.round((tokenInfo.exp * 1000 - Date.now()) / 1000))}s
					{/if}
				</span>
			</div>
		{/if}
		<div class="controls-row">
			{#if status === 'solving'}
				<button class="btn btn-danger" onclick={stopSolve}>
					ABORT
				</button>
			{:else}
				{@const hasSolution = solutions[selectedDifficulty]?.score > 10}
				{@const hasCapture = solutions[selectedDifficulty]?.capture_hash}
				{#if wsUrl.trim()}
					<button class="btn btn-primary" onclick={startCapture}>
						CAPTURE & SOLVE
					</button>
					{#if hasCapture}
						<button class="btn btn-primary" onclick={startSolve}>
							RE-SOLVE (GPU)
						</button>
					{/if}
					{#if hasSolution}
						<button class="btn btn-gold" onclick={startReplay}>
							3. REPLAY [{solutions[selectedDifficulty]?.score}]
						</button>
					{/if}
				{:else}
					{#if hasCapture}
						<button class="btn btn-primary" onclick={startSolve}>
							GPU SOLVE FROM CAPTURE
						</button>
					{/if}
					<span class="hint">Paste a token URL above for live capture/replay</span>
				{/if}
			{/if}
		</div>
	</div>

	<!-- Terminal -->
	<div class="terminal-section">
		<div class="terminal-header">
			<span class="terminal-dot red"></span>
			<span class="terminal-dot yellow"></span>
			<span class="terminal-dot green"></span>
			<span class="terminal-title">SYSTEM LOG</span>
		</div>
		<div class="terminal" bind:this={terminalEl}>
			{#if terminalLines.length === 0}
				<div class="terminal-line dim">
					<span class="prompt">$</span> Awaiting input. Select difficulty and press SOLVE.
				</div>
			{/if}
			{#each terminalLines as line}
				<div class="terminal-line {line.type}">
					<span class="prompt">{'>'}</span>
					{line.text}
				</div>
			{/each}
			{#if status === 'solving'}
				<div class="terminal-line cursor-blink">
					<span class="prompt">{'>'}</span>
					<span class="cursor">_</span>
				</div>
			{/if}
		</div>
	</div>

	<!-- Footer -->
	<div class="footer">
		<span>GROCERY BOT GPU SOLVER // CUDA DP // RTX 5090</span>
		<span>FRAME {animFrame}</span>
	</div>
</div>

<style>
	/* === MATRIX BACKGROUND === */
	.matrix-bg {
		position: fixed;
		top: 0;
		left: 0;
		width: 100vw;
		height: 100vh;
		z-index: 0;
		pointer-events: none;
	}

	.scanlines {
		position: fixed;
		top: 0;
		left: 0;
		width: 100vw;
		height: 100vh;
		z-index: 1;
		pointer-events: none;
		background: repeating-linear-gradient(
			0deg,
			transparent,
			transparent 2px,
			rgba(0, 0, 0, 0.03) 2px,
			rgba(0, 0, 0, 0.03) 4px
		);
	}

	/* === PAGE CONTAINER === */
	.gpu-page {
		position: relative;
		z-index: 2;
		max-width: 1300px;
		margin: 0 auto;
		padding: 0 1rem;
		font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
		color: #00ff41;
	}

	/* === HEADER === */
	.gpu-header {
		text-align: center;
		padding: 1rem 0 0.5rem;
	}

	.title-row {
		display: flex;
		align-items: baseline;
		justify-content: center;
		gap: 1rem;
	}

	.title-glitch {
		font-size: 2rem;
		font-weight: 900;
		letter-spacing: 0.15em;
		color: #00ff41;
		text-shadow:
			0 0 10px rgba(0, 255, 65, 0.5),
			0 0 30px rgba(0, 255, 65, 0.3),
			0 0 60px rgba(0, 255, 65, 0.15);
		position: relative;
	}

	.title-glitch::before,
	.title-glitch::after {
		content: attr(data-text);
		position: absolute;
		top: 0;
		left: 0;
		overflow: hidden;
	}

	.title-glitch::before {
		animation: glitch1 3s infinite;
		color: #ff00ff;
		opacity: 0.3;
		clip-path: inset(0 0 65% 0);
	}

	.title-glitch::after {
		animation: glitch2 3s infinite;
		color: #00ffff;
		opacity: 0.3;
		clip-path: inset(35% 0 0 0);
	}

	@keyframes glitch1 {
		0%, 95%, 100% { transform: translate(0); }
		96% { transform: translate(-3px, 1px); }
		97% { transform: translate(3px, -1px); }
		98% { transform: translate(-1px, 2px); }
	}

	@keyframes glitch2 {
		0%, 93%, 100% { transform: translate(0); }
		94% { transform: translate(2px, -1px); }
		95% { transform: translate(-2px, 1px); }
		96% { transform: translate(1px, -2px); }
	}

	.version {
		font-size: 0.7rem;
		color: rgba(0, 255, 65, 0.4);
		letter-spacing: 0.1em;
	}

	.subtitle {
		font-size: 0.7rem;
		color: rgba(0, 255, 65, 0.35);
		letter-spacing: 0.08em;
		margin-top: 0.25rem;
	}

	.gpu-info {
		display: flex;
		gap: 0.5rem;
		justify-content: center;
		margin-top: 0.5rem;
		flex-wrap: wrap;
	}

	.chip {
		font-size: 0.65rem;
		padding: 0.2rem 0.6rem;
		border: 1px solid rgba(0, 255, 65, 0.25);
		border-radius: 2px;
		color: rgba(0, 255, 65, 0.7);
		background: rgba(0, 255, 65, 0.05);
		letter-spacing: 0.05em;
	}

	.chip.dim {
		color: rgba(0, 255, 65, 0.3);
		border-color: rgba(0, 255, 65, 0.1);
	}

	/* === STATS ROW === */
	.stats-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.75rem;
		margin: 1rem 0;
	}

	.stat-card {
		background: rgba(0, 0, 0, 0.6);
		border: 1px solid rgba(0, 255, 65, 0.2);
		border-radius: 4px;
		padding: 0.75rem;
		text-align: center;
		position: relative;
		overflow: hidden;
	}

	.stat-card::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 2px;
	}

	.stat-card.glow-green {
		border-color: rgba(0, 255, 65, 0.3);
	}
	.stat-card.glow-green::before {
		background: linear-gradient(90deg, transparent, #00ff41, transparent);
		box-shadow: 0 0 15px rgba(0, 255, 65, 0.5);
	}

	.stat-card.glow-cyan {
		border-color: rgba(0, 200, 255, 0.25);
	}
	.stat-card.glow-cyan::before {
		background: linear-gradient(90deg, transparent, #00c8ff, transparent);
		box-shadow: 0 0 15px rgba(0, 200, 255, 0.4);
	}
	.stat-card.glow-cyan .stat-label,
	.stat-card.glow-cyan .stat-value,
	.stat-card.glow-cyan .stat-sub {
		color: #00c8ff;
	}

	.stat-card.glow-purple {
		border-color: rgba(160, 100, 255, 0.25);
	}
	.stat-card.glow-purple::before {
		background: linear-gradient(90deg, transparent, #a064ff, transparent);
		box-shadow: 0 0 15px rgba(160, 100, 255, 0.4);
	}
	.stat-card.glow-purple .stat-label,
	.stat-card.glow-purple .stat-value,
	.stat-card.glow-purple .stat-sub {
		color: #a064ff;
	}

	.stat-label {
		font-size: 0.55rem;
		letter-spacing: 0.15em;
		color: rgba(0, 255, 65, 0.5);
		margin-bottom: 0.25rem;
	}

	.stat-value {
		font-size: 1.5rem;
		font-weight: 700;
		color: #00ff41;
		text-shadow: 0 0 15px rgba(0, 255, 65, 0.5);
		line-height: 1.2;
	}

	.stat-value.big {
		font-size: 2.2rem;
	}

	.stat-sub {
		font-size: 0.55rem;
		color: rgba(0, 255, 65, 0.35);
		margin-top: 0.15rem;
	}

	.optimal-tag {
		display: inline-block;
		font-size: 0.55rem;
		padding: 0.1rem 0.4rem;
		background: rgba(255, 215, 0, 0.15);
		border: 1px solid rgba(255, 215, 0, 0.4);
		color: #ffd700;
		border-radius: 2px;
		letter-spacing: 0.1em;
		animation: pulse-gold 2s infinite;
	}

	@keyframes pulse-gold {
		0%, 100% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.2); }
		50% { box-shadow: 0 0 15px rgba(255, 215, 0, 0.5); }
	}

	/* === PROGRESS BAR === */
	.progress-bar-container {
		position: relative;
		height: 24px;
		background: rgba(0, 0, 0, 0.5);
		border: 1px solid rgba(0, 255, 65, 0.2);
		border-radius: 2px;
		margin: 0.5rem 0;
		overflow: hidden;
	}

	.progress-bar {
		height: 100%;
		background: linear-gradient(90deg,
			rgba(0, 255, 65, 0.2),
			rgba(0, 255, 65, 0.4));
		transition: width 0.3s ease;
		position: relative;
	}

	.progress-bar::after {
		content: '';
		position: absolute;
		right: 0;
		top: 0;
		bottom: 0;
		width: 50px;
		background: linear-gradient(90deg, transparent, rgba(0, 255, 65, 0.6));
		animation: shimmer 1s infinite;
	}

	@keyframes shimmer {
		0% { opacity: 0.3; }
		50% { opacity: 1; }
		100% { opacity: 0.3; }
	}

	.progress-text {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		font-size: 0.65rem;
		color: #00ff41;
		letter-spacing: 0.1em;
		text-shadow: 0 0 5px rgba(0, 255, 65, 0.5);
	}

	/* === GRAPHS === */
	.graphs-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.75rem;
		margin: 0.75rem 0;
	}

	.graph-card {
		background: rgba(0, 0, 0, 0.5);
		border: 1px solid rgba(0, 255, 65, 0.15);
		border-radius: 4px;
		padding: 0.5rem;
	}

	.graph-title {
		font-size: 0.6rem;
		letter-spacing: 0.15em;
		color: rgba(0, 255, 65, 0.5);
		margin-bottom: 0.25rem;
		text-align: center;
	}

	.graph-card canvas {
		width: 100%;
		height: auto;
		display: block;
	}

	/* === SOLUTIONS === */
	.solutions-row {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 0.75rem;
		margin: 0.75rem 0;
	}

	.solution-card {
		background: rgba(0, 0, 0, 0.5);
		border: 1px solid rgba(0, 255, 65, 0.1);
		border-radius: 4px;
		padding: 0.6rem;
		text-align: center;
		cursor: pointer;
		transition: all 0.2s;
		font-family: inherit;
		color: inherit;
	}

	.solution-card:hover {
		border-color: rgba(0, 255, 65, 0.3);
		background: rgba(0, 255, 65, 0.05);
	}

	.solution-card.active {
		border-color: rgba(0, 255, 65, 0.5);
		box-shadow: 0 0 15px rgba(0, 255, 65, 0.15), inset 0 0 15px rgba(0, 255, 65, 0.05);
	}

	.solution-card.has-solution {
		border-color: rgba(0, 255, 65, 0.3);
	}

	.sol-diff {
		font-size: 0.6rem;
		letter-spacing: 0.15em;
		color: rgba(0, 255, 65, 0.5);
	}

	.sol-score {
		font-size: 1.4rem;
		font-weight: 700;
		color: #00ff41;
		text-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
		line-height: 1.3;
	}

	.sol-meta {
		font-size: 0.55rem;
		color: rgba(0, 255, 65, 0.3);
	}

	.stale-tag {
		display: inline-block;
		font-size: 0.5rem;
		padding: 0.05rem 0.3rem;
		background: rgba(255, 68, 68, 0.15);
		border: 1px solid rgba(255, 68, 68, 0.4);
		color: #ff6666;
		border-radius: 2px;
		letter-spacing: 0.05em;
		margin-left: 0.25rem;
	}

	.solution-card.stale {
		border-color: rgba(255, 68, 68, 0.2);
		opacity: 0.6;
	}

	.clear-row {
		display: flex;
		gap: 0.5rem;
		justify-content: center;
		margin-top: 0.25rem;
	}

	.btn-clear {
		font-family: 'JetBrains Mono', monospace;
		font-size: 0.6rem;
		padding: 0.3rem 0.8rem;
		color: #ff6666;
		border: 1px solid rgba(255, 68, 68, 0.3);
		background: rgba(255, 68, 68, 0.05);
		border-radius: 3px;
		cursor: pointer;
		letter-spacing: 0.1em;
	}

	.btn-clear:hover {
		background: rgba(255, 68, 68, 0.12);
		border-color: rgba(255, 68, 68, 0.5);
	}

	.btn-clear-sm {
		font-family: 'JetBrains Mono', monospace;
		font-size: 0.55rem;
		padding: 0.25rem 0.6rem;
		color: rgba(255, 102, 102, 0.6);
		border: 1px solid rgba(255, 68, 68, 0.2);
		background: transparent;
		border-radius: 3px;
		cursor: pointer;
		letter-spacing: 0.05em;
	}

	.btn-clear-sm:hover {
		background: rgba(255, 68, 68, 0.08);
	}

	/* === WORKFLOW === */
	.workflow-section {
		margin: 0.75rem 0 0.25rem;
		text-align: center;
	}

	.workflow-title {
		font-size: 0.55rem;
		letter-spacing: 0.2em;
		color: rgba(0, 255, 65, 0.3);
		margin-bottom: 0.4rem;
	}

	.workflow-steps {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
	}

	.wf-step {
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: 0.4rem 0.8rem;
		border: 1px solid rgba(0, 255, 65, 0.12);
		border-radius: 4px;
		background: rgba(0, 0, 0, 0.3);
		min-width: 100px;
	}

	.wf-step.wf-done {
		border-color: rgba(0, 255, 65, 0.35);
		background: rgba(0, 255, 65, 0.05);
	}

	.wf-num {
		font-size: 0.8rem;
		font-weight: 700;
		color: rgba(0, 255, 65, 0.25);
	}

	.wf-done .wf-num {
		color: #00ff41;
	}

	.wf-label {
		font-size: 0.6rem;
		letter-spacing: 0.12em;
		color: rgba(0, 255, 65, 0.5);
		font-weight: 600;
	}

	.wf-done .wf-label {
		color: #00ff41;
	}

	.wf-desc {
		font-size: 0.5rem;
		color: rgba(0, 255, 65, 0.2);
		margin-top: 0.1rem;
	}

	.wf-arrow {
		color: rgba(0, 255, 65, 0.2);
		font-size: 1.2rem;
	}

	/* === CONTROLS === */
	.controls-section {
		margin: 0.75rem 0;
	}

	.url-row {
		margin-bottom: 0.5rem;
	}

	.url-input-wrap {
		display: flex;
		align-items: center;
		background: rgba(0, 0, 0, 0.6);
		border: 1px solid rgba(0, 255, 65, 0.2);
		border-radius: 3px;
		overflow: hidden;
	}

	.url-prefix {
		padding: 0.5rem 0.6rem;
		font-size: 0.65rem;
		color: rgba(0, 255, 65, 0.3);
		background: rgba(0, 255, 65, 0.05);
		border-right: 1px solid rgba(0, 255, 65, 0.1);
		letter-spacing: 0.05em;
	}

	.url-input {
		flex: 1;
		background: transparent;
		border: none;
		color: #00ff41;
		font-family: 'JetBrains Mono', monospace;
		font-size: 0.7rem;
		padding: 0.5rem;
		outline: none;
		letter-spacing: 0.02em;
	}

	.url-input::placeholder {
		color: rgba(0, 255, 65, 0.2);
	}

	.url-input:focus {
		box-shadow: inset 0 0 20px rgba(0, 255, 65, 0.05);
	}

	.controls-row {
		display: flex;
		gap: 0.75rem;
		justify-content: center;
		align-items: center;
		margin: 0.5rem 0;
	}

	.hint {
		font-size: 0.6rem;
		color: rgba(0, 255, 65, 0.25);
	}

	.btn {
		font-family: 'JetBrains Mono', monospace;
		font-size: 0.75rem;
		padding: 0.6rem 2rem;
		letter-spacing: 0.15em;
		border-radius: 3px;
		border: 1px solid;
		cursor: pointer;
		transition: all 0.2s;
		text-transform: uppercase;
	}

	.btn-primary {
		color: #00ff41;
		border-color: #00ff41;
		background: rgba(0, 255, 65, 0.1);
	}

	.btn-primary:hover {
		background: rgba(0, 255, 65, 0.2);
		box-shadow: 0 0 20px rgba(0, 255, 65, 0.3), inset 0 0 20px rgba(0, 255, 65, 0.1);
	}

	.btn-gold {
		color: #ffd700;
		border-color: #ffd700;
		background: rgba(255, 215, 0, 0.1);
	}

	.btn-gold:hover {
		background: rgba(255, 215, 0, 0.2);
		box-shadow: 0 0 20px rgba(255, 215, 0, 0.3), inset 0 0 20px rgba(255, 215, 0, 0.1);
	}

	.btn-secondary {
		color: rgba(0, 255, 65, 0.5);
		border-color: rgba(0, 255, 65, 0.2);
		background: rgba(0, 255, 65, 0.03);
		font-size: 0.65rem;
		padding: 0.4rem 1rem;
	}

	.btn-secondary:hover {
		background: rgba(0, 255, 65, 0.08);
		border-color: rgba(0, 255, 65, 0.4);
	}

	.btn-danger {
		color: #ff4444;
		border-color: #ff4444;
		background: rgba(255, 68, 68, 0.1);
	}

	.btn-danger:hover {
		background: rgba(255, 68, 68, 0.2);
		box-shadow: 0 0 20px rgba(255, 68, 68, 0.3);
	}

	.token-info {
		display: flex;
		gap: 0.5rem;
		justify-content: center;
		margin-bottom: 0.5rem;
	}

	.token-info .expired {
		color: #ff4444 !important;
		border-color: rgba(255, 68, 68, 0.4) !important;
	}

	/* === TERMINAL === */
	.terminal-section {
		margin: 0.75rem 0;
		border: 1px solid rgba(0, 255, 65, 0.15);
		border-radius: 4px;
		overflow: hidden;
		background: rgba(0, 0, 0, 0.7);
	}

	.terminal-header {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		padding: 0.4rem 0.6rem;
		background: rgba(0, 255, 65, 0.05);
		border-bottom: 1px solid rgba(0, 255, 65, 0.1);
	}

	.terminal-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}

	.terminal-dot.red { background: #ff5f56; }
	.terminal-dot.yellow { background: #ffbd2e; }
	.terminal-dot.green { background: #27c93f; }

	.terminal-title {
		font-size: 0.6rem;
		color: rgba(0, 255, 65, 0.4);
		letter-spacing: 0.15em;
		margin-left: 0.5rem;
	}

	.terminal {
		height: 220px;
		overflow-y: auto;
		padding: 0.5rem;
		font-size: 0.68rem;
		line-height: 1.6;
	}

	.terminal-line {
		white-space: pre-wrap;
		word-break: break-all;
	}

	.terminal-line .prompt {
		color: rgba(0, 255, 65, 0.4);
		margin-right: 0.3rem;
	}

	.terminal-line.system { color: rgba(0, 200, 255, 0.8); }
	.terminal-line.info { color: rgba(0, 255, 65, 0.7); }
	.terminal-line.success { color: #00ff41; text-shadow: 0 0 5px rgba(0, 255, 65, 0.3); }
	.terminal-line.warn { color: #ffd700; }
	.terminal-line.error { color: #ff4444; }
	.terminal-line.dim { color: rgba(0, 255, 65, 0.25); }

	.cursor {
		animation: blink-cursor 1s infinite;
	}

	@keyframes blink-cursor {
		0%, 100% { opacity: 1; }
		50% { opacity: 0; }
	}

	/* === FOOTER === */
	.footer {
		display: flex;
		justify-content: space-between;
		padding: 0.5rem 0;
		font-size: 0.55rem;
		color: rgba(0, 255, 65, 0.2);
		letter-spacing: 0.1em;
		border-top: 1px solid rgba(0, 255, 65, 0.08);
		margin-top: 0.5rem;
	}

	/* === RESPONSIVE === */
	@media (max-width: 900px) {
		.stats-row { grid-template-columns: repeat(2, 1fr); }
		.graphs-row { grid-template-columns: 1fr; }
		.solutions-row { grid-template-columns: repeat(2, 1fr); }
		.title-glitch { font-size: 1.4rem; }
	}
</style>
