<script>
	import { invalidateAll } from '$app/navigation';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';

	let { data } = $props();

	// Poll for new runs every 30 seconds
	onMount(() => {
		const interval = setInterval(() => invalidateAll(), 30_000);
		return () => clearInterval(interval);
	});

	// Panel visibility (persisted to localStorage)
	const PANEL_KEY = 'panels';
	const defaultPanels = { stats: true, breakdown: true, chart: true };

	let panels = $state({ ...defaultPanels });
	let mounted = $state(false);

	onMount(() => {
		try {
			const saved = localStorage.getItem(PANEL_KEY);
			if (saved) Object.assign(panels, JSON.parse(saved));
		} catch { // Stream closed — client disconnected
		}
		// Delay so layout scrambleIn finishes before we allow page-level scramble
		requestAnimationFrame(() => { mounted = true; });
	});

	function togglePanel(key) {
		panels[key] = !panels[key];
		if (browser) localStorage.setItem(PANEL_KEY, JSON.stringify(panels));
	}

	const diffColors = {
		easy: '#39d353',
		medium: '#d29922',
		hard: '#f85149',
		expert: '#da3633',
	};

	let filterDiff = $state('all');
	let filterType = $state('all');

	let filteredRuns = $derived(
		data.runs.filter(r =>
			(filterDiff === 'all' || r.difficulty === filterDiff) &&
			(filterType === 'all' || r.run_type === filterType)
		)
	);

	let totalMaxSum = $derived(data.stats.reduce((sum, s) => sum + (s.max_score || 0), 0));
	let totalAvgSum = $derived(data.stats.reduce((sum, s) => sum + (s.avg_score || 0), 0).toFixed(1));

	let typeBreakdown = $derived(() => {
		const map = {};
		for (const ts of data.typeStats || []) {
			if (!map[ts.difficulty]) map[ts.difficulty] = {};
			map[ts.difficulty][ts.run_type] = ts;
		}
		return map;
	});

	const TARGETS = { easy: 150, medium: 225, hard: 260, expert: 310 };

	// Chart constants
	const chartW = 700;
	const chartH = 200;
	const pad = {t: 10, r: 10, b: 24, l: 40};
	let chartSorted = $derived([...data.runs].sort((a,b) => new Date(a.created_at) - new Date(b.created_at)));
	let tMin = $derived(new Date(chartSorted[0]?.created_at).getTime());
	let tMax = $derived(new Date(chartSorted[chartSorted.length-1]?.created_at).getTime());
	let tRange = $derived(Math.max(1, tMax - tMin));
	let maxScore = $derived(Math.max(...data.runs.map(r => r.final_score), 1));
	let yMax = $derived(Math.ceil(maxScore / 50) * 50);

	// Pagination
	function goPage(p) {
		goto(`?page=${p}`, { keepFocus: true, noScroll: true });
	}

	// Scramble-in effect (same as layout, applied per-panel on toggle)
	const GLYPHS = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~';

	function scrambleIn(node) {
		// On initial render, layout's scrambleIn handles everything — skip to avoid race
		if (!mounted) return;
		const SKIP = new Set(['SCRIPT', 'STYLE', 'SVG', 'CANVAS', 'IMG', 'INPUT', 'TEXTAREA', 'SELECT']);
		const entries = [];

		function walk(el) {
			if (SKIP.has(el.tagName)) return;
			for (const child of el.childNodes) {
				if (child.nodeType === Node.TEXT_NODE) {
					const original = child.textContent;
					if (original.trim().length > 0) entries.push({ node: child, original });
				} else if (child.nodeType === Node.ELEMENT_NODE) {
					walk(child);
				}
			}
		}
		walk(node);
		if (entries.length === 0) return;

		for (const e of entries) {
			e.node.textContent = e.original.replace(/\S/g, () => GLYPHS[Math.random() * GLYPHS.length | 0]);
		}

		const DURATION = 400, STAGGER = 30;
		const t0 = performance.now();
		let frame;

		function tick() {
			const elapsed = performance.now() - t0;
			let allDone = true;
			for (const e of entries) {
				const chars = [...e.original];
				const out = [];
				for (let i = 0; i < chars.length; i++) {
					const ch = chars[i];
					if (ch === ' ' || ch === '\n' || ch === '\t') { out.push(ch); continue; }
					const lockAt = (i * STAGGER) + Math.random() * 20;
					if (elapsed >= lockAt + DURATION * 0.3) { out.push(ch); }
					else { out.push(GLYPHS[Math.random() * GLYPHS.length | 0]); allDone = false; }
				}
				e.node.textContent = out.join('');
			}
			if (!allDone && elapsed < DURATION + entries.reduce((m, e) => Math.max(m, e.original.length), 0) * STAGGER) {
				frame = requestAnimationFrame(tick);
			} else {
				for (const e of entries) e.node.textContent = e.original;
			}
		}

		frame = requestAnimationFrame(tick);
		return { destroy() { cancelAnimationFrame(frame); for (const e of entries) e.node.textContent = e.original; } };
	}
</script>

<div class="stagger">
<div class="header">
	<h1>Game Runs <span class="run-count">{data.totalRuns}</span></h1>
	<div class="filters">
		<button class="chip" class:active={filterDiff === 'all'} onclick={() => filterDiff = 'all'}>All</button>
		{#each ['easy', 'medium', 'hard', 'expert'] as diff}
			<button
				class="chip"
				class:active={filterDiff === diff}
				style="--chip-color: {diffColors[diff]}"
				onclick={() => filterDiff = diff}
			>{diff}</button>
		{/each}
		<span class="filter-sep">|</span>
		<button class="chip" class:active={filterType === 'all'} onclick={() => filterType = 'all'}>All types</button>
		<button class="chip type-live" class:active={filterType === 'live'} onclick={() => filterType = 'live'}>Live</button>
		<button class="chip type-replay" class:active={filterType === 'replay'} onclick={() => filterType = 'replay'}>Replay</button>
		<button class="chip type-synthetic" class:active={filterType === 'synthetic'} onclick={() => filterType = 'synthetic'}>Synthetic</button>
		<span class="filter-sep">|</span>
		<button class="chip panel-toggle" class:active={panels.stats} onclick={() => togglePanel('stats')}>Stats</button>
		<button class="chip panel-toggle" class:active={panels.breakdown} onclick={() => togglePanel('breakdown')}>Breakdown</button>
		<button class="chip panel-toggle" class:active={panels.chart} onclick={() => togglePanel('chart')}>Chart</button>
	</div>
</div>

{#if panels.stats && data.stats.length > 0}
<div class="stats-row" use:scrambleIn>
	{#each data.stats as s}
		<div class="stat-card card">
			<div class="stat-label" style="color: {diffColors[s.difficulty]}">{s.difficulty}</div>
			<div class="stat-main">{s.max_score}</div>
			<div class="stat-sub">max / {s.avg_score} avg / {s.count} runs</div>
		</div>
	{/each}
	{#if data.stats.length > 1}
		<div class="stat-card card total-card">
			<div class="stat-label" style="color: var(--accent-light)">Total</div>
			<div class="stat-main">{totalMaxSum}</div>
			<div class="stat-sub">sum max / {totalAvgSum} sum avg</div>
		</div>
	{/if}
</div>
{/if}

{#if panels.breakdown && data.typeStats && data.typeStats.length > 0}
<div class="type-breakdown card" use:scrambleIn>
	<h2>Best Scores by Type</h2>
	<table class="breakdown-table">
		<thead>
			<tr>
				<th>Difficulty</th>
				<th>Live</th>
				<th>Replay</th>
				<th>Synthetic</th>
				<th>Best</th>
				<th>Target</th>
			</tr>
		</thead>
		<tbody>
			{#each ['easy', 'medium', 'hard', 'expert'] as diff}
				{@const tb = typeBreakdown()[diff] || {}}
				{@const best = Math.max(tb.live?.max_score || 0, tb.replay?.max_score || 0, tb.synthetic?.max_score || 0)}
				{@const target = TARGETS[diff] || 250}
				{@const pct = Math.min(100, Math.round(best / target * 100))}
				<tr>
					<td><span class="badge" style="background: {diffColors[diff]}20; color: {diffColors[diff]}; border: 1px solid {diffColors[diff]}44">{diff}</span></td>
					<td class="mono">{tb.live?.max_score ?? '—'}{#if tb.live} <span class="muted">({tb.live.count})</span>{/if}</td>
					<td class="mono">{tb.replay?.max_score ?? '—'}{#if tb.replay} <span class="muted">({tb.replay.count})</span>{/if}</td>
					<td class="mono">{tb.synthetic?.max_score ?? '—'}{#if tb.synthetic} <span class="muted">({tb.synthetic.count})</span>{/if}</td>
					<td class="score">{best || '—'}</td>
					<td>
						<div class="progress-bar">
							<div class="progress-fill" style="width: {pct}%; background: {best >= target ? '#39d353' : diffColors[diff]}"></div>
							<span class="progress-label">{pct}%</span>
						</div>
					</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
{/if}

{#if panels.chart && data.runs.length > 1}
<div class="chart-card card" use:scrambleIn>
	<h2>Score Timeline</h2>
	<svg viewBox="0 0 {chartW} {chartH}" class="timeline-svg">
		{#each Array(5) as _, i}
			{@const yVal = Math.round(yMax * (1 - i/4))}
			{@const y = pad.t + (chartH - pad.t - pad.b) * i / 4}
			<line x1={pad.l} y1={y} x2={chartW - pad.r} y2={y} stroke="#30363d" stroke-width="0.5"/>
			<text x={pad.l - 4} y={y + 3} fill="#8b949e" font-size="9" text-anchor="end" font-family="var(--font-mono)">{yVal}</text>
		{/each}
		{#each chartSorted as run}
			{@const x = pad.l + (new Date(run.created_at).getTime() - tMin) / tRange * (chartW - pad.l - pad.r)}
			{@const y = pad.t + (1 - run.final_score / yMax) * (chartH - pad.t - pad.b)}
			{@const c = diffColors[run.difficulty] || '#8b949e'}
			<circle cx={x} cy={y} r="3.5"
				fill={c} opacity="0.8" stroke={c} stroke-width="0.5"
				stroke-opacity="0.3">
				<title>{run.difficulty} {run.run_type}: {run.final_score} (seed {run.seed})</title>
			</circle>
		{/each}
		{#each Object.entries(diffColors) as [diff, color], i}
			<circle cx={pad.l + i * 80} cy={chartH - 6} r="4" fill={color} opacity="0.8"/>
			<text x={pad.l + i * 80 + 8} y={chartH - 3} fill="#8b949e" font-size="9">{diff}</text>
		{/each}
	</svg>
</div>
{/if}

{#if filteredRuns.length === 0}
	<div class="empty card">No runs recorded yet. Use <code>recorder.py</code> to record games.</div>
{:else}
<div class="table-wrap card">
	<table>
		<thead>
			<tr>
				<th>ID</th>
				<th>Type</th>
				<th>Difficulty</th>
				<th>Seed</th>
				<th>Grid</th>
				<th>Bots</th>
				<th>Score</th>
				<th>Orders</th>
				<th>Items</th>
				<th>Recorded</th>
				<th></th>
			</tr>
		</thead>
		<tbody>
			{#each filteredRuns as run}
				<tr>
					<td class="mono">#{run.id}</td>
					<td>
						<span class="badge type-badge type-{run.run_type}">
							{run.run_type}
						</span>
					</td>
					<td>
						<span class="badge" style="background: {diffColors[run.difficulty]}20; color: {diffColors[run.difficulty]}; border: 1px solid {diffColors[run.difficulty]}44">
							{run.difficulty}
						</span>
					</td>
					<td class="mono">{run.seed}</td>
					<td class="mono">{run.grid_width}x{run.grid_height}</td>
					<td>{run.bot_count}</td>
					<td class="score">{run.final_score}</td>
					<td>{run.orders_completed}</td>
					<td>{run.items_delivered}</td>
					<td class="muted">{new Date(run.created_at).toLocaleString()}</td>
					<td><a href="/run/{run.id}" class="view-btn">View</a></td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
{/if}

{#if data.totalPages > 1}
<div class="pagination">
	<button class="page-btn" disabled={data.page <= 1} onclick={() => goPage(data.page - 1)}>Prev</button>
	{#each Array(data.totalPages) as _, i}
		{@const p = i + 1}
		{#if p === 1 || p === data.totalPages || (p >= data.page - 2 && p <= data.page + 2)}
			<button class="page-btn" class:page-active={p === data.page} onclick={() => goPage(p)}>{p}</button>
		{:else if p === data.page - 3 || p === data.page + 3}
			<span class="page-dots">...</span>
		{/if}
	{/each}
	<button class="page-btn" disabled={data.page >= data.totalPages} onclick={() => goPage(data.page + 1)}>Next</button>
</div>
{/if}
</div>

<style>
	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1.5rem;
		flex-wrap: wrap;
		gap: 1rem;
	}
	h1 { font-size: 1.5rem; font-family: var(--font-display); font-weight: 700; }
	.run-count {
		font-size: 0.85rem;
		font-family: var(--font-mono);
		color: var(--text-muted);
		font-weight: 400;
		margin-left: 0.5rem;
	}
	.filters { display: flex; gap: 0.5rem; flex-wrap: wrap; }
	.chip {
		padding: 0.35rem 0.75rem;
		background: var(--bg-card);
		border: 1px solid var(--border);
		color: var(--text-muted);
		text-transform: capitalize;
		font-size: 0.8rem;
		backdrop-filter: blur(4px);
		transition: all 0.2s ease;
		cursor: pointer;
	}
	.chip.active {
		background: var(--chip-color, var(--accent))15;
		color: var(--chip-color, var(--accent-light));
		border-color: var(--chip-color, var(--accent));
		box-shadow: 0 0 8px color-mix(in srgb, var(--chip-color, var(--accent)) 20%, transparent);
	}
	.panel-toggle { font-size: 0.7rem; padding: 0.25rem 0.5rem; }
	.panel-toggle.active {
		background: rgba(88, 166, 255, 0.1);
		color: #58a6ff;
		border-color: #58a6ff44;
	}
	.stats-row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 0.75rem;
		margin-bottom: 1.5rem;
	}
	.stat-card { text-align: center; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4); transition: all 0.25s ease; }
	.stat-card:hover { box-shadow: 0 0 20px rgba(57, 211, 83, 0.06), 0 2px 12px rgba(0, 0, 0, 0.4); border-color: #484f58; }
	.total-card { border: 1px solid rgba(57, 211, 83, 0.3); background: rgba(57, 211, 83, 0.04); }
	.stat-label { font-size: 0.75rem; text-transform: uppercase; font-family: var(--font-body); font-weight: 500; letter-spacing: 0.05em; }
	.stat-main { font-size: 2.25rem; font-family: var(--font-mono); font-weight: 700; color: var(--text); }
	.stat-sub { font-size: 0.75rem; color: var(--text-muted); }
	.table-wrap { overflow-x: auto; padding: 0; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4); }
	table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
	th {
		text-align: left;
		padding: 0.75rem 1rem;
		border-bottom: 1px solid var(--border);
		color: var(--text-muted);
		font-weight: 500;
		font-size: 0.75rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		font-family: var(--font-body);
	}
	td {
		padding: 0.6rem 1rem;
		border-bottom: 1px solid var(--border);
	}
	tr:hover td { background: rgba(22, 27, 34, 0.8); }
	.mono { font-family: var(--font-mono); font-size: 0.8rem; }
	.muted { color: var(--text-muted); font-size: 0.8rem; }
	.score { font-weight: 600; font-size: 1rem; color: var(--green); }
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: capitalize;
		letter-spacing: 0.02em;
	}
	.view-btn {
		padding: 0.3rem 0.75rem;
		background: transparent;
		border: 1px solid var(--accent);
		color: var(--accent);
		border-radius: var(--radius);
		font-size: 0.8rem;
		font-weight: 500;
		letter-spacing: 0.02em;
		transition: all 0.2s ease;
	}
	.view-btn:hover { background: var(--accent); color: #0d1117; text-decoration: none; }
	.empty { padding: 3rem; text-align: center; color: var(--text-muted); }
	code { background: #010409; border: 1px solid rgba(48, 54, 61, 0.5); padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }
	.filter-sep { color: var(--border); padding: 0 0.25rem; align-self: center; }
	.type-badge { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
	.type-live { background: #58a6ff20; color: #58a6ff; border: 1px solid #58a6ff44; }
	.type-replay { background: #d2992220; color: #d29922; border: 1px solid #d2992244; }
	.type-synthetic { background: #39d35320; color: #39d353; border: 1px solid #39d35344; }
	.chip.type-live.active { background: #58a6ff22; color: #58a6ff; border-color: #58a6ff; }
	.chip.type-replay.active { background: #d2992222; color: #d29922; border-color: #d29922; }
	.chip.type-synthetic.active { background: #39d35322; color: #39d353; border-color: #39d353; }
	.type-breakdown { margin-bottom: 1.5rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4); }
	.type-breakdown h2 { font-size: 1rem; font-family: var(--font-display); font-weight: 600; margin-bottom: 0.75rem; }
	.breakdown-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
	.breakdown-table th {
		text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border);
		color: var(--text-muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase;
		letter-spacing: 0.05em; font-family: var(--font-body);
	}
	.breakdown-table td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); }
	.progress-bar {
		position: relative; height: 20px; background: rgba(255,255,255,0.05);
		border-radius: 3px; overflow: hidden; min-width: 100px;
	}
	.progress-fill { height: 100%; border-radius: 3px; opacity: 0.7; transition: width 0.3s ease; }
	.progress-label {
		position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
		font-size: 0.7rem; font-weight: 600; font-family: var(--font-mono); color: var(--text);
	}
	.chart-card { margin-bottom: 1.5rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4); }
	.chart-card h2 { font-size: 1rem; font-family: var(--font-display); font-weight: 600; margin-bottom: 0.75rem; }
	.timeline-svg { width: 100%; max-height: 220px; }

	/* Pagination */
	.pagination {
		display: flex;
		justify-content: center;
		align-items: center;
		gap: 0.35rem;
		margin-top: 1.25rem;
		padding-bottom: 1rem;
	}
	.page-btn {
		padding: 0.35rem 0.7rem;
		background: var(--bg-card);
		border: 1px solid var(--border);
		color: var(--text-muted);
		font-size: 0.8rem;
		font-family: var(--font-mono);
		cursor: pointer;
		transition: all 0.15s ease;
		border-radius: 4px;
	}
	.page-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
	.page-btn:disabled { opacity: 0.3; cursor: default; }
	.page-active {
		background: var(--accent);
		color: #0d1117;
		border-color: var(--accent);
		font-weight: 600;
	}
	.page-dots { color: var(--text-muted); font-size: 0.8rem; padding: 0 0.25rem; }
</style>
