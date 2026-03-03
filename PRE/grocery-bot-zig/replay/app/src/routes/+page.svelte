<script>
	let { data } = $props();

	const diffColors = {
		easy: '#00b894',
		medium: '#fdcb6e',
		hard: '#e17055',
		expert: '#e74c3c',
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
</script>

<div class="header">
	<h1>Game Runs</h1>
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
	</div>
</div>

{#if data.stats.length > 0}
<div class="stats-row">
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

<style>
	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 1.5rem;
		flex-wrap: wrap;
		gap: 1rem;
	}
	h1 { font-size: 1.5rem; font-weight: 700; }
	.filters { display: flex; gap: 0.5rem; }
	.chip {
		padding: 0.35rem 0.75rem;
		background: var(--bg-card);
		border: 1px solid var(--border);
		color: var(--text-muted);
		text-transform: capitalize;
		font-size: 0.8rem;
	}
	.chip.active {
		background: var(--chip-color, var(--accent))22;
		color: var(--chip-color, var(--accent-light));
		border-color: var(--chip-color, var(--accent));
	}
	.stats-row {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
		gap: 1rem;
		margin-bottom: 1.5rem;
	}
	.stat-card { text-align: center; }
	.total-card { border: 1px solid var(--accent); background: var(--accent)08; }
	.stat-label { font-size: 0.75rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
	.stat-main { font-size: 2rem; font-weight: 800; color: var(--text); }
	.stat-sub { font-size: 0.75rem; color: var(--text-muted); }
	.table-wrap { overflow-x: auto; padding: 0; }
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
	}
	td {
		padding: 0.6rem 1rem;
		border-bottom: 1px solid var(--border);
	}
	tr:hover td { background: var(--bg-hover); }
	.mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.8rem; }
	.muted { color: var(--text-muted); font-size: 0.8rem; }
	.score { font-weight: 700; font-size: 1rem; color: var(--green); }
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 600;
		text-transform: capitalize;
	}
	.view-btn {
		padding: 0.3rem 0.75rem;
		background: var(--accent);
		color: white;
		border-radius: var(--radius);
		font-size: 0.8rem;
		font-weight: 500;
	}
	.view-btn:hover { background: var(--accent-light); text-decoration: none; }
	.empty { padding: 3rem; text-align: center; color: var(--text-muted); }
	code { background: var(--bg); padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85em; }
	.filter-sep { color: var(--border); padding: 0 0.25rem; align-self: center; }
	.type-badge { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
	.type-live { background: #0984e320; color: #0984e3; border: 1px solid #0984e344; }
	.type-replay { background: #6c5ce720; color: #a29bfe; border: 1px solid #6c5ce744; }
	.chip.type-live.active { background: #0984e322; color: #0984e3; border-color: #0984e3; }
	.chip.type-replay.active { background: #6c5ce722; color: #a29bfe; border-color: #6c5ce7; }
</style>
