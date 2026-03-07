<script>
	let { data } = $props();

	const diffColors = {
		nightmare: '#a855f7',
		expert: '#d29922',
		hard: '#f472b6',
		medium: '#58a6ff',
		easy: '#39d353'
	};

	const diffLabels = {
		nightmare: 'Nightmare',
		expert: 'Expert',
		hard: 'Hard',
		medium: 'Medium',
		easy: 'Easy'
	};

	function formatDate(dateStr) {
		const d = new Date(dateStr + 'T00:00:00');
		const days = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat'];
		const months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'];
		return `${days[d.getDay()]} ${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}`;
	}

	function isToday(dateStr) {
		const today = new Date().toISOString().slice(0, 10);
		return dateStr === today;
	}

	function formatTime(isoStr) {
		if (!isoStr) return '—';
		const d = new Date(isoStr);
		return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
	}

	function timeSince(isoStr) {
		if (!isoStr) return '';
		const diff = Date.now() - new Date(isoStr).getTime();
		const mins = Math.floor(diff / 60000);
		if (mins < 60) return `${mins}m ago`;
		const hrs = Math.floor(mins / 60);
		if (hrs < 24) return `${hrs}h ago`;
		const days = Math.floor(hrs / 24);
		return `${days}d ago`;
	}
</script>

<div class="page stagger">
	<div class="header">
		<h1>solutions</h1>
		<p class="subtitle">GPU-optimized action sequences per difficulty per day</p>
		{#if data.total > 0}
			<span class="total-badge">{data.total} solution{data.total !== 1 ? 's' : ''}</span>
		{/if}
	</div>

	{#if data.error}
		<div class="card error-card">
			<span class="error-icon">!</span>
			DB error: {data.error}
		</div>
	{/if}

	{#if Object.keys(data.byDifficulty).length === 0 && !data.error}
		<div class="card empty">No solutions found. Run GPU optimization to generate action sequences.</div>
	{/if}

	{#each Object.entries(data.byDifficulty) as [difficulty, solutions]}
		<section class="diff-section">
			<h2 class="diff-title" style="color: {diffColors[difficulty] || '#c9d1d9'}">
				{diffLabels[difficulty] || difficulty}
				<span class="diff-count">{solutions.length}</span>
			</h2>

			{#each solutions as sol}
				<div class="sol-card card" class:today={isToday(sol.date)}>
					<div class="sol-header">
						<div class="sol-date">
							<span class="date-text">{formatDate(sol.date)}</span>
							{#if isToday(sol.date)}
								<span class="today-badge">today</span>
							{/if}
						</div>
						<div class="sol-score" style="color: {diffColors[difficulty] || '#c9d1d9'}">
							{sol.score} pts
						</div>
					</div>

					<div class="sol-details">
						<div class="detail-row">
							<span class="detail-label">bots</span>
							<span class="detail-value">{sol.num_bots}</span>
						</div>
						<div class="detail-row">
							<span class="detail-label">rounds</span>
							<span class="detail-value">{sol.num_rounds}</span>
						</div>
						<div class="detail-row">
							<span class="detail-label">optimizations</span>
							<span class="detail-value">{sol.optimizations_run}</span>
						</div>
						<div class="detail-row">
							<span class="detail-label">capture</span>
							<span class="detail-value hash">{sol.capture_hash || '—'}</span>
						</div>
						{#if sol.seed}
							<div class="detail-row">
								<span class="detail-label">seed</span>
								<span class="detail-value">{sol.seed}</span>
							</div>
						{/if}
					</div>

					<div class="sol-footer">
						<span class="time-info" title={sol.created_at}>
							created {formatTime(sol.created_at)}
						</span>
						{#if sol.updated_at}
							<span class="time-info" title={sol.updated_at}>
								updated {timeSince(sol.updated_at)}
							</span>
						{/if}
					</div>
				</div>
			{/each}
		</section>
	{/each}
</div>

<style>
	.page {
		max-width: 900px;
		margin: 0 auto;
	}

	.header {
		margin-bottom: 1.5rem;
		display: flex;
		align-items: baseline;
		gap: 0.75rem;
		flex-wrap: wrap;
	}
	.header h1 {
		font-size: 1.4rem;
		font-weight: 700;
		color: var(--orange);
		letter-spacing: 0.02em;
	}
	.subtitle {
		color: var(--text-muted);
		font-size: 0.75rem;
		margin-top: 0.25rem;
		flex-basis: 100%;
	}
	.total-badge {
		background: rgba(57, 211, 83, 0.1);
		color: var(--accent);
		padding: 0.15rem 0.5rem;
		border-radius: 2px;
		font-size: 0.7rem;
		font-weight: 600;
		border: 1px solid rgba(57, 211, 83, 0.15);
	}

	.error-card {
		color: #f85149;
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.8rem;
	}
	.error-icon {
		background: #f85149;
		color: #000;
		width: 1.2rem;
		height: 1.2rem;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 2px;
		font-weight: 800;
		font-size: 0.7rem;
	}

	.empty {
		color: var(--text-muted);
		text-align: center;
		padding: 3rem;
	}

	.diff-section {
		margin-bottom: 2rem;
	}

	.diff-title {
		font-size: 1rem;
		font-weight: 700;
		margin-bottom: 0.75rem;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.diff-count {
		background: rgba(255, 255, 255, 0.08);
		padding: 0.1rem 0.4rem;
		border-radius: 2px;
		font-size: 0.65rem;
		color: var(--text-muted);
	}

	.sol-card {
		margin-bottom: 0.75rem;
	}
	.sol-card.today {
		border-color: var(--orange);
		box-shadow: 0 0 12px rgba(210, 153, 34, 0.15);
	}

	.sol-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.6rem;
	}
	.sol-date {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
	.date-text {
		font-weight: 600;
		color: var(--text);
		font-size: 0.85rem;
	}
	.today-badge {
		background: var(--orange);
		color: #000;
		padding: 0.1rem 0.4rem;
		border-radius: 2px;
		font-size: 0.65rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}
	.sol-score {
		font-weight: 800;
		font-size: 1.1rem;
		letter-spacing: 0.02em;
	}

	.sol-details {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
		gap: 0.35rem 1rem;
		margin-bottom: 0.6rem;
	}
	.detail-row {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.25rem 0.5rem;
		background: rgba(0, 0, 0, 0.2);
		border-radius: 2px;
		border-left: 2px solid var(--border);
	}
	.detail-label {
		color: var(--text-muted);
		font-size: 0.7rem;
	}
	.detail-value {
		font-weight: 600;
		font-size: 0.8rem;
		color: var(--text);
	}
	.detail-value.hash {
		font-family: var(--font-mono, monospace);
		font-size: 0.7rem;
		color: var(--text-muted);
	}

	.sol-footer {
		display: flex;
		gap: 1.5rem;
		font-size: 0.65rem;
		color: var(--text-muted);
	}
	.time-info {
		cursor: default;
	}
</style>
