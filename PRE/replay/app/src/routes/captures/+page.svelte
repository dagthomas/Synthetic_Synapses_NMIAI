<script>
	let { data } = $props();

	const diffColors = {
		nightmare: '#f85149',
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
</script>

<div class="page stagger">
	<div class="header">
		<h1>captures</h1>
		<p class="subtitle">best captured orders per difficulty per day</p>
	</div>

	{#if Object.keys(data.byDifficulty).length === 0}
		<div class="card empty">No capture files found.</div>
	{/if}

	{#each Object.entries(data.byDifficulty) as [difficulty, days]}
		<section class="diff-section">
			<h2 class="diff-title" style="color: {diffColors[difficulty] || '#c9d1d9'}">
				{diffLabels[difficulty] || difficulty}
			</h2>

			{#each days as day}
				<div class="day-card card" class:today={isToday(day.date)}>
					<div class="day-header">
						<div class="day-date">
							<span class="date-text">{formatDate(day.date)}</span>
							{#if isToday(day.date)}
								<span class="today-badge">today</span>
							{/if}
						</div>
						<div class="day-meta">
							<span class="order-count" style="color: {diffColors[difficulty] || '#c9d1d9'}">
								{day.best?.total || 0} orders
							</span>
							<span class="file-count">{day.fileCount} capture{day.fileCount !== 1 ? 's' : ''}</span>
						</div>
					</div>

					<div class="best-file">
						<span class="file-label">best:</span>
						<code>{day.best?.filename || '?'}</code>
					</div>

					{#if day.best?.orders?.length}
						<div class="orders-grid">
							{#each day.best.orders as order, i}
								<div class="order-row">
									<span class="order-id">#{i}</span>
									<div class="order-items">
										{#each order.items as item}
											<span class="item-tag">{item}</span>
										{/each}
									</div>
									<span class="item-count">{order.items.length} items</span>
								</div>
							{/each}
						</div>
					{/if}
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
	}

	.day-card {
		margin-bottom: 0.75rem;
	}
	.day-card.today {
		border-color: var(--orange);
		box-shadow: 0 0 12px rgba(210, 153, 34, 0.15);
	}

	.day-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.5rem;
	}
	.day-date {
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
	.day-meta {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	.order-count {
		font-weight: 700;
		font-size: 0.9rem;
	}
	.file-count {
		color: var(--text-muted);
		font-size: 0.72rem;
	}

	.best-file {
		margin-bottom: 0.75rem;
		font-size: 0.72rem;
	}
	.file-label {
		color: var(--text-muted);
	}
	.best-file code {
		color: var(--text-muted);
		background: rgba(0, 0, 0, 0.3);
		padding: 0.1rem 0.35rem;
		border-radius: 2px;
		font-size: 0.7rem;
	}

	.orders-grid {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
	}

	.order-row {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		padding: 0.35rem 0.5rem;
		background: rgba(0, 0, 0, 0.2);
		border-radius: 2px;
		border-left: 2px solid var(--border);
	}

	.order-id {
		color: var(--text-muted);
		font-size: 0.7rem;
		min-width: 1.5rem;
		font-weight: 600;
	}

	.order-items {
		display: flex;
		flex-wrap: wrap;
		gap: 0.25rem;
		flex: 1;
	}

	.item-tag {
		background: rgba(57, 211, 83, 0.1);
		color: var(--accent);
		padding: 0.1rem 0.4rem;
		border-radius: 2px;
		font-size: 0.7rem;
		border: 1px solid rgba(57, 211, 83, 0.15);
	}

	.item-count {
		color: var(--text-muted);
		font-size: 0.65rem;
		min-width: 3.5rem;
		text-align: right;
	}
</style>
