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
</script>

<div class="page stagger">
	<div class="header">
		<h1>captures</h1>
		<p class="subtitle">today ({data.today}) &mdash; best per difficulty from DB</p>
	</div>

	{#if Object.keys(data.byDifficulty).length === 0}
		<div class="card empty">No captures or runs found for today.</div>
	{/if}

	{#each Object.entries(data.byDifficulty) as [difficulty, info]}
		<section class="diff-section">
			<h2 class="diff-title" style="color: {diffColors[difficulty] || '#c9d1d9'}">
				{diffLabels[difficulty] || difficulty}
			</h2>

			<div class="day-card card today">
				<!-- Best run -->
				{#if info.bestRun}
					<div class="day-header">
						<div class="day-date">
							<span class="today-badge">best run</span>
							<span class="run-meta">
								run #{info.bestRun.id} &middot; {info.bestRun.runType} &middot; {info.bestRun.botCount} bots &middot; seed {info.bestRun.seed}
							</span>
						</div>
						<div class="day-meta">
							<span class="score" style="color: {diffColors[difficulty] || '#c9d1d9'}">
								{info.bestRun.score} pts
							</span>
						</div>
					</div>
					<div class="stats-row">
						<span class="stat"><b>{info.bestRun.itemsDelivered}</b> items delivered</span>
						<span class="stat"><b>{info.bestRun.ordersCompleted}</b> orders completed</span>
						{#if info.gpuScore != null}
							<span class="stat"><b>{info.gpuScore}</b> GPU plan score</span>
						{/if}
					</div>
				{:else if info.gpuScore != null}
					<div class="day-header">
						<span class="today-badge">gpu only</span>
						<span class="score" style="color: {diffColors[difficulty] || '#c9d1d9'}">
							{info.gpuScore} pts (plan)
						</span>
					</div>
				{/if}

				<!-- Orders -->
				<div class="orders-header">
					<span class="order-count" style="color: {diffColors[difficulty] || '#c9d1d9'}">
						{info.totalOrders} orders discovered
					</span>
				</div>

				{#if info.captureOrders.length}
					<div class="orders-grid">
						{#each info.captureOrders as order, i}
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
				{:else}
					<div class="no-orders">No orders in capture data</div>
				{/if}
			</div>
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
	.run-meta {
		color: var(--text-muted);
		font-size: 0.72rem;
	}
	.day-meta {
		display: flex;
		align-items: center;
		gap: 1rem;
	}
	.score {
		font-weight: 700;
		font-size: 1.1rem;
	}

	.stats-row {
		display: flex;
		gap: 1.5rem;
		margin-bottom: 0.75rem;
		padding: 0.4rem 0.5rem;
		background: rgba(0, 0, 0, 0.15);
		border-radius: 2px;
	}
	.stat {
		color: var(--text-muted);
		font-size: 0.72rem;
	}
	.stat b {
		color: var(--text);
		font-weight: 600;
	}

	.orders-header {
		margin-bottom: 0.5rem;
	}
	.order-count {
		font-weight: 700;
		font-size: 0.85rem;
	}

	.no-orders {
		color: var(--text-muted);
		font-size: 0.72rem;
		padding: 0.5rem;
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
