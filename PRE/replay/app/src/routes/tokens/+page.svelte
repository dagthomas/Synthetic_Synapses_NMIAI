<script>
	let { data } = $props();
	let tokens = $state(data.tokens);
	let wsInput = $state('');
	let labelInput = $state('');
	let saving = $state(false);
	let error = $state('');
	let filter = $state('all');

	const diffColors = {
		easy: '#39d353',
		medium: '#d29922',
		hard: '#f85149',
		expert: '#da3633',
	};

	const ALL_TYPES = [
		'milk', 'bread', 'eggs', 'butter', 'cheese', 'pasta', 'rice', 'juice',
		'yogurt', 'cereal', 'flour', 'sugar', 'coffee', 'tea', 'oil', 'salt',
	];

	function parseTokenPreview(url) {
		try {
			const tokenMatch = url.match(/[?&]token=([^&]+)/);
			if (!tokenMatch) return null;
			const parts = tokenMatch[1].split('.');
			if (parts.length < 2) return null;
			const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
			return JSON.parse(atob(payload));
		} catch {
			return null;
		}
	}

	let preview = $derived(wsInput.trim() ? parseTokenPreview(wsInput) : null);

	let filtered = $derived(
		filter === 'all' ? tokens : tokens.filter(t => t.difficulty === filter)
	);

	// ── Token CRUD ───────────────────────────────────────────────────────────
	async function saveToken() {
		if (!wsInput.trim()) return;
		saving = true;
		error = '';
		try {
			const res = await fetch('/api/tokens', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ ws_url: wsInput.trim(), label: labelInput.trim() || null }),
			});
			const row = await res.json();
			if (!res.ok) { error = row.error; return; }
			tokens = [row, ...tokens.filter(t => t.id !== row.id)];
			labelInput = '';
		} catch (e) {
			error = e.message;
		} finally {
			saving = false;
		}
	}

	async function deleteToken(id) {
		try {
			await fetch('/api/tokens', {
				method: 'DELETE',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ id }),
			});
			tokens = tokens.filter(t => t.id !== id);
		} catch (e) {
			error = e.message;
		}
	}

	async function deleteAll() {
		const toDelete = filter === 'all' ? tokens : tokens.filter(t => t.difficulty === filter);
		for (const t of toDelete) {
			await fetch('/api/tokens', {
				method: 'DELETE',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ id: t.id }),
			});
		}
		tokens = tokens.filter(t => !toDelete.find(d => d.id === t.id));
	}

	function copyUrl(url) {
		navigator.clipboard.writeText(url);
	}

	function timeAgo(dateStr) {
		const diff = Date.now() - new Date(dateStr).getTime();
		const mins = Math.floor(diff / 60000);
		if (mins < 1) return 'just now';
		if (mins < 60) return `${mins}m ago`;
		const hours = Math.floor(mins / 60);
		if (hours < 24) return `${hours}h ago`;
		return `${Math.floor(hours / 24)}d ago`;
	}

	function isExpired(dateStr) {
		return Date.now() - new Date(dateStr).getTime() > 288000;
	}

	let counts = $derived.by(() => {
		const c = { all: tokens.length, easy: 0, medium: 0, hard: 0, expert: 0, unknown: 0 };
		for (const t of tokens) {
			if (t.difficulty && c[t.difficulty] !== undefined) c[t.difficulty]++;
			else c.unknown++;
		}
		return c;
	});

	// ── Live Inspector ───────────────────────────────────────────────────────
	let inspecting = $state(false);
	let inspectUrl = $state('');
	let inspectTokenId = $state(null);
	let abortController = $state(null);
	let sessionId = $state(null);
	let wsStatus = $state('idle'); // idle | connecting | connected | finished | error
	let wsRound = $state(0);
	let wsScore = $state(0);
	let wsFinalScore = $state(null);
	let wsMessages = $state([]);
	let wsDifficulty = $state(null);
	let wsMapSeed = $state(null);

	// Structured game data extracted from messages
	let gameGrid = $state(null);
	let gameItems = $state([]);
	let gameBots = $state([]);
	let gameOrders = $state([]);
	let gameDropOff = $state(null);
	let gameSpawn = $state(null);
	let allOrdersSeen = $state([]);
	let seenOrderIds = $state(new Set());

	// Items grouped by type for the grocery list
	let itemsByType = $derived.by(() => {
		const map = {};
		for (const item of gameItems) {
			const t = item.type;
			if (!map[t]) map[t] = [];
			map[t].push(item);
		}
		return Object.entries(map).sort((a, b) => a[0].localeCompare(b[0]));
	});

	// Active/preview order needs
	let activeOrder = $derived(gameOrders.find(o => o.status === 'active'));
	let previewOrder = $derived(gameOrders.find(o => o.status === 'preview'));

	// Inspector view tab
	let inspectTab = $state('orders'); // orders | items | bots | grid | raw

	function connectToWs(url, tokenId) {
		if (inspecting) disconnectWs();
		inspectUrl = url;
		inspectTokenId = tokenId;
		inspecting = true;
		wsStatus = 'connecting';
		wsRound = 0;
		wsScore = 0;
		wsFinalScore = null;
		wsMessages = [];
		gameGrid = null;
		gameItems = [];
		gameBots = [];
		gameOrders = [];
		gameDropOff = null;
		gameSpawn = null;
		allOrdersSeen = [];
		seenOrderIds = new Set();
		sessionId = null;
		wsDifficulty = null;
		wsMapSeed = null;

		abortController = new AbortController();

		(async () => {
			try {
				const res = await fetch('/api/tokens/connect', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ ws_url: url, token_id: tokenId }),
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
							const event = JSON.parse(line.slice(6));
							handleInspectEvent(event);
						} catch {}
					}
				}
			} catch (e) {
				if (e.name !== 'AbortError') {
					wsStatus = 'error';
					wsMessages = [...wsMessages, { type: 'error', text: e.message, t: Date.now() }];
				}
			}
			inspecting = false;
		})();
	}

	function disconnectWs() {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		inspecting = false;
		wsStatus = 'disconnected';
	}

	function handleInspectEvent(event) {
		switch (event.type) {
			case 'session':
				sessionId = event.session_id;
				wsDifficulty = event.difficulty;
				wsMapSeed = event.map_seed;
				break;
			case 'connected':
				wsStatus = 'connected';
				break;
			case 'ws_message': {
				const msg = event.data;
				const msgType = event.msg_type;

				// Keep last 500 raw messages
				wsMessages = [...wsMessages.slice(-499), {
					seq: event.seq,
					type: msgType,
					round: event.round,
					data: msg,
					t: Date.now(),
				}];

				if (msgType === 'game_state' || event.round !== null) {
					wsRound = msg.round ?? wsRound;
					wsScore = msg.score ?? wsScore;

					// Round 0: extract grid, items, map structure
					if (msg.round === 0) {
						gameGrid = msg.grid || null;
						gameItems = msg.items || [];
						gameDropOff = msg.drop_off || null;
						if (msg.grid) {
							const w = msg.grid.width || 0;
							const h = msg.grid.height || 0;
							gameSpawn = [w - 2, h - 2];
						}
					}

					// Every round: update bots and orders
					gameBots = msg.bots || gameBots;
					gameOrders = msg.orders || gameOrders;

					// Track all orders seen
					for (const order of (msg.orders || [])) {
						if (!seenOrderIds.has(order.id)) {
							seenOrderIds = new Set([...seenOrderIds, order.id]);
							allOrdersSeen = [...allOrdersSeen, {
								id: order.id,
								items_required: order.items_required,
								status: order.status,
								first_seen_round: msg.round,
							}];
						}
					}
				}
				break;
			}
			case 'game_over':
				wsFinalScore = event.score;
				wsStatus = 'finished';
				break;
			case 'ws_closed':
				if (wsStatus !== 'finished') wsStatus = 'disconnected';
				break;
			case 'error':
				wsStatus = 'error';
				wsMessages = [...wsMessages, { type: 'error', text: event.message, t: Date.now() }];
				break;
			case 'heartbeat':
				break;
		}
	}

	// Save+Connect in one action
	async function saveAndConnect() {
		if (!wsInput.trim()) return;
		saving = true;
		error = '';
		try {
			const res = await fetch('/api/tokens', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ ws_url: wsInput.trim(), label: labelInput.trim() || null }),
			});
			const row = await res.json();
			if (!res.ok) { error = row.error; saving = false; return; }
			tokens = [row, ...tokens.filter(t => t.id !== row.id)];
			labelInput = '';
			saving = false;
			connectToWs(row.ws_url, row.id);
		} catch (e) {
			error = e.message;
			saving = false;
		}
	}

	// Auto-scroll raw log
	let rawLogEl = $state(null);
	$effect(() => {
		if (rawLogEl && wsMessages.length) {
			rawLogEl.scrollTop = rawLogEl.scrollHeight;
		}
	});
</script>

<div class="tokens-page stagger">
	<div class="page-header">
		<h1>WebSocket Inspector</h1>
		<span class="token-count">{tokens.length} stored</span>
		{#if inspecting}
			<span class="live-dot"></span>
			<span class="live-label">LIVE</span>
		{/if}
	</div>

	<!-- Input section -->
	<div class="input-section card">
		<div class="input-row">
			<input
				type="text"
				bind:value={wsInput}
				placeholder="wss://game.ainm.no/ws?token=..."
				class="url-input"
				disabled={inspecting}
				onkeydown={(e) => e.key === 'Enter' && saveAndConnect()}
			/>
			<input
				type="text"
				bind:value={labelInput}
				placeholder="label"
				class="label-input"
				disabled={inspecting}
				onkeydown={(e) => e.key === 'Enter' && saveAndConnect()}
			/>
			{#if inspecting}
				<button class="stop-btn" onclick={disconnectWs}>Disconnect</button>
			{:else}
				<button class="save-btn" onclick={saveToken} disabled={saving || !wsInput.trim()}>
					Save
				</button>
				<button class="connect-btn" onclick={saveAndConnect} disabled={saving || !wsInput.trim()}>
					Connect
				</button>
			{/if}
		</div>

		{#if error}
			<div class="error-msg">{error}</div>
		{/if}

		{#if preview && !inspecting}
			<div class="preview-bar">
				<span class="preview-label">Detected:</span>
				{#if preview.difficulty}
					<span class="diff-badge" style="color: {diffColors[preview.difficulty] || 'var(--text)'}">
						{preview.difficulty.toUpperCase()}
					</span>
				{/if}
				{#if preview.map_seed}
					<span class="seed-badge">seed #{preview.map_seed}</span>
				{/if}
				{#if preview.exp}
					<span class="exp-badge" class:expired={Date.now() / 1000 > preview.exp}>
						{Date.now() / 1000 > preview.exp ? 'EXPIRED' : `expires in ${Math.max(0, Math.round(preview.exp - Date.now() / 1000))}s`}
					</span>
				{/if}
			</div>
		{/if}
	</div>

	<!-- ── LIVE INSPECTOR ────────────────────────────────────────────────── -->
	{#if inspecting || wsStatus === 'finished' || wsStatus === 'disconnected'}
		<div class="inspector">
			<!-- Status bar -->
			<div class="inspector-status card">
				<div class="status-row">
					<span class="status-dot" class:dot-connecting={wsStatus === 'connecting'}
						class:dot-connected={wsStatus === 'connected'}
						class:dot-finished={wsStatus === 'finished'}
						class:dot-error={wsStatus === 'error'}
						class:dot-disconnected={wsStatus === 'disconnected'}></span>
					<span class="status-text">{wsStatus.toUpperCase()}</span>
					{#if wsDifficulty}
						<span class="status-diff" style="color: {diffColors[wsDifficulty]}">{wsDifficulty.toUpperCase()}</span>
					{/if}
					{#if wsMapSeed}
						<span class="status-seed">#{wsMapSeed}</span>
					{/if}
					<span class="status-round">R{wsRound}/300</span>
					<span class="status-score">Score: <strong>{wsScore}</strong></span>
					{#if wsFinalScore !== null}
						<span class="status-final">Final: {wsFinalScore}</span>
					{/if}
					<span class="status-msgs">{wsMessages.length} msgs</span>
					{#if sessionId}
						<span class="status-session">session #{sessionId}</span>
					{/if}
				</div>
				<div class="progress-bar">
					<div class="progress-fill" style="width: {wsRound / 300 * 100}%"></div>
				</div>
			</div>

			<!-- Tab bar -->
			<div class="tab-bar">
				{#each [['orders', `Orders (${allOrdersSeen.length})`], ['items', `Items (${gameItems.length})`], ['bots', `Bots (${gameBots.length})`], ['grid', 'Grid'], ['raw', `Raw (${wsMessages.length})`]] as [key, label]}
					<button class="tab" class:active={inspectTab === key} onclick={() => inspectTab = key}>
						{label}
					</button>
				{/each}
			</div>

			<!-- Tab content -->
			<div class="tab-content">
				{#if inspectTab === 'orders'}
					<!-- Active & Preview orders -->
					<div class="orders-panel">
						{#if activeOrder}
							<div class="order-card active-order">
								<div class="order-head">
									<span class="order-badge active">ACTIVE</span>
									<span class="order-id">#{activeOrder.id}</span>
									<span class="order-progress-text">{activeOrder.items_delivered?.length || 0}/{activeOrder.items_required.length}</span>
								</div>
								<div class="order-items-list">
									{#each activeOrder.items_required as item, i}
										<span class="order-item-tag" class:delivered={activeOrder.items_delivered?.includes(item)}>
											{item}
										</span>
									{/each}
								</div>
							</div>
						{/if}
						{#if previewOrder}
							<div class="order-card preview-order">
								<div class="order-head">
									<span class="order-badge preview">PREVIEW</span>
									<span class="order-id">#{previewOrder.id}</span>
								</div>
								<div class="order-items-list">
									{#each previewOrder.items_required as item}
										<span class="order-item-tag">{item}</span>
									{/each}
								</div>
							</div>
						{/if}

						<!-- All orders seen history -->
						<h3 class="section-title">All Orders Discovered ({allOrdersSeen.length})</h3>
						<div class="orders-table">
							<div class="orders-header">
								<span>ID</span>
								<span>Round</span>
								<span>Items</span>
							</div>
							{#each allOrdersSeen as order}
								<div class="orders-row" class:is-active={order.id === activeOrder?.id} class:is-preview={order.id === previewOrder?.id}>
									<span class="mono">#{order.id}</span>
									<span class="mono">R{order.first_seen_round}</span>
									<span class="order-items-inline">
										{#each order.items_required as item}
											<span class="item-chip">{item}</span>
										{/each}
									</span>
								</div>
							{/each}
						</div>
					</div>

				{:else if inspectTab === 'items'}
					<!-- Grocery list: items grouped by type -->
					<div class="items-panel">
						<h3 class="section-title">Grocery List ({gameItems.length} items, {itemsByType.length} types)</h3>
						<div class="grocery-list">
							{#each itemsByType as [typeName, items]}
								<div class="grocery-type">
									<div class="grocery-type-header">
										<span class="grocery-type-name">{typeName}</span>
										<span class="grocery-type-count">{items.length}x</span>
									</div>
									<div class="grocery-positions">
										{#each items as item}
											<span class="pos-chip" title="id: {item.id}">
												({item.position[0]},{item.position[1]})
											</span>
										{/each}
									</div>
								</div>
							{/each}
						</div>

						{#if activeOrder || previewOrder}
							<h3 class="section-title" style="margin-top: 1rem">Needed Items</h3>
							<div class="needed-list">
								{#if activeOrder}
									<div class="needed-group">
										<span class="needed-label active">Active needs:</span>
										{#each activeOrder.items_required as item}
											{@const delivered = activeOrder.items_delivered?.includes(item)}
											<span class="needed-chip" class:done={delivered}>{item}</span>
										{/each}
									</div>
								{/if}
								{#if previewOrder}
									<div class="needed-group">
										<span class="needed-label preview">Preview needs:</span>
										{#each previewOrder.items_required as item}
											<span class="needed-chip">{item}</span>
										{/each}
									</div>
								{/if}
							</div>
						{/if}
					</div>

				{:else if inspectTab === 'bots'}
					<div class="bots-panel">
						{#each gameBots as bot}
							<div class="bot-card">
								<div class="bot-head">
									<span class="bot-id">Bot {bot.id}</span>
									<span class="bot-pos mono">({bot.position[0]}, {bot.position[1]})</span>
								</div>
								<div class="bot-inv">
									{#if bot.inventory?.length}
										{#each bot.inventory as item}
											<span class="inv-chip">{item}</span>
										{/each}
									{:else}
										<span class="empty-inv">empty</span>
									{/if}
								</div>
							</div>
						{/each}
						{#if gameBots.length === 0}
							<div class="empty-panel">Waiting for game data...</div>
						{/if}
					</div>

				{:else if inspectTab === 'grid'}
					<div class="grid-panel">
						{#if gameGrid}
							<div class="grid-info">
								<span>Size: {gameGrid.width}x{gameGrid.height}</span>
								<span>Walls: {gameGrid.walls?.length || 0}</span>
								<span>Items: {gameItems.length}</span>
								{#if gameDropOff}
									<span>Dropoff: ({gameDropOff[0]},{gameDropOff[1]})</span>
								{/if}
								{#if gameSpawn}
									<span>Spawn: ({gameSpawn[0]},{gameSpawn[1]})</span>
								{/if}
							</div>
							<pre class="grid-json">{JSON.stringify(gameGrid, null, 2)}</pre>
						{:else}
							<div class="empty-panel">Waiting for round 0 data...</div>
						{/if}
					</div>

				{:else if inspectTab === 'raw'}
					<div class="raw-panel" bind:this={rawLogEl}>
						{#each wsMessages as msg}
							<div class="raw-msg" class:raw-error={msg.type === 'error'}>
								<span class="raw-seq">#{msg.seq ?? '-'}</span>
								<span class="raw-type">{msg.type}</span>
								{#if msg.round !== null && msg.round !== undefined}
									<span class="raw-round">R{msg.round}</span>
								{/if}
								{#if msg.data}
									<span class="raw-preview">{JSON.stringify(msg.data).slice(0, 200)}</span>
								{:else if msg.text}
									<span class="raw-preview">{msg.text}</span>
								{/if}
							</div>
						{/each}
						{#if wsMessages.length === 0}
							<div class="empty-panel">No messages yet...</div>
						{/if}
					</div>
				{/if}
			</div>
		</div>
	{/if}

	<!-- ── STORED TOKENS ─────────────────────────────────────────────────── -->
	{#if !inspecting && wsStatus === 'idle'}
		<div class="filter-bar">
			<div class="filter-tabs">
				{#each ['all', 'easy', 'medium', 'hard', 'expert'] as d}
					<button
						class="filter-tab"
						class:active={filter === d}
						style={d !== 'all' && filter === d ? `color: ${diffColors[d]}; border-color: ${diffColors[d]}` : ''}
						onclick={() => filter = d}
					>
						{d === 'all' ? 'All' : d.charAt(0).toUpperCase() + d.slice(1)}
						<span class="tab-count">{counts[d]}</span>
					</button>
				{/each}
			</div>
			{#if filtered.length > 0}
				<button class="delete-all-btn" onclick={deleteAll}>
					Delete {filter === 'all' ? 'all' : filter} ({filtered.length})
				</button>
			{/if}
		</div>

		<div class="token-list">
			{#each filtered as token (token.id)}
				<div class="token-row card" class:expired={isExpired(token.created_at)}>
					<div class="token-main">
						<div class="token-top">
							{#if token.difficulty}
								<span class="diff-tag" style="background: {diffColors[token.difficulty]}22; color: {diffColors[token.difficulty]}; border-color: {diffColors[token.difficulty]}44">
									{token.difficulty.toUpperCase()}
								</span>
							{:else}
								<span class="diff-tag unknown">UNKNOWN</span>
							{/if}
							{#if token.map_seed}
								<span class="seed-tag">#{token.map_seed}</span>
							{/if}
							{#if token.label}
								<span class="label-tag">{token.label}</span>
							{/if}
							<span class="time-tag" class:expired-time={isExpired(token.created_at)}>
								{timeAgo(token.created_at)}
								{#if isExpired(token.created_at)}
									<span class="expired-badge">expired</span>
								{/if}
							</span>
						</div>
						<div class="token-url">
							<code>{token.ws_url}</code>
						</div>
					</div>
					<div class="token-actions">
						<button class="connect-btn-sm" onclick={() => { wsInput = token.ws_url; connectToWs(token.ws_url, token.id); }}>
							inspect
						</button>
						<button class="copy-btn" onclick={() => copyUrl(token.ws_url)}>copy</button>
						<button class="del-btn" onclick={() => deleteToken(token.id)}>del</button>
					</div>
				</div>
			{:else}
				<div class="empty-state card">
					<p>No tokens {filter !== 'all' ? `for ${filter}` : 'saved yet'}. Paste a WebSocket URL above.</p>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.tokens-page { display: flex; flex-direction: column; gap: 1rem; max-width: 1400px; margin: 0 auto; }

	.page-header { display: flex; align-items: baseline; gap: 0.75rem; }
	.page-header h1 { font-size: 1.2rem; font-family: var(--font-mono); }
	.token-count { font-size: 0.75rem; color: var(--text-muted); }
	.live-dot {
		width: 8px; height: 8px; border-radius: 50%; background: #f85149;
		animation: livePulse 1s infinite; margin-left: 0.5rem;
	}
	.live-label { font-size: 0.7rem; font-weight: 700; color: #f85149; letter-spacing: 0.1em; }
	@keyframes livePulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

	/* Input */
	.input-section { display: flex; flex-direction: column; gap: 0.75rem; }
	.input-row { display: flex; gap: 0.5rem; align-items: center; }
	.url-input {
		flex: 1; padding: 0.6rem 0.75rem; background: var(--bg);
		border: 1px solid var(--border); border-radius: var(--radius-sm);
		color: var(--text); font-family: var(--font-mono); font-size: 0.8rem; outline: none;
	}
	.url-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(57, 211, 83, 0.1); }
	.label-input {
		width: 120px; padding: 0.6rem 0.75rem; background: var(--bg);
		border: 1px solid var(--border); border-radius: var(--radius-sm);
		color: var(--text); font-family: var(--font-mono); font-size: 0.8rem; outline: none;
	}
	.label-input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(57, 211, 83, 0.1); }
	.save-btn {
		padding: 0.6rem 1rem; background: var(--bg-card); color: var(--text);
		font-weight: 600; border: 1px solid var(--border); border-radius: var(--radius); white-space: nowrap;
	}
	.save-btn:hover:not(:disabled) { border-color: var(--accent); }
	.save-btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.connect-btn {
		padding: 0.6rem 1.2rem; background: var(--accent); color: #0d1117;
		font-weight: 600; border-radius: var(--radius); white-space: nowrap;
	}
	.connect-btn:hover:not(:disabled) { background: var(--accent-light); box-shadow: 0 0 12px rgba(57, 211, 83, 0.2); }
	.connect-btn:disabled { opacity: 0.5; cursor: not-allowed; }
	.stop-btn {
		padding: 0.6rem 1.2rem; background: var(--red); color: white;
		font-weight: 600; border-radius: var(--radius); white-space: nowrap;
	}
	.stop-btn:hover { opacity: 0.85; }

	.error-msg { color: var(--red); font-size: 0.75rem; }
	.preview-bar {
		display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0.75rem;
		background: rgba(1, 4, 9, 0.5); border-radius: 4px; font-size: 0.8rem;
	}
	.preview-label { color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; }
	.diff-badge { font-weight: 700; letter-spacing: 0.05em; }
	.seed-badge { font-family: var(--font-mono); color: var(--text-muted); font-size: 0.75rem; }
	.exp-badge { font-family: var(--font-mono); font-size: 0.75rem; color: var(--accent); }
	.exp-badge.expired { color: var(--red); font-weight: 600; }

	/* ── Inspector ── */
	.inspector { display: flex; flex-direction: column; gap: 0.75rem; }

	.inspector-status { padding: 0.75rem 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
	.status-row { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; font-size: 0.8rem; }
	.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
	.dot-connecting { background: var(--orange); animation: livePulse 1s infinite; }
	.dot-connected { background: var(--accent); animation: livePulse 1.5s infinite; }
	.dot-finished { background: var(--blue); }
	.dot-error { background: var(--red); }
	.dot-disconnected { background: var(--text-muted); }
	.status-text { font-weight: 700; font-family: var(--font-mono); }
	.status-diff { font-weight: 700; }
	.status-seed { font-family: var(--font-mono); color: var(--text-muted); }
	.status-round { font-family: var(--font-mono); }
	.status-score { font-family: var(--font-mono); }
	.status-score strong { color: var(--accent); }
	.status-final { color: var(--accent); font-weight: 700; font-family: var(--font-mono); }
	.status-msgs { color: var(--text-muted); font-family: var(--font-mono); font-size: 0.7rem; }
	.status-session { color: var(--text-muted); font-family: var(--font-mono); font-size: 0.7rem; margin-left: auto; }
	.progress-bar { width: 100%; height: 4px; background: var(--bg); border-radius: 2px; overflow: hidden; }
	.progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); border-radius: 2px; transition: width 0.3s; }

	/* Tabs */
	.tab-bar { display: flex; gap: 0.25rem; }
	.tab {
		padding: 0.4rem 0.85rem; background: transparent; border: 1px solid var(--border);
		border-radius: var(--radius); color: var(--text-muted); font-size: 0.75rem;
		font-family: var(--font-mono);
	}
	.tab:hover { border-color: #484f58; color: var(--text); }
	.tab.active { border-color: var(--accent); color: var(--accent); background: rgba(57, 211, 83, 0.05); }

	.tab-content {
		background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius);
		padding: 1rem; min-height: 300px; max-height: calc(100vh - 350px); overflow-y: auto;
	}

	.section-title {
		font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
		color: var(--text-muted); margin-bottom: 0.5rem; padding-bottom: 0.35rem;
		border-bottom: 1px solid rgba(48, 54, 61, 0.4);
	}
	.empty-panel { color: var(--text-muted); font-size: 0.8rem; padding: 1rem; text-align: center; }
	.mono { font-family: var(--font-mono); }

	/* Orders tab */
	.orders-panel { display: flex; flex-direction: column; gap: 0.75rem; }
	.order-card {
		padding: 0.75rem; border-radius: 6px; border: 1px solid var(--border);
	}
	.active-order { border-color: #facc1544; background: #facc1508; }
	.preview-order { border-color: #f472b644; background: #f472b608; }
	.order-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
	.order-badge {
		font-size: 0.6rem; font-weight: 700; text-transform: uppercase; padding: 0.1rem 0.35rem;
		border-radius: 3px; letter-spacing: 0.05em;
	}
	.order-badge.active { background: #facc1522; color: var(--yellow); }
	.order-badge.preview { background: #f472b622; color: var(--pink); }
	.order-id { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); }
	.order-progress-text { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); margin-left: auto; }
	.order-items-list { display: flex; flex-wrap: wrap; gap: 0.25rem; }
	.order-item-tag {
		font-size: 0.7rem; padding: 0.15rem 0.4rem; background: var(--bg);
		border: 1px solid var(--border); border-radius: 3px;
	}
	.order-item-tag.delivered {
		background: #39d35322; border-color: #39d35344; color: var(--green); text-decoration: line-through;
	}

	.orders-table { display: flex; flex-direction: column; gap: 0; }
	.orders-header {
		display: grid; grid-template-columns: 60px 60px 1fr; gap: 0.5rem;
		padding: 0.3rem 0.5rem; font-size: 0.65rem; color: var(--text-muted);
		text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border);
	}
	.orders-row {
		display: grid; grid-template-columns: 60px 60px 1fr; gap: 0.5rem;
		padding: 0.35rem 0.5rem; font-size: 0.75rem; border-bottom: 1px solid rgba(48, 54, 61, 0.3);
		transition: background 0.1s;
	}
	.orders-row:hover { background: var(--bg-hover); }
	.orders-row.is-active { background: #facc1508; }
	.orders-row.is-preview { background: #f472b608; }
	.order-items-inline { display: flex; flex-wrap: wrap; gap: 0.2rem; }
	.item-chip {
		font-size: 0.65rem; padding: 0.05rem 0.3rem; background: var(--bg);
		border: 1px solid var(--border); border-radius: 3px;
	}

	/* Items tab - grocery list */
	.items-panel { display: flex; flex-direction: column; gap: 0.5rem; }
	.grocery-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 0.5rem; }
	.grocery-type {
		padding: 0.6rem; background: rgba(1, 4, 9, 0.4); border: 1px solid var(--border);
		border-radius: 6px;
	}
	.grocery-type-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.35rem; }
	.grocery-type-name { font-weight: 600; font-size: 0.85rem; text-transform: capitalize; }
	.grocery-type-count {
		font-size: 0.7rem; font-family: var(--font-mono); color: var(--accent);
		background: rgba(57, 211, 83, 0.1); padding: 0.1rem 0.35rem; border-radius: 3px;
	}
	.grocery-positions { display: flex; flex-wrap: wrap; gap: 0.2rem; }
	.pos-chip {
		font-size: 0.6rem; font-family: var(--font-mono); padding: 0.05rem 0.25rem;
		background: var(--bg); border: 1px solid var(--border); border-radius: 2px;
		color: var(--text-muted);
	}

	.needed-list { display: flex; flex-direction: column; gap: 0.5rem; }
	.needed-group { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem; }
	.needed-label { font-size: 0.7rem; font-weight: 600; }
	.needed-label.active { color: var(--yellow); }
	.needed-label.preview { color: var(--pink); }
	.needed-chip {
		font-size: 0.7rem; padding: 0.1rem 0.35rem; background: var(--bg);
		border: 1px solid var(--border); border-radius: 3px;
	}
	.needed-chip.done {
		background: #39d35322; border-color: #39d35344; color: var(--green); text-decoration: line-through;
	}

	/* Bots tab */
	.bots-panel { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem; }
	.bot-card { padding: 0.6rem; background: rgba(1, 4, 9, 0.4); border: 1px solid var(--border); border-radius: 6px; }
	.bot-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
	.bot-id { font-weight: 700; font-size: 0.85rem; }
	.bot-pos { font-size: 0.75rem; color: var(--text-muted); }
	.bot-inv { display: flex; flex-wrap: wrap; gap: 0.2rem; }
	.inv-chip {
		font-size: 0.7rem; padding: 0.1rem 0.35rem; background: #39d35318;
		border: 1px solid #39d35333; border-radius: 3px; color: var(--accent-light);
	}
	.empty-inv { font-size: 0.7rem; color: var(--text-muted); font-style: italic; }

	/* Grid tab */
	.grid-panel { display: flex; flex-direction: column; gap: 0.75rem; }
	.grid-info { display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.8rem; font-family: var(--font-mono); }
	.grid-json {
		font-size: 0.65rem; font-family: var(--font-mono); background: rgba(1, 4, 9, 0.5);
		padding: 0.75rem; border-radius: 4px; overflow-x: auto; max-height: 400px;
		color: var(--text-muted); white-space: pre-wrap; word-break: break-all;
	}

	/* Raw tab */
	.raw-panel {
		max-height: calc(100vh - 400px); overflow-y: auto; display: flex; flex-direction: column; gap: 0;
	}
	.raw-msg {
		display: flex; align-items: baseline; gap: 0.5rem; padding: 0.2rem 0.35rem;
		border-bottom: 1px solid rgba(48, 54, 61, 0.2); font-size: 0.7rem;
	}
	.raw-msg:hover { background: var(--bg-hover); }
	.raw-msg.raw-error { background: rgba(248, 81, 73, 0.08); }
	.raw-seq { font-family: var(--font-mono); color: var(--text-muted); min-width: 35px; font-size: 0.6rem; }
	.raw-type {
		font-family: var(--font-mono); font-weight: 600; min-width: 80px; font-size: 0.65rem;
		color: var(--accent-light);
	}
	.raw-round { font-family: var(--font-mono); color: var(--orange); min-width: 30px; font-size: 0.65rem; }
	.raw-preview {
		font-family: var(--font-mono); color: var(--text-muted); font-size: 0.6rem;
		overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0;
	}

	/* ── Token list (when not inspecting) ── */
	.filter-bar { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
	.filter-tabs { display: flex; gap: 0.25rem; }
	.filter-tab {
		padding: 0.35rem 0.75rem; background: transparent; border: 1px solid var(--border);
		border-radius: var(--radius); color: var(--text-muted); font-size: 0.75rem; font-family: var(--font-mono);
	}
	.filter-tab:hover { border-color: #484f58; color: var(--text); }
	.filter-tab.active { border-color: var(--accent); color: var(--accent); background: rgba(57, 211, 83, 0.05); }
	.tab-count {
		font-size: 0.65rem; padding: 0 0.3rem; background: rgba(255, 255, 255, 0.05);
		border-radius: 3px; margin-left: 0.25rem;
	}
	.delete-all-btn {
		padding: 0.3rem 0.75rem; background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.3); border-radius: var(--radius);
		color: var(--red); font-size: 0.7rem; font-family: var(--font-mono);
	}
	.delete-all-btn:hover { background: rgba(248, 81, 73, 0.2); }

	.token-list { display: flex; flex-direction: column; gap: 0.5rem; }
	.token-row {
		display: flex; align-items: center; justify-content: space-between; gap: 1rem;
		padding: 0.75rem 1rem; transition: border-color 0.15s ease, opacity 0.15s ease;
	}
	.token-row.expired { opacity: 0.5; }
	.token-row:hover { border-color: var(--accent); }
	.token-main { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 0.35rem; }
	.token-top { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
	.diff-tag {
		font-size: 0.65rem; font-weight: 700; padding: 0.1rem 0.4rem;
		border-radius: 3px; border: 1px solid; letter-spacing: 0.05em;
	}
	.diff-tag.unknown { background: rgba(139, 148, 158, 0.1); color: var(--text-muted); border-color: var(--border); }
	.seed-tag { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted); }
	.label-tag {
		font-size: 0.7rem; padding: 0.1rem 0.4rem; background: rgba(88, 166, 255, 0.1);
		border: 1px solid rgba(88, 166, 255, 0.3); border-radius: 3px; color: var(--blue);
	}
	.time-tag { font-size: 0.7rem; color: var(--text-muted); margin-left: auto; }
	.expired-badge {
		font-size: 0.6rem; padding: 0 0.3rem; background: rgba(248, 81, 73, 0.15);
		color: var(--red); border-radius: 2px; margin-left: 0.25rem; font-weight: 600;
	}
	.token-url { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
	.token-url code { font-size: 0.7rem; color: var(--text-muted); font-family: var(--font-mono); }
	.token-actions { display: flex; gap: 0.35rem; flex-shrink: 0; }
	.connect-btn-sm {
		padding: 0.3rem 0.6rem; background: rgba(57, 211, 83, 0.15);
		border: 1px solid rgba(57, 211, 83, 0.4); border-radius: var(--radius);
		color: var(--accent); font-size: 0.7rem; font-family: var(--font-mono); font-weight: 600;
	}
	.connect-btn-sm:hover { background: rgba(57, 211, 83, 0.25); }
	.copy-btn {
		padding: 0.3rem 0.6rem; background: rgba(57, 211, 83, 0.1);
		border: 1px solid rgba(57, 211, 83, 0.3); border-radius: var(--radius);
		color: var(--accent); font-size: 0.7rem; font-family: var(--font-mono);
	}
	.copy-btn:hover { background: rgba(57, 211, 83, 0.2); }
	.del-btn {
		padding: 0.3rem 0.6rem; background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.3); border-radius: var(--radius);
		color: var(--red); font-size: 0.7rem; font-family: var(--font-mono);
	}
	.del-btn:hover { background: rgba(248, 81, 73, 0.2); }

	.empty-state { text-align: center; padding: 2rem; }
	.empty-state p { color: var(--text-muted); }
</style>
