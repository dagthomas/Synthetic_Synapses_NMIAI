<script>
	import '../app.css';
	let { children } = $props();
</script>

<nav>
	<a href="/" class="logo">Grocery Bot Replay</a>
	<div class="nav-links">
		<a href="/">Runs</a>
		<a href="/live">Live</a>
		<a href="/optimize">Optimize</a>
		<a href="/gpu" class="gpu-link">GPU</a>
		<a href="/pipeline" class="pipeline-link">Pipeline</a>
	</div>
</nav>

<main>
	{@render children()}
</main>

<style>
	nav {
		background: var(--bg-card);
		border-bottom: 1px solid var(--border);
		padding: 0.75rem 1.5rem;
		display: flex;
		align-items: center;
		gap: 1.5rem;
	}
	.nav-links {
		display: flex;
		gap: 1rem;
		font-size: 0.85rem;
	}
	.nav-links a {
		color: var(--text-muted);
		text-decoration: none;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
	}
	.nav-links a:hover {
		color: var(--text);
		background: var(--bg-hover);
		text-decoration: none;
	}
	.logo {
		font-size: 1.1rem;
		font-weight: 700;
		color: var(--text);
		text-decoration: none;
	}
	.logo:hover {
		color: var(--accent-light);
		text-decoration: none;
	}
	:global(.gpu-link) {
		color: #00ff41 !important;
		font-weight: 600;
		text-shadow: 0 0 8px rgba(0, 255, 65, 0.4);
	}
	:global(.gpu-link:hover) {
		background: rgba(0, 255, 65, 0.1) !important;
		color: #00ff41 !important;
	}
	:global(.pipeline-link) {
		color: #fdcb6e !important;
		font-weight: 600;
		text-shadow: 0 0 8px rgba(253, 203, 110, 0.4);
	}
	:global(.pipeline-link:hover) {
		background: rgba(253, 203, 110, 0.1) !important;
		color: #fdcb6e !important;
	}
	main {
		width: 100%;
		padding: 1.5rem;
	}
</style>
