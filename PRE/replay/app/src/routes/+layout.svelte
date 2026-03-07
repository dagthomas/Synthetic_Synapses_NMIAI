<script>
	import '../app.css';
	import { page } from '$app/stores';
	let { children } = $props();

	const GLYPHS = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~';

	function scrambleIn(node) {
		// Collect all text nodes (skip script/style/svg)
		const SKIP = new Set(['SCRIPT', 'STYLE', 'SVG', 'CANVAS', 'IMG', 'INPUT', 'TEXTAREA', 'SELECT']);
		const entries = [];

		function walk(el) {
			if (SKIP.has(el.tagName)) return;
			for (const child of el.childNodes) {
				if (child.nodeType === Node.TEXT_NODE) {
					const original = child.textContent;
					if (original.trim().length > 0) {
						entries.push({ node: child, original });
					}
				} else if (child.nodeType === Node.ELEMENT_NODE) {
					walk(child);
				}
			}
		}
		walk(node);
		if (entries.length === 0) return;

		// Scramble all text immediately
		for (const e of entries) {
			e.node.textContent = e.original.replace(/\S/g, () => GLYPHS[Math.random() * GLYPHS.length | 0]);
		}

		const DURATION = 400;   // ms total
		const STAGGER  = 30;    // ms between each char locking in
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
					if (ch === ' ' || ch === '\n' || ch === '\t') {
						out.push(ch);
						continue;
					}
					// Each char locks in at its own time
					const lockAt = (i * STAGGER) + Math.random() * 20;
					if (elapsed >= lockAt + DURATION * 0.3) {
						out.push(ch);
					} else {
						out.push(GLYPHS[Math.random() * GLYPHS.length | 0]);
						allDone = false;
					}
				}
				e.node.textContent = out.join('');
			}

			if (!allDone && elapsed < DURATION + entries.reduce((m, e) => Math.max(m, e.original.length), 0) * STAGGER) {
				frame = requestAnimationFrame(tick);
			} else {
				// Ensure final text is correct
				for (const e of entries) {
					e.node.textContent = e.original;
				}
			}
		}

		frame = requestAnimationFrame(tick);

		return {
			destroy() {
				cancelAnimationFrame(frame);
				for (const e of entries) {
					e.node.textContent = e.original;
				}
			}
		};
	}
</script>

<!-- Scanline overlay -->
<div class="scanlines"></div>

<nav>
	<div class="nav-inner">
		<a href="/" class="logo">
			<span class="logo-prompt">&gt;</span> grocery-bot
			<span class="logo-cursor">_</span>
		</a>
		<div class="nav-divider"></div>
		<div class="nav-links">
			<a href="/">runs</a>
			<a href="/live">live</a>
			<a href="/optimize">optimize</a>
			<a href="/gpu" class="gpu-link">gpu</a>
			<a href="/pipeline" class="pipeline-link">pipeline</a>
			<a href="/tokens">tokens</a>
		</div>
	</div>
</nav>

{#key $page.url.pathname}
<main use:scrambleIn>
	{@render children()}
</main>
{/key}

<style>
	/* Scanline overlay */
	.scanlines {
		position: fixed;
		inset: 0;
		pointer-events: none;
		z-index: 100;
		background: repeating-linear-gradient(
			0deg,
			transparent,
			transparent 2px,
			rgba(0, 0, 0, 0.03) 2px,
			rgba(0, 0, 0, 0.03) 4px
		);
	}

	nav {
		position: relative;
		z-index: 10;
		background: rgba(13, 17, 23, 0.95);
		border-bottom: 1px solid var(--border);
		padding: 0.75rem 1.5rem;
	}
	.nav-inner {
		display: flex;
		align-items: center;
		gap: 1.25rem;
		max-width: 1400px;
		margin: 0 auto;
	}
	.nav-divider {
		width: 1px;
		height: 20px;
		background: var(--border);
	}
	.nav-links {
		display: flex;
		gap: 0.25rem;
		font-size: 0.8rem;
		font-weight: 400;
	}
	.nav-links a {
		color: var(--text-muted);
		text-decoration: none;
		padding: 0.3rem 0.65rem;
		border-radius: 2px;
		transition: all 0.2s ease;
		letter-spacing: 0.02em;
	}
	.nav-links a:hover {
		color: var(--accent);
		background: rgba(57, 211, 83, 0.08);
		text-decoration: none;
	}
	.logo {
		font-family: var(--font-display);
		font-size: 1rem;
		font-weight: 700;
		color: var(--accent);
		text-decoration: none;
		letter-spacing: 0.02em;
	}
	.logo:hover {
		color: var(--accent-light);
		text-decoration: none;
	}
	.logo-prompt {
		color: var(--accent);
		font-weight: 800;
	}
	.logo-cursor {
		color: var(--accent);
		animation: blink-cursor 1s step-end infinite;
	}
	@keyframes blink-cursor {
		0%, 100% { opacity: 1; }
		50% { opacity: 0; }
	}
	:global(.gpu-link) {
		color: var(--accent) !important;
		font-weight: 600;
	}
	:global(.gpu-link:hover) {
		background: rgba(57, 211, 83, 0.1) !important;
		color: var(--accent-light) !important;
	}
	:global(.pipeline-link) {
		color: var(--accent) !important;
		font-weight: 600;
	}
	:global(.pipeline-link:hover) {
		background: rgba(57, 211, 83, 0.1) !important;
		color: var(--accent-light) !important;
	}
	main {
		position: relative;
		z-index: 1;
		width: 100%;
		padding: 1.5rem;
	}
</style>
