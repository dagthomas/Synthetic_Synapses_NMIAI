<script lang="ts">
	import { page } from '$app/state';

	const navItems = [
		{ href: '/', label: 'Dashboard', rune: 'ᛟ' },
		{ href: '/rounds', label: 'Rounds', rune: 'ᚱ' },
		{ href: '/explorer', label: 'Explorer', rune: 'ᛋ' },
		{ href: '/autoloop', label: 'Autoloop', rune: 'ᛏ' },
		{ href: '/daemon', label: 'Daemon', rune: 'ᚹ' },
		{ href: '/flow', label: 'Flow', rune: 'ᚠ' },
		{ href: '/research', label: 'Research', rune: 'ᚨ' },
		{ href: '/metrics', label: 'Metrics', rune: 'ᛁ' },
		{ href: '/logs', label: 'Logs', rune: 'ᛚ' }
	];

	function isActive(href: string): boolean {
		if (href === '/') return page.url.pathname === '/';
		return page.url.pathname.startsWith(href);
	}
</script>

<nav class="flex flex-col w-48 h-full bg-cyber-surface border-r border-cyber-border shrink-0 relative">
	<!-- Sidebar ambient glow -->
	<div class="absolute top-0 right-0 w-px h-full bg-gradient-to-b from-neon-cyan/20 via-neon-magenta/10 to-neon-gold/20"></div>

	<!-- Logo -->
	<div class="p-4 border-b border-cyber-border relative overflow-hidden">
		<div class="text-neon-cyan text-xs tracking-[0.3em] neon-text font-bold">ᛟ ASTAR</div>
		<div class="text-neon-gold text-[10px] tracking-wider neon-text-gold">ISLAND EXPLORER</div>
		<!-- Subtle glow behind logo -->
		<div class="absolute -top-4 -left-4 w-24 h-24 bg-neon-cyan/5 rounded-full blur-2xl pointer-events-none"></div>
	</div>

	<!-- Nav items -->
	<div class="flex-1 py-2 overflow-y-auto">
		{#each navItems as item}
			{@const active = isActive(item.href)}
			<a
				href={item.href}
				class="nav-glow flex items-center gap-3 px-4 py-2.5 text-[12px] transition-all duration-200
					{active
					? 'bg-neon-cyan/10 text-neon-cyan border-r-2 border-neon-cyan shadow-[inset_-2px_0_12px_rgba(0,255,240,0.1)]'
					: 'text-cyber-muted hover:text-cyber-fg hover:bg-cyber-panel/50 hover:pl-5'}"
			>
				<span class="text-sm transition-all duration-200 {active ? 'neon-text scale-110' : 'group-hover:scale-110'}">{item.rune}</span>
				<span>{item.label}</span>
			</a>
		{/each}
	</div>

	<!-- Footer -->
	<div class="p-3 border-t border-cyber-border text-[9px] text-cyber-muted text-center">
		<span class="opacity-60">NM i AI 2026</span>
	</div>
</nav>
