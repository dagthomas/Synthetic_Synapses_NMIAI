<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { Separator } from '$lib/components/ui/separator';

	const links = [
		{ href: `${base}/`, label: 'Command', icon: '>' },
		{ href: `${base}/rounds`, label: 'Planets', icon: '#' },
		{ href: `${base}/teams`, label: 'Corps', icon: '@' },
		{ href: `${base}/stats`, label: 'Intel', icon: '%' },
	];
</script>

<nav class="w-56 bg-cyber-surface border-r border-cyber-border min-h-screen p-4 flex flex-col gap-1">
	<div class="flex items-center gap-2 mb-2 px-2">
		<div class="w-1.5 h-5 bg-neon-cyan rounded-sm animate-pulse-glow"></div>
		<span class="text-lg font-bold text-neon-cyan neon-text tracking-wider uppercase">Q* FRONTIER</span>
	</div>
	<p class="text-[10px] text-cyber-muted tracking-widest uppercase px-2 mb-4">Command Center</p>

	<Separator class="bg-cyber-border/50 mb-3" />

	{#each links as link}
		{@const active = page.url.pathname === link.href || (link.href !== `${base}/` && page.url.pathname.startsWith(link.href))}
		<a
			href={link.href}
			class="flex items-center gap-3 px-3 py-2.5 rounded text-sm transition-all duration-200
				{active
					? 'bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/20 shadow-[0_0_10px_rgba(0,255,240,0.06)]'
					: 'text-cyber-muted hover:text-neon-cyan hover:bg-cyber-panel border border-transparent'}"
		>
			<span class="text-xs font-bold opacity-60">{link.icon}</span>
			<span class="tracking-wide">{link.label}</span>
			{#if active}
				<div class="ml-auto w-1 h-4 bg-neon-cyan rounded-full animate-pulse-glow"></div>
			{/if}
		</a>
	{/each}

	<div class="mt-auto pt-4">
		<Separator class="bg-cyber-border/50 mb-3" />
		<div class="px-2 text-[10px] text-cyber-muted/50 tracking-wider">
			SYS::ONLINE
		</div>
	</div>
</nav>
