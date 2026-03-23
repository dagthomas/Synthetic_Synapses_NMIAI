<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';

	const links = [
		{ href: `${base}/`, label: 'Dashboard', icon: '⌘' },
		{ href: `${base}/rounds`, label: 'Rounds', icon: '◎' },
		{ href: `${base}/teams`, label: 'Teams', icon: '⚑' },
		{ href: `${base}/stats`, label: 'Stats', icon: '▤' },
	];
</script>

<nav class="w-56 bg-gray-900 border-r border-gray-800 min-h-screen p-4 flex flex-col gap-1">
	<div class="text-lg font-bold text-white mb-6 px-2">Astar Island</div>

	{#each links as link}
		{@const active = page.url.pathname === link.href || (link.href !== `${base}/` && page.url.pathname.startsWith(link.href))}
		<a
			href={link.href}
			class="flex items-center gap-2 px-3 py-2 rounded text-sm transition-colors
				{active ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white hover:bg-gray-800'}"
		>
			<span class="text-base">{link.icon}</span>
			{link.label}
		</a>
	{/each}
</nav>
