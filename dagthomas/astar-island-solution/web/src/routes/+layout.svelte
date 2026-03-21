<script lang="ts">
	import '../app.css';
	import Sidebar from '$lib/components/layout/Sidebar.svelte';
	import StatusBar from '$lib/components/layout/StatusBar.svelte';
	import type { Snippet } from 'svelte';

	let { children, data }: { children: Snippet; data: any } = $props();
</script>

<svelte:head>
	<title>Astar Island Explorer</title>
</svelte:head>

<div class="flex h-screen w-screen overflow-hidden relative">
	<!-- Ambient corner glows -->
	<div class="fixed top-0 left-0 w-64 h-64 bg-neon-cyan/3 rounded-full blur-3xl pointer-events-none"></div>
	<div class="fixed bottom-0 right-0 w-96 h-96 bg-neon-magenta/2 rounded-full blur-3xl pointer-events-none"></div>
	<div class="fixed top-1/2 right-0 w-48 h-48 bg-neon-gold/2 rounded-full blur-3xl pointer-events-none"></div>

	<Sidebar />

	<div class="flex flex-col flex-1 min-w-0">
		<main class="flex-1 overflow-y-auto p-4">
			{@render children()}
		</main>

		<StatusBar
			budget={data?.budget ?? null}
			processes={data?.processes ?? []}
			connected={data?.connected ?? true}
		/>
	</div>
</div>
