<script lang="ts">
	import '../app.css';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import { isLoggedIn, login } from '$lib/api';
	import { browser } from '$app/environment';

	let { children } = $props();
	let loggedIn = $state(browser ? isLoggedIn() : false);
	let username = $state('admin');
	let password = $state('admin');
	let error = $state('');

	async function handleLogin() {
		try {
			error = '';
			await login(username, password);
			loggedIn = true;
		} catch (e: any) {
			error = e.message;
		}
	}
</script>

<svelte:head>
	<title>Astar Island Admin</title>
</svelte:head>

{#if !loggedIn}
	<div class="min-h-screen bg-gray-950 flex items-center justify-center">
		<div class="bg-gray-900 border border-gray-800 rounded-lg p-8 w-80">
			<h1 class="text-xl font-bold text-white mb-6">Astar Island Admin</h1>
			<form onsubmit={(e) => { e.preventDefault(); handleLogin(); }}>
				<input
					bind:value={username}
					placeholder="Username"
					class="w-full mb-3 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
				/>
				<input
					bind:value={password}
					type="password"
					placeholder="Password"
					class="w-full mb-4 px-3 py-2 bg-gray-800 border border-gray-700 rounded text-white text-sm"
				/>
				{#if error}
					<p class="text-red-400 text-sm mb-3">{error}</p>
				{/if}
				<button class="w-full bg-blue-600 hover:bg-blue-500 text-white py-2 rounded text-sm font-medium">
					Login
				</button>
			</form>
		</div>
	</div>
{:else}
	<div class="min-h-screen bg-gray-950 text-gray-200 flex">
		<Sidebar />
		<main class="flex-1 p-6 overflow-auto">
			{@render children()}
		</main>
	</div>
{/if}
