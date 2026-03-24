<script lang="ts">
	import '../app.css';
	import Sidebar from '$lib/components/Sidebar.svelte';
	import { isLoggedIn, login } from '$lib/api';
	import { browser } from '$app/environment';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';

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
	<title>Q* Frontier // Command Center</title>
</svelte:head>

{#if !loggedIn}
	<div class="min-h-screen bg-cyber-bg flex items-center justify-center">
		<div class="glass glass-glow p-8 w-96">
			<div class="flex items-center gap-3 mb-8">
				<div class="w-2 h-8 bg-neon-cyan rounded-full animate-pulse-glow"></div>
				<h1 class="text-xl font-bold text-neon-cyan neon-text tracking-wider uppercase">Q* Frontier</h1>
			</div>
			<p class="text-cyber-muted text-xs mb-6 tracking-wide uppercase">Command Center</p>
			<form onsubmit={(e) => { e.preventDefault(); handleLogin(); }}>
				<div class="mb-3">
					<Input
						bind:value={username}
						placeholder="Username"
						class="w-full bg-cyber-surface border-cyber-border text-cyber-fg placeholder:text-cyber-muted focus:border-neon-cyan focus:ring-neon-cyan/20"
					/>
				</div>
				<div class="mb-4">
					<Input
						bind:value={password}
						type="password"
						placeholder="Password"
						class="w-full bg-cyber-surface border-cyber-border text-cyber-fg placeholder:text-cyber-muted focus:border-neon-cyan focus:ring-neon-cyan/20"
					/>
				</div>
				{#if error}
					<p class="text-score-bad text-sm mb-3">{error}</p>
				{/if}
				<Button
					type="submit"
					class="w-full bg-neon-cyan/10 border border-neon-cyan/40 text-neon-cyan hover:bg-neon-cyan/20 hover:border-neon-cyan/60 hover:shadow-[0_0_15px_rgba(0,255,240,0.2)] transition-all"
				>
					Access Terminal
				</Button>
			</form>
		</div>
	</div>
{:else}
	<div class="min-h-screen bg-cyber-bg text-cyber-fg flex">
		<Sidebar />
		<main class="flex-1 p-6 overflow-auto">
			{@render children()}
		</main>
	</div>
{/if}
