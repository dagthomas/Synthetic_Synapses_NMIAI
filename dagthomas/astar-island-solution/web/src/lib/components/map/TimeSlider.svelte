<script lang="ts">
	let {
		timeOfDay = $bindable(12),
		autoMode = $bindable(true)
	}: {
		timeOfDay: number;
		autoMode: boolean;
	} = $props();

	let displayTime = $derived(() => {
		const h = Math.floor(timeOfDay);
		const m = Math.floor((timeOfDay % 1) * 60);
		return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
	});

	function handleInput(e: Event) {
		const target = e.target as HTMLInputElement;
		timeOfDay = parseFloat(target.value);
		autoMode = false;
	}

	function toggleAuto() {
		autoMode = !autoMode;
	}
</script>

<div class="space-y-1">
	<div class="flex items-center justify-between">
		<span class="text-[10px] text-cyber-muted">Time of Day</span>
		<button
			class="text-[9px] px-1.5 py-0.5 rounded border transition-colors
				{autoMode
				? 'border-neon-cyan/60 text-neon-cyan bg-neon-cyan/10'
				: 'border-cyber-border text-cyber-muted hover:border-neon-cyan/40'}"
			onclick={toggleAuto}
		>
			{autoMode ? 'AUTO' : 'MANUAL'}
		</button>
	</div>

	<div class="flex items-center gap-2">
		<input
			type="range"
			min="0"
			max="24"
			step="0.1"
			value={timeOfDay}
			oninput={handleInput}
			class="flex-1 h-1 appearance-none rounded-full cursor-pointer
				[&::-webkit-slider-thumb]:appearance-none
				[&::-webkit-slider-thumb]:w-3
				[&::-webkit-slider-thumb]:h-3
				[&::-webkit-slider-thumb]:rounded-full
				[&::-webkit-slider-thumb]:bg-neon-cyan
				[&::-webkit-slider-thumb]:shadow-[0_0_6px_rgba(0,255,240,0.5)]
				[&::-webkit-slider-thumb]:cursor-pointer"
			style="background: linear-gradient(to right,
				#0a0a2a 0%,
				#0a0a2a 20%,
				#ff9966 25%,
				#87ceeb 35%,
				#87ceeb 70%,
				#ff6b35 77%,
				#0a0a2a 82%,
				#0a0a2a 100%)"
		/>
		<span class="text-[11px] text-neon-cyan font-mono w-10 text-right">{displayTime()}</span>
	</div>
</div>
