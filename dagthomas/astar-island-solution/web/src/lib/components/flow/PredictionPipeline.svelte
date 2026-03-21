<script lang="ts">
	import { PIPELINE_STAGES } from './flow-types';

	let { onclose }: { onclose: () => void } = $props();
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
	class="absolute inset-0 z-40 flex items-center justify-center pipeline-panel"
	style="background: rgba(10, 10, 26, 0.8); backdrop-filter: blur(8px);"
	onclick={(e) => { if (e.target === e.currentTarget) onclose(); }}
>
	<div class="glass glass-border-glow p-6 max-w-[800px] w-full mx-8" style="
		border-color: rgba(255, 215, 0, 0.3);
		box-shadow: 0 0 40px rgba(255, 215, 0, 0.1), inset 0 0 20px rgba(255, 215, 0, 0.03);
	">
		<!-- Header -->
		<div class="flex items-center justify-between mb-5">
			<div>
				<h2 class="text-sm font-bold tracking-wider neon-text-gold" style="color: var(--color-neon-gold);">
					PREDICTION PIPELINE
				</h2>
				<p class="text-[10px] text-cyber-muted mt-0.5">predict_gemini.py &mdash; 9-stage processing</p>
			</div>
			<button
				class="w-7 h-7 flex items-center justify-center text-cyber-muted hover:text-neon-gold transition-colors rounded border border-cyber-border hover:border-neon-gold/30"
				onclick={onclose}
			>
				&times;
			</button>
		</div>

		<!-- Pipeline stages as connected flow -->
		<div class="relative">
			<!-- Connection line -->
			<div class="absolute left-6 top-0 bottom-0 w-px" style="background: linear-gradient(to bottom, var(--color-neon-gold), var(--color-neon-cyan), var(--color-neon-magenta));  opacity: 0.3;"></div>

			<div class="space-y-2">
				{#each PIPELINE_STAGES as stage, i}
					<div class="flex items-start gap-4 relative group">
						<!-- Stage number node -->
						<div class="relative z-10 flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center text-sm font-bold transition-all duration-300 group-hover:scale-110" style="
							background: rgba(26, 26, 46, 0.9);
							border: 1px solid color-mix(in srgb, {stage.color} 30%, transparent);
							box-shadow: 0 0 10px color-mix(in srgb, {stage.color} 15%, transparent);
							color: {stage.color};
						">
							{stage.id}
						</div>

						<!-- Stage content -->
						<div class="flex-1 py-1.5">
							<div class="text-[11px] font-bold tracking-wider" style="color: {stage.color};">
								{stage.name}
							</div>
							<div class="text-[10px] text-cyber-muted mt-0.5">
								{stage.description}
							</div>
						</div>

						<!-- Arrow to next -->
						{#if i < PIPELINE_STAGES.length - 1}
							<div class="absolute left-6 -bottom-1 w-px h-2" style="background: {stage.color}; opacity: 0.4;"></div>
						{/if}
					</div>
				{/each}
			</div>
		</div>

		<!-- Output info -->
		<div class="mt-5 pt-4 border-t border-cyber-border flex items-center justify-between">
			<div class="text-[10px] text-cyber-muted">
				Output: <span class="text-neon-gold">40 &times; 40 &times; 6</span> probability tensor
			</div>
			<div class="text-[10px] text-cyber-muted">
				Classes: <span class="text-cyber-fg">Empty, Settlement, Port, Ruin, Forest, Mountain</span>
			</div>
		</div>
	</div>
</div>
