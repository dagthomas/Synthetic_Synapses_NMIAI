<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';

	interface Props {
		status: string;
	}

	let { status }: Props = $props();

	const styles: Record<string, string> = {
		active: 'bg-neon-cyan/15 text-neon-cyan border-neon-cyan/30 shadow-[0_0_8px_rgba(0,255,240,0.15)]',
		pending: 'bg-neon-gold/15 text-neon-gold border-neon-gold/30 shadow-[0_0_8px_rgba(255,215,0,0.15)]',
		scoring: 'bg-neon-magenta/15 text-neon-magenta border-neon-magenta/30 shadow-[0_0_8px_rgba(255,0,255,0.15)]',
		completed: 'bg-cyber-muted/15 text-cyber-muted border-cyber-muted/30'
	};

	const labels: Record<string, string> = {
		pending: 'DORMANT',
		active: 'SCANNING',
		scoring: 'ANALYZING',
		completed: 'ARCHIVED'
	};
</script>

<Badge
	variant="outline"
	class="text-[10px] tracking-wider uppercase font-bold {styles[status] || 'bg-cyber-muted/15 text-cyber-muted border-cyber-muted/30'}"
>
	{labels[status] || status}
</Badge>
