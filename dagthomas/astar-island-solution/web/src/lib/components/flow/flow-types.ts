export type NodeTier = 'core' | 'research' | 'pipeline' | 'data';
export type NodeStatus = 'active' | 'idle' | 'error' | 'unknown';

export interface FlowNodeDef {
	id: string;
	label: string;
	sublabel?: string;
	tier: NodeTier;
	icon: string;
	color: string;
	position: { x: number; y: number };
	size: { w: number; h: number };
	processName?: string;
	isExpandable?: boolean;
	metrics: NodeMetricDef[];
}

export interface NodeMetricDef {
	key: string;
	label: string;
	format: 'number' | 'score' | 'duration' | 'rate' | 'text';
}

export interface FlowEdgeDef {
	id: string;
	from: string;
	to: string;
	label?: string;
	color?: string;
	animated: boolean;
	dashed?: boolean;
	activeWhen?: string; // node ID whose active state triggers animation
}

export interface FlowState {
	nodeStatuses: Record<string, NodeStatus>;
	nodeMetrics: Record<string, Record<string, string | number>>;
}

export interface PipelineStage {
	id: number;
	name: string;
	description: string;
	color: string;
}

export const PIPELINE_STAGES: PipelineStage[] = [
	{ id: 1, name: 'Feature Keys', description: 'terrain + settlements -> fkey grid', color: 'var(--color-neon-gold)' },
	{ id: 2, name: 'Cal Prior', description: 'fine->coarse->base->global', color: 'var(--color-neon-gold)' },
	{ id: 3, name: 'FK Empirical', description: 'observation-based distributions', color: 'var(--color-neon-gold)' },
	{ id: 4, name: 'FK Blending', description: 'prior * w + empirical * sqrt(count)', color: 'var(--color-neon-gold)' },
	{ id: 5, name: 'Global Mult', description: 'observed/expected ratio per class', color: 'var(--color-neon-cyan)' },
	{ id: 6, name: 'Temperature', description: 'entropy-weighted sharpening', color: 'var(--color-neon-cyan)' },
	{ id: 7, name: 'Smoothing', description: 'spatial alpha=0.15 sett+ruin', color: 'var(--color-neon-magenta)' },
	{ id: 8, name: 'Struct Zeros', description: 'mountain/ocean locks', color: 'var(--color-neon-magenta)' },
	{ id: 9, name: 'Floor', description: 'min 0.005 for nonzero classes', color: 'var(--color-neon-magenta)' },
];

export const TIER_COLORS: Record<NodeTier, string> = {
	core: 'var(--color-neon-cyan)',
	research: 'var(--color-neon-magenta)',
	pipeline: 'var(--color-neon-gold)',
	data: 'var(--color-neon-orange)',
};
