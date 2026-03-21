import type { FlowNodeDef, FlowEdgeDef } from './flow-types';

// Layout: 1800x900 logical coordinate space
// Left-to-right flow: API <-> Daemon -> Autoloop/Researchers -> Data/Precompute -> Pipeline stages -> Submit -> API

export const NODES: FlowNodeDef[] = [
	// === Tier 1: Core Processes (neon-cyan) ===
	{
		id: 'api',
		label: 'API SERVER',
		sublabel: 'api.ainm.no',
		tier: 'core',
		icon: '\u16DF', // ᛟ
		color: 'var(--color-neon-cyan)',
		position: { x: 40, y: 300 },
		size: { w: 170, h: 130 },
		metrics: [
			{ key: 'queries', label: 'Budget', format: 'text' },
			{ key: 'endpoint', label: 'Endpoint', format: 'text' },
		],
	},
	{
		id: 'daemon',
		label: 'DAEMON',
		sublabel: 'daemon.py',
		tier: 'core',
		icon: '\u16B9', // ᚹ
		color: 'var(--color-neon-cyan)',
		position: { x: 280, y: 250 },
		size: { w: 170, h: 150 },
		processName: 'daemon',
		metrics: [
			{ key: 'status', label: 'Status', format: 'text' },
			{ key: 'log_lines', label: 'Log', format: 'number' },
		],
	},

	// === Tier 2: Research (neon-magenta) ===
	{
		id: 'gemini',
		label: 'GEMINI',
		sublabel: 'gemini_researcher.py',
		tier: 'research',
		icon: '\u16B7', // ᚷ
		color: 'var(--color-neon-magenta)',
		position: { x: 510, y: 80 },
		size: { w: 170, h: 120 },
		processName: 'gemini',
		metrics: [
			{ key: 'proposals', label: 'Proposals', format: 'number' },
			{ key: 'improvements', label: 'Improved', format: 'number' },
		],
	},
	{
		id: 'multi',
		label: 'MULTI',
		sublabel: 'multi_researcher.py',
		tier: 'research',
		icon: '\u16D6', // ᛖ
		color: 'var(--color-neon-magenta)',
		position: { x: 510, y: 220 },
		size: { w: 170, h: 120 },
		processName: 'multi',
		metrics: [
			{ key: 'proposals', label: 'Proposals', format: 'number' },
			{ key: 'improvements', label: 'Improved', format: 'number' },
		],
	},

	// === Tier 1: Autoloop (neon-cyan) ===
	{
		id: 'autoloop',
		label: 'AUTOLOOP',
		sublabel: 'autoloop_fast.py',
		tier: 'core',
		icon: '\u16CF', // ᛏ
		color: 'var(--color-neon-cyan)',
		position: { x: 510, y: 380 },
		size: { w: 170, h: 140 },
		processName: 'autoloop',
		metrics: [
			{ key: 'experiments', label: 'Experiments', format: 'number' },
			{ key: 'best_score', label: 'Best', format: 'score' },
			{ key: 'rate', label: 'Exp/hr', format: 'rate' },
		],
	},

	// === Tier 4: Pre-computation & Data (neon-orange) ===
	{
		id: 'fast_harness',
		label: 'FastHarness',
		sublabel: 'pre-compute & eval',
		tier: 'data',
		icon: '\u26A1', // ⚡
		color: 'var(--color-neon-orange)',
		position: { x: 750, y: 180 },
		size: { w: 160, h: 110 },
		metrics: [
			{ key: 'speed', label: 'Speed', format: 'text' },
			{ key: 'cached', label: 'Cache', format: 'text' },
		],
	},
	{
		id: 'cal_model',
		label: 'CalibModel',
		sublabel: 'calibration.py',
		tier: 'data',
		icon: '\u16DE', // ᛞ
		color: 'var(--color-neon-orange)',
		position: { x: 750, y: 320 },
		size: { w: 160, h: 100 },
		metrics: [
			{ key: 'levels', label: 'Hierarchy', format: 'text' },
		],
	},
	{
		id: 'best_params',
		label: 'best_params',
		sublabel: '.json',
		tier: 'data',
		icon: '\u2699', // ⚙
		color: 'var(--color-neon-orange)',
		position: { x: 750, y: 450 },
		size: { w: 160, h: 110 },
		metrics: [
			{ key: 'score', label: 'Score', format: 'score' },
			{ key: 'updated', label: 'Updated', format: 'text' },
		],
	},
	{
		id: 'calibration_data',
		label: 'Calibration',
		sublabel: 'data/calibration/',
		tier: 'data',
		icon: '\u1F4C1', // folder
		color: 'var(--color-neon-orange)',
		position: { x: 750, y: 590 },
		size: { w: 160, h: 90 },
		metrics: [
			{ key: 'rounds', label: 'Rounds', format: 'text' },
		],
	},
	{
		id: 'autoloop_log',
		label: 'Autoloop Log',
		sublabel: 'autoloop_fast_log.jsonl',
		tier: 'data',
		icon: '\u1F4DD', // memo
		color: 'var(--color-neon-orange)',
		position: { x: 510, y: 560 },
		size: { w: 160, h: 90 },
		metrics: [
			{ key: 'entries', label: 'Entries', format: 'number' },
		],
	},
	{
		id: 'research_logs',
		label: 'Research Logs',
		sublabel: '*.jsonl',
		tier: 'data',
		icon: '\u1F4DD',
		color: 'var(--color-neon-orange)',
		position: { x: 280, y: 80 },
		size: { w: 150, h: 90 },
		metrics: [
			{ key: 'total', label: 'Total entries', format: 'number' },
		],
	},

	// === Tier 3: Pipeline (neon-gold) ===
	{
		id: 'exploration',
		label: 'EXPLORATION',
		sublabel: 'explore.py',
		tier: 'pipeline',
		icon: '\u16CB', // ᛋ
		color: 'var(--color-neon-gold)',
		position: { x: 280, y: 480 },
		size: { w: 170, h: 120 },
		metrics: [
			{ key: 'strategy', label: 'Strategy', format: 'text' },
			{ key: 'viewports', label: 'Viewports', format: 'text' },
		],
	},
	// === Prediction Pipeline Stages (3x3 grid) ===
	{
		id: 'pred_1',
		label: 'Feature Keys',
		sublabel: 'terrain+sett\u2192fkey',
		tier: 'pipeline',
		icon: '\u2731', // ✱
		color: 'var(--color-neon-gold)',
		position: { x: 980, y: 100 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_2',
		label: 'Cal Prior',
		sublabel: 'fine\u2192coarse\u2192base',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-gold)',
		position: { x: 1140, y: 100 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_3',
		label: 'FK Empirical',
		sublabel: 'obs distributions',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-gold)',
		position: { x: 1300, y: 100 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_4',
		label: 'FK Blending',
		sublabel: 'prior\u00d7w + emp\u00d7\u221acount',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-gold)',
		position: { x: 980, y: 200 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_5',
		label: 'Global Mult',
		sublabel: 'obs/expected ratio',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-cyan)',
		position: { x: 1140, y: 200 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_6',
		label: 'Temperature',
		sublabel: 'entropy sharpening',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-cyan)',
		position: { x: 1300, y: 200 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_7',
		label: 'Smoothing',
		sublabel: '\u03b1=0.15 spatial',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-magenta)',
		position: { x: 980, y: 300 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_8',
		label: 'Struct Zeros',
		sublabel: 'mtn/ocean locks',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-magenta)',
		position: { x: 1140, y: 300 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'pred_9',
		label: 'Floor',
		sublabel: 'min 0.005 nonzero',
		tier: 'pipeline',
		icon: '\u2731',
		color: 'var(--color-neon-magenta)',
		position: { x: 1300, y: 300 },
		size: { w: 140, h: 80 },
		metrics: [{ key: 'desc', label: 'Stage', format: 'text' }],
	},
	{
		id: 'submission',
		label: 'SUBMISSION',
		sublabel: 'POST /submit',
		tier: 'pipeline',
		icon: '\u16A0', // ᚠ
		color: 'var(--color-neon-gold)',
		position: { x: 1140, y: 420 },
		size: { w: 170, h: 110 },
		metrics: [
			{ key: 'tensor', label: 'Tensor', format: 'text' },
		],
	},
];

export const EDGES: FlowEdgeDef[] = [
	// Daemon orchestration
	{ id: 'daemon-autoloop', from: 'daemon', to: 'autoloop', label: 'spawns', animated: true, activeWhen: 'daemon' },
	{ id: 'daemon-exploration', from: 'daemon', to: 'exploration', label: 'triggers', animated: true, activeWhen: 'exploration' },
	{ id: 'daemon-api', from: 'daemon', to: 'api', label: '/rounds', animated: true, activeWhen: 'daemon' },
	{ id: 'daemon-caldata', from: 'daemon', to: 'calibration_data', label: 'downloads GT', animated: true, dashed: true, activeWhen: 'daemon' },

	// Exploration -> Prediction pipeline
	{ id: 'api-exploration', from: 'api', to: 'exploration', label: '/simulate', animated: true, activeWhen: 'exploration', color: 'var(--color-neon-gold)' },
	{ id: 'exploration-pred1', from: 'exploration', to: 'pred_1', label: 'obs, terrain', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-gold)' },

	// Pipeline stage chain: 1->2->3->4->5->6->7->8->9
	{ id: 'pred-1-2', from: 'pred_1', to: 'pred_2', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-gold)' },
	{ id: 'pred-2-3', from: 'pred_2', to: 'pred_3', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-gold)' },
	{ id: 'pred-3-4', from: 'pred_3', to: 'pred_4', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-gold)' },
	{ id: 'pred-4-5', from: 'pred_4', to: 'pred_5', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-cyan)' },
	{ id: 'pred-5-6', from: 'pred_5', to: 'pred_6', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-cyan)' },
	{ id: 'pred-6-7', from: 'pred_6', to: 'pred_7', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-magenta)' },
	{ id: 'pred-7-8', from: 'pred_7', to: 'pred_8', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-magenta)' },
	{ id: 'pred-8-9', from: 'pred_8', to: 'pred_9', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-magenta)' },
	{ id: 'pred-9-submit', from: 'pred_9', to: 'submission', label: '40\u00d740\u00d76', animated: true, activeWhen: 'prediction', color: 'var(--color-neon-gold)' },

	{ id: 'submission-api', from: 'submission', to: 'api', label: 'POST /submit', animated: true, activeWhen: 'submission', color: 'var(--color-neon-gold)' },

	// Calibration data flow
	{ id: 'caldata-calmodel', from: 'calibration_data', to: 'cal_model', label: 'loads', animated: false, dashed: true },
	{ id: 'calmodel-pred2', from: 'cal_model', to: 'pred_2', label: 'priors', animated: true, activeWhen: 'prediction', dashed: true },
	{ id: 'calmodel-harness', from: 'cal_model', to: 'fast_harness', label: 'pre-compute', animated: true, activeWhen: 'autoloop', dashed: true },

	// Autoloop optimization loop
	{ id: 'harness-autoloop', from: 'fast_harness', to: 'autoloop', label: '~70ms eval', animated: true, activeWhen: 'autoloop' },
	{ id: 'autoloop-params', from: 'autoloop', to: 'best_params', label: 'writes', animated: true, activeWhen: 'autoloop', dashed: true },
	{ id: 'params-pred4', from: 'best_params', to: 'pred_4', label: 'params', animated: true, activeWhen: 'prediction', dashed: true },
	{ id: 'autoloop-log', from: 'autoloop', to: 'autoloop_log', label: 'JSONL', animated: true, activeWhen: 'autoloop', dashed: true },

	// Research -> harness
	{ id: 'gemini-harness', from: 'gemini', to: 'fast_harness', label: 'backtest', animated: true, activeWhen: 'gemini', color: 'var(--color-neon-magenta)' },
	{ id: 'multi-harness', from: 'multi', to: 'fast_harness', label: 'backtest', animated: true, activeWhen: 'multi', color: 'var(--color-neon-magenta)' },

	// Research -> logs
	{ id: 'gemini-logs', from: 'gemini', to: 'research_logs', label: 'JSONL', animated: true, activeWhen: 'gemini', dashed: true, color: 'var(--color-neon-magenta)' },
	{ id: 'multi-logs', from: 'multi', to: 'research_logs', label: 'JSONL', animated: true, activeWhen: 'multi', dashed: true, color: 'var(--color-neon-magenta)' },
];

// Helper: get node by ID
export function getNode(id: string): FlowNodeDef | undefined {
	return NODES.find((n) => n.id === id);
}

// Helper: get center position of a node
export function getNodeCenter(node: FlowNodeDef): { x: number; y: number } {
	return {
		x: node.position.x + node.size.w / 2,
		y: node.position.y + node.size.h / 2,
	};
}

// Helper: compute cubic bezier path between two nodes
export function computeEdgePath(from: FlowNodeDef, to: FlowNodeDef): string {
	const fc = getNodeCenter(from);
	const tc = getNodeCenter(to);

	// Determine best connection points (edge of node closest to other node)
	const fromPt = getConnectionPoint(from, tc);
	const toPt = getConnectionPoint(to, fc);

	const dx = toPt.x - fromPt.x;
	const dy = toPt.y - fromPt.y;

	// Control point offset: proportional to distance, min 40px
	const cpOffset = Math.max(40, Math.min(Math.abs(dx) * 0.4, 120));

	// Bias control points horizontally for left-to-right flow
	const cp1x = fromPt.x + cpOffset;
	const cp1y = fromPt.y + dy * 0.1;
	const cp2x = toPt.x - cpOffset;
	const cp2y = toPt.y - dy * 0.1;

	return `M ${fromPt.x},${fromPt.y} C ${cp1x},${cp1y} ${cp2x},${cp2y} ${toPt.x},${toPt.y}`;
}

// Get the connection point on the edge of a node closest to a target point
function getConnectionPoint(
	node: FlowNodeDef,
	target: { x: number; y: number }
): { x: number; y: number } {
	const cx = node.position.x + node.size.w / 2;
	const cy = node.position.y + node.size.h / 2;
	const hw = node.size.w / 2;
	const hh = node.size.h / 2;

	const dx = target.x - cx;
	const dy = target.y - cy;
	const angle = Math.atan2(dy, dx);

	// Find intersection with rectangle
	const tanAngle = Math.tan(angle);
	let ix: number, iy: number;

	if (Math.abs(dx) * hh > Math.abs(dy) * hw) {
		// Intersects left or right edge
		ix = dx > 0 ? hw : -hw;
		iy = ix * tanAngle;
	} else {
		// Intersects top or bottom edge
		iy = dy > 0 ? hh : -hh;
		ix = tanAngle !== 0 ? iy / tanAngle : 0;
	}

	return { x: cx + ix, y: cy + iy };
}
