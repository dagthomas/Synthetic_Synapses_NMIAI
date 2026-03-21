// Mirror of Go api/types.go

export interface Round {
	id: string;
	round_number: number;
	status: string;
	map_width: number;
	map_height: number;
	started_at: string;
	closes_at: string;
	event_date: string;
	round_weight: number;
}

export interface RoundDetail extends Round {
	initial_states: InitialState[];
}

export interface InitialState {
	grid: number[][];
	settlements: Settlement[];
}

export interface Settlement {
	x: number;
	y: number;
	population: number;
	food: number;
	wealth: number;
	defense: number;
	has_port: boolean;
	alive: boolean;
	owner_id: number;
}

export interface Budget {
	queries_used: number;
	queries_max: number;
}

export interface MyRound {
	id: string;
	round_number: number;
	status: string;
	round_score: number | null;
	rank: number | null;
	seed_scores: number[];
	seeds_submitted: number;
	seeds_count: number;
	queries_used: number;
	queries_max: number;
}

export interface Analysis {
	round_id: string;
	seed_index: number;
	score: number;
	ground_truth: number[][][];
	prediction: number[][][];
	cell_scores: number[][];
}

export interface LeaderboardEntry {
	rank: number;
	team_name: string;
	weighted_score: number;
	rounds_participated: number;
	hot_streak_score: number;
}

export interface Observation {
	seed_index: number;
	query_num: number;
	viewport: Viewport;
	grid: number[][];
	settlements: Settlement[];
}

export interface Viewport {
	x: number;
	y: number;
	w: number;
	h: number;
}

export interface AutoloopEntry {
	id: number;
	timestamp: string;
	name: string;
	params: Record<string, number>;
	scores_quick: Record<string, number>;
	scores_full: Record<string, number>;
	accepted: boolean;
	baseline_avg: number;
	elapsed: number;
}

export interface ResearchEntry {
	id: number;
	timestamp: string;
	name: string;
	status: string;
	hypothesis: string;
	model: string;
	scores: Record<string, number>;
	improvement: number;
	warnings: string[];
	elapsed: number;
	error: string;
	code: string;
	timings: ResearchTimings | null;
}

export interface ResearchTimings {
	analysis: number;
	code: number;
	backtest: number;
	total: number;
}

export interface GeminiResearchEntry {
	id: number;
	timestamp: string;
	name: string;
	hypothesis: string;
	scores: Record<string, number>;
	improvement: number | string;
	elapsed: number;
	error: string;
	proposal_summary: string;
}

export interface ProcessInfo {
	name: string;
	state: string;
	uptime?: string;
	output_lines: number;
}

export interface Metrics {
	autoloop_count: number;
	adk_count: number;
	gemini_count: number;
	multi_count: number;
	best_score: number;
	queries_used: number;
	queries_max: number;
}

// Imagen types
export interface ImagenGenerateRequest {
	image: string; // base64 PNG, no data: prefix
	prompt?: string;
}

export interface GeneratedImage {
	id: string;
	filename: string;
	image_base64: string;
	prompt: string;
	created_at: string;
}

export interface GalleryEntry {
	id: string;
	filename: string;
	prompt: string;
	created_at: string;
	url: string;
}

// Terrain codes
export const TerrainCode = {
	EMPTY: 0,
	SETTLEMENT: 1,
	PORT: 2,
	RUIN: 3,
	FOREST: 4,
	MOUNTAIN: 5,
	OCEAN: 10,
	PLAINS: 11
} as const;

export type TerrainCodeType = (typeof TerrainCode)[keyof typeof TerrainCode];

export const TerrainNames: Record<number, string> = {
	0: 'Empty',
	1: 'Settlement',
	2: 'Port',
	3: 'Ruin',
	4: 'Forest',
	5: 'Mountain',
	10: 'Ocean',
	11: 'Plains'
};

export const TerrainColors: Record<number, string> = {
	0: '#2a2a3a',
	1: '#d4a843',
	2: '#4fc3f7',
	3: '#e53935',
	4: '#2e7d32',
	5: '#8a8a8a',
	10: '#1565c0',
	11: '#558b2f'
};

// Daemon types
export type DaemonStatus = string[];

export interface DaemonParams {
	prior_w: number;
	emp_max: number;
	exp_damp: number;
	base_power: number;
	T_high: number;
	smooth_alpha: number;
	floor: number;
	updated_at: string;
	source: string;
	score_avg: number;
	score_boom: number;
	score_nonboom: number;
	experiment_id?: number;
}

export interface DaemonAutoloop {
	total_experiments: number;
	best_score: number;
	best_boom: number;
	best_nonboom: number;
	experiments_per_hour: number;
	last_improvement_id: number;
	accepted_count: number;
	top_params: AutoloopEntry[];
}

export interface DaemonRoundScore {
	round_number: number;
	avg_score: number;
	regime: string;
	settlement_pct: number;
}
