export interface RoundListEntry {
	id: string;
	round_number: number;
	status: string;
	round_weight: number;
	started_at: string | null;
	closes_at: string | null;
	created_at: string;
	teams_participated: number;
	avg_score: number | null;
}

export interface AdminDashboard {
	active_round: RoundListEntry | null;
	team_count: number;
	total_predictions: number;
	total_rounds: number;
}

export interface AdminRoundDetail {
	id: string;
	round_number: number;
	status: string;
	hidden_params: Record<string, number>;
	seeds: AdminSeedInfo[];
	team_scores: TeamRoundScore[];
}

export interface AdminSeedInfo {
	seed_index: number;
	map_seed: number;
	initial_grid: number[][];
	settlement_count: number;
}

export interface TeamRoundScore {
	team_id: string;
	team_name: string;
	seed_scores: (number | null)[];
	average_score: number | null;
	queries_used: number;
}

export interface TeamInfo {
	id: string;
	name: string;
	is_admin: boolean;
	created_at: string;
	rounds_participated: number;
	total_queries: number;
}

// Terrain color mapping
export const CELL_COLORS: Record<number, string> = {
	10: '#0a0a2e', // Void (deep dark blue)
	11: '#2a1f3d', // Regolith (dark purple)
	0: '#2a1f3d',  // Empty (Regolith)
	1: '#00fff0',  // Crystal Node (neon cyan)
	2: '#ff00ff',  // Refinery (neon magenta)
	3: '#4a4a5a',  // Depleted Vein (dark gray)
	4: '#39ff14',  // Xenoflora (neon green)
	5: '#1a1a2e',  // Obsidian Ridge (very dark)
};

export const CELL_NAMES: Record<number, string> = {
	10: 'Void', 11: 'Regolith', 0: 'Empty',
	1: 'Crystal Node', 2: 'Refinery', 3: 'Depleted Vein',
	4: 'Xenoflora', 5: 'Obsidian Ridge'
};
