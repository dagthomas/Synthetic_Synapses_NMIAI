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
	10: '#1a5276', // Ocean
	11: '#f9e79f', // Plains
	0: '#f9e79f',  // Empty
	1: '#e74c3c',  // Settlement
	2: '#8e44ad',  // Port
	3: '#7f8c8d',  // Ruin
	4: '#27ae60',  // Forest
	5: '#bdc3c7',  // Mountain
};

export const CELL_NAMES: Record<number, string> = {
	10: 'Ocean', 11: 'Plains', 0: 'Empty',
	1: 'Settlement', 2: 'Port', 3: 'Ruin',
	4: 'Forest', 5: 'Mountain'
};
