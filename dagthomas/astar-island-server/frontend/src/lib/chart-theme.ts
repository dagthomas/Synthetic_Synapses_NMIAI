import type { ChartOptions } from 'chart.js';

export const CYBER = {
	cyan: '#00fff0',
	magenta: '#ff00ff',
	gold: '#ffd700',
	orange: '#ff8a65',
	green: '#00e676',
	red: '#e53935',
	lime: '#39ff14',
	purple: '#7c4dff',
	muted: '#6b6b7b',
	fg: '#e0d6c2',
	bg: '#0a0a1a',
	surface: '#12121e',
	panel: '#1a1a2e',
	border: '#2d2d44'
};

export const TEAM_PALETTE = [
	'#00fff0', '#ff00ff', '#ffd700', '#ff8a65', '#00e676',
	'#e53935', '#39ff14', '#7c4dff', '#ff6e40', '#18ffff',
	'#f50057', '#76ff03', '#ffab00', '#536dfe', '#ff1744',
	'#64ffda', '#ea80fc', '#ffd740', '#ff9100', '#b2ff59'
];

export function scoreColor(score: number): string {
	if (score >= 80) return CYBER.green;
	if (score >= 60) return CYBER.lime;
	if (score >= 40) return CYBER.gold;
	if (score >= 20) return CYBER.orange;
	return CYBER.red;
}

export function scoreColorAlpha(score: number, alpha: number): string {
	if (score >= 80) return `rgba(0, 230, 118, ${alpha})`;
	if (score >= 60) return `rgba(57, 255, 20, ${alpha})`;
	if (score >= 40) return `rgba(255, 215, 0, ${alpha})`;
	if (score >= 20) return `rgba(255, 138, 101, ${alpha})`;
	return `rgba(229, 57, 53, ${alpha})`;
}

export function bucketColors(count: number): string[] {
	const colors: string[] = [];
	for (let i = 0; i < count; i++) {
		const score = (i / count) * 100;
		colors.push(scoreColor(score));
	}
	return colors;
}

const FONT_FAMILY = "'JetBrains Mono', monospace";

export const cyberChartDefaults: ChartOptions = {
	responsive: true,
	maintainAspectRatio: false,
	animation: { duration: 600 },
	plugins: {
		legend: {
			labels: {
				color: CYBER.fg,
				font: { family: FONT_FAMILY, size: 11 },
				padding: 12
			}
		},
		tooltip: {
			backgroundColor: CYBER.surface,
			borderColor: 'rgba(0,255,240,0.3)',
			borderWidth: 1,
			titleColor: CYBER.cyan,
			bodyColor: CYBER.fg,
			titleFont: { family: FONT_FAMILY },
			bodyFont: { family: FONT_FAMILY }
		}
	},
	scales: {
		x: {
			ticks: { color: CYBER.muted, font: { family: FONT_FAMILY, size: 10 } },
			grid: { color: 'rgba(0,255,240,0.06)' },
			border: { color: CYBER.border }
		},
		y: {
			ticks: { color: CYBER.muted, font: { family: FONT_FAMILY, size: 10 } },
			grid: { color: 'rgba(0,255,240,0.06)' },
			border: { color: CYBER.border }
		}
	}
};
