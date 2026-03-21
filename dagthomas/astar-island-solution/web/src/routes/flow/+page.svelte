<script lang="ts">
	import FlowCanvas from '$lib/components/flow/FlowCanvas.svelte';
	import NodeDetail from '$lib/components/flow/NodeDetail.svelte';
	import type { FlowState, NodeStatus } from '$lib/components/flow/flow-types';
	import type {
		ProcessInfo,
		DaemonAutoloop,
		DaemonParams,
		DaemonRoundScore,
		Metrics,
		Budget,
		AutoloopEntry,
	} from '$lib/types';
	import { invalidateAll } from '$app/navigation';
	import { formatTime } from '$lib/api';
	import { simulateTrigger } from '$lib/stores/simulate';

	interface PageData {
		processes: ProcessInfo[];
		autoloop: DaemonAutoloop | null;
		params: DaemonParams | null;
		scores: DaemonRoundScore[];
		status: string[];
		metrics: Metrics | null;
		budget: Budget | null;
	}

	let { data }: { data: PageData } = $props();

	// Auto-refresh every 5 seconds
	$effect(() => {
		const interval = setInterval(() => {
			invalidateAll();
		}, 5000);
		return () => clearInterval(interval);
	});

	let selectedNodeId = $state<string | null>(null);

	// Watch clock clicks from StatusBar
	let lastTrigger = $state(0);
	$effect(() => {
		const unsub = simulateTrigger.subscribe(val => {
			if (val > 0 && val !== lastTrigger) {
				lastTrigger = val;
				startSimulation();
			}
		});
		return unsub;
	});

	// === SIMULATION MODE ===
	let simulating = $state(false);
	let simPhase = $state(0); // 0=idle, 1..N = timed phases
	let simScore = $state(0);
	let simSeedScores = $state<number[]>([]);
	let simStatusText = $state('');
	let simFlash = $state(''); // node ID currently "flashing"

	// Simulation timeline: [delay_ms, phase_name, active_nodes[], status_text]
	const SIM_TIMELINE: [number, string, string[], string][] = [
		[0,    'boot',        ['api'],                                        'Connecting to api.ainm.no...'],
		[800,  'api_up',      ['api'],                                        'API SERVER online. Checking rounds...'],
		[1200, 'daemon',      ['api', 'daemon'],                              'DAEMON initialized. Scanning for active rounds...'],
		[1800, 'round',       ['api', 'daemon'],                              'Round 17 detected! 8 settlements, 3 ports. Regime: BOOM'],
		[2400, 'explore',     ['api', 'daemon', 'exploration'],               'EXPLORATION: deploying 9-viewport grid strategy...'],
		[3200, 'explore2',    ['api', 'daemon', 'exploration'],               'EXPLORATION: 45/50 queries used. Coverage: 97.4%'],
		[4000, 'caldata',     ['api', 'daemon', 'exploration', 'calibration_data'], 'Loading calibration data: 16 rounds, 80k cells...'],
		[4500, 'calmodel',    ['api', 'daemon', 'calibration_data', 'cal_model'],   'CalibModel: 14,200 fine-level priors loaded'],
		[5000, 'params',      ['api', 'daemon', 'cal_model', 'best_params'],  'Loading best_params.json (avg=93.87)'],
		[5500, 'pred1',       ['api', 'daemon', 'best_params', 'pred_1'],     'Stage 1: Building feature key grid (40x40)...'],
		[6000, 'pred2',       ['api', 'daemon', 'pred_1', 'pred_2'],          'Stage 2: Cal prior lookup (fine->coarse->base->global)'],
		[6400, 'pred3',       ['api', 'daemon', 'pred_2', 'pred_3'],          'Stage 3: FK empirical distributions from observations'],
		[6800, 'pred4',       ['api', 'daemon', 'pred_3', 'pred_4'],          'Stage 4: FK blending (prior*0.972 + emp*sqrt(n))'],
		[7200, 'pred5',       ['api', 'daemon', 'pred_4', 'pred_5'],          'Stage 5: Global multipliers (6 class ratios)'],
		[7600, 'pred6',       ['api', 'daemon', 'pred_5', 'pred_6'],          'Stage 6: Temperature sharpening (T=1.035)'],
		[8000, 'pred7',       ['api', 'daemon', 'pred_6', 'pred_7'],          'Stage 7: Spatial smoothing (alpha=0.152)'],
		[8400, 'pred8',       ['api', 'daemon', 'pred_7', 'pred_8'],          'Stage 8: Structural zeros (mtn/ocean locks)'],
		[8800, 'pred9',       ['api', 'daemon', 'pred_8', 'pred_9'],          'Stage 9: Floor enforcement (min 0.005)'],
		[9400, 'submit',      ['api', 'daemon', 'pred_9', 'submission'],      'SUBMITTING 40x40x6 probability tensor...'],
		[10200,'submit2',     ['api', 'daemon', 'submission'],                'Seed 0: submitted. Seed 1: submitted. Seed 2: submitted...'],
		[11000,'scoring',     ['api', 'submission'],                          'All 5 seeds submitted. Waiting for scores...'],
		[12000,'score1',      ['api'],                                        ''],
		[12800,'research',    ['api', 'daemon', 'autoloop', 'gemini', 'multi', 'fast_harness', 'autoloop_log', 'research_logs'], 'Research agents activated. Autoloop: 730k exp/hr'],
		[14000,'fullsystem',  ['api', 'daemon', 'autoloop', 'gemini', 'multi', 'fast_harness', 'cal_model', 'best_params', 'calibration_data', 'autoloop_log', 'research_logs', 'pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9'], 'FULL SYSTEM ONLINE. All nodes active.'],
		[16000,'done',        [],                                             ''],
	];

	function startSimulation() {
		if (simulating) return;
		simulating = true;
		simPhase = 0;
		simScore = 0;
		simSeedScores = [];
		simStatusText = '';
		simFlash = '';

		// Schedule each phase
		for (let i = 0; i < SIM_TIMELINE.length; i++) {
			const [delay, name, , status] = SIM_TIMELINE[i];
			setTimeout(() => {
				simPhase = i;
				simStatusText = status;
				// Flash the newest node
				const nodes = SIM_TIMELINE[i][2];
				simFlash = nodes[nodes.length - 1] || '';

				// Score reveal phase
				if (name === 'score1') {
					simSeedScores = [95.14, 93.87, 94.62, 96.31, 93.28];
					simScore = 94.64;
					simStatusText = 'SCORES IN: 95.14 | 93.87 | 94.62 | 96.31 | 93.28  AVG: 94.64';
				}
				if (name === 'done') {
					simulating = false;
					simPhase = 0;
					simFlash = '';
				}
			}, delay);
		}
	}

	// Build sim overrides
	let simStatuses = $derived.by((): Record<string, NodeStatus> => {
		if (!simulating || simPhase >= SIM_TIMELINE.length) return {};
		const activeNodes = SIM_TIMELINE[simPhase][2];
		const s: Record<string, NodeStatus> = {};
		for (const id of activeNodes) {
			s[id] = 'active';
		}
		return s;
	});

	// Derive flow state from API data, merged with simulation overrides
	let flowState = $derived<FlowState>({
		nodeStatuses: simulating
			? { ...buildNodeStatuses(data), ...Object.fromEntries(Object.keys(buildNodeStatuses(data)).map(k => [k, 'idle' as NodeStatus])), ...simStatuses }
			: buildNodeStatuses(data),
		nodeMetrics: buildNodeMetrics(data),
	});

	// Sparkline data for autoloop detail
	let sparklineData = $derived(
		(data.autoloop?.top_params ?? [])
			.filter((e: AutoloopEntry) => e.accepted)
			.slice(-30)
			.map((e: AutoloopEntry) => {
				const vals = Object.values(e.scores_quick || {});
				return vals.length ? vals.reduce((a: number, b: number) => a + b, 0) / vals.length : 0;
			})
			.filter((v: number) => v > 0)
	);

	function buildNodeStatuses(d: PageData): Record<string, NodeStatus> {
		const statuses: Record<string, NodeStatus> = {};
		const procs = d.processes || [];
		const statusLines = (d.status || []).join(' ').toLowerCase();

		for (const p of procs) {
			const mapping: Record<string, string> = {
				daemon: 'daemon',
				autoloop: 'autoloop',
				gemini: 'gemini',
				multi: 'multi',
			};
			for (const [procName, nodeId] of Object.entries(mapping)) {
				if (p.name === procName) {
					statuses[nodeId] = p.state === 'Running' ? 'active' : 'idle';
				}
			}
		}

		statuses['api'] = d.budget ? 'active' : 'idle';
		statuses['exploration'] = statusLines.includes('explor') ? 'active' : 'idle';
		const predActive = statusLines.includes('predict') ? 'active' : 'idle';
		statuses['prediction'] = predActive;
		// All 9 prediction stages follow the prediction status
		for (let i = 1; i <= 9; i++) {
			statuses[`pred_${i}`] = predActive;
		}
		statuses['submission'] = statusLines.includes('submit') ? 'active' : 'idle';
		statuses['fast_harness'] = statuses['autoloop'] === 'active' ? 'active' : 'idle';
		statuses['cal_model'] = 'idle';
		statuses['best_params'] = d.params ? 'active' : 'idle';
		statuses['calibration_data'] = 'idle';
		statuses['autoloop_log'] = statuses['autoloop'] === 'active' ? 'active' : 'idle';
		statuses['research_logs'] = (statuses['gemini'] === 'active' || statuses['multi'] === 'active') ? 'active' : 'idle';

		return statuses;
	}

	function buildNodeMetrics(d: PageData): Record<string, Record<string, string | number>> {
		const m: Record<string, Record<string, string | number>> = {};

		m['api'] = {
			queries: d.budget ? `${d.budget.queries_used}/${d.budget.queries_max}` : '\u2014',
			endpoint: 'astar-island',
		};
		m['daemon'] = {
			status: d.status?.length ? d.status[d.status.length - 1]?.substring(0, 40) || 'OK' : 'No logs',
			log_lines: d.status?.length ?? 0,
		};
		const al = d.autoloop;
		m['autoloop'] = {
			experiments: al?.total_experiments ?? 0,
			best_score: al?.best_score ?? 0,
			rate: al?.experiments_per_hour ?? 0,
		};
		m['gemini'] = { proposals: d.metrics?.gemini_count ?? 0, improvements: '\u2014' };
		m['multi'] = { proposals: d.metrics?.multi_count ?? 0, improvements: '\u2014' };
		m['fast_harness'] = { speed: '~70ms/eval', cached: 'vectorized numpy' };
		m['cal_model'] = { levels: 'fine\u2192coarse\u2192base\u2192global' };
		m['best_params'] = {
			score: d.params?.score_avg ?? 0,
			updated: d.params?.updated_at ? formatTime(d.params.updated_at) : '\u2014',
		};
		m['calibration_data'] = { rounds: 'R1\u2013R15 ground truth' };
		m['autoloop_log'] = { entries: al?.total_experiments ?? 0 };
		m['research_logs'] = { total: (d.metrics?.gemini_count ?? 0) + (d.metrics?.multi_count ?? 0) };
		m['exploration'] = { strategy: '9-viewport grid', viewports: '45 queries / round' };
		m['pred_1'] = { desc: 'terrain+sett\u2192fkey grid' };
		m['pred_2'] = { desc: 'fine\u2192coarse\u2192base\u2192global' };
		m['pred_3'] = { desc: 'observation distributions' };
		m['pred_4'] = { desc: 'prior\u00d7w + emp\u00d7\u221acount' };
		m['pred_5'] = { desc: 'observed/expected ratio' };
		m['pred_6'] = { desc: 'entropy-weighted T' };
		m['pred_7'] = { desc: 'spatial \u03b1=0.15' };
		m['pred_8'] = { desc: 'mtn/ocean locks' };
		m['pred_9'] = { desc: 'min 0.005 floor' };
		m['submission'] = { tensor: '40\u00d740\u00d76 probabilities' };

		return m;
	}

	// =============================================
	// ANIMATED EFFECTS
	// =============================================

	// Master tick (100ms intervals)
	let tickCount = $state(0);
	$effect(() => {
		const interval = setInterval(() => { tickCount++; }, 100);
		return () => clearInterval(interval);
	});

	// Fast tick for typing effects (50ms)
	let fastTick = $state(0);
	$effect(() => {
		const interval = setInterval(() => { fastTick++; }, 50);
		return () => clearInterval(interval);
	});

	// === LLM STREAMING TEXT ===
	const llmResponses = [
		"Analyzing settlement expansion patterns... The sigmoid decay function shows optimal gamma=3.2 for coastal settlements, but inland clusters benefit from a sharper falloff (gamma=2.1). This suggests terrain-dependent parameterization could improve boom-round predictions by ~0.8 points. Current best: 93.41.",
		"Hypothesis: Adding forest_neighbor count to the global multiplier computation should better capture the forest-to-settlement transition dynamics observed in rounds 6,7,11. The current model treats forest proximity as binary — switching to a continuous gradient could push avg past 94.",
		"BREAKTHROUGH: Backtest results: avg=93.87 (+0.46). The entropy-weighted temperature adjustment at stage 6 massively improves non-boom rounds (R2:+1.2, R5:+0.9, R12:+0.7) while maintaining boom performance. This is the biggest single improvement this week. ACCEPTING.",
		"Proposing structural change: Replace linear FK blending with adaptive weighting based on observation count confidence intervals. When count < 10, the empirical distribution is unreliable — increase prior weight exponentially. Estimated gain: +0.15 avg.",
		"Code generation: def experimental_pred_fn(state, global_mult, fk_buckets):\n    # Compute adaptive prior weight\n    conf = 1.0 - np.exp(-count / tau)\n    pred = prior * (1-conf) + empirical * conf\n    # NEW: terrain-aware smoothing\n    if terrain == FOREST: pred *= forest_decay(dist)",
		"Running multi-seed validation across 15 rounds... R1:95.2 R2:92.8 R3:94.1 R4:96.3 R5:91.7 R6:93.4 R7:94.8 R8:95.1 R9:93.2 R10:94.6 R11:92.9 R12:95.4 R13:93.8 R14:94.2 R15:95.7. NEW ALL-TIME HIGH: avg=93.95!",
		"Investigating settlement survival model... Current base_survival=0.82 appears suboptimal for high-population settlements (pop>5). Fitting per-population survival curves from ground truth data. R2-R15 analysis shows pop>5 settlements survive 94.3% of the time vs 71.2% for pop<2.",
		"Architecture proposal: Add a secondary prediction pass that uses settlement survival probabilities to re-weight the ruin vs settlement channels. The current model under-predicts ruin for isolated low-pop settlements. Backtest shows +0.22 on collapse rounds specifically.",
		"URGENT: Round 17 detected! Regime analysis: 8 settlements, 3 ports, high forest density (34%). This looks like a BOOM round. Deploying optimized boom parameters: T_high=1.035, prior_w=0.972, exp_damp=0.385. Submitting predictions for all 5 seeds...",
		"Observation analysis complete: seed 0 shows unusual mountain clustering in NW quadrant. Settlement at (7,10) has port access AND forest adjacency — rare combination seen only in R3 and R11 (both scored 96+). Adjusting local priors for this cell pattern.",
		"Feature engineering: Testing new feature key (terrain, dist, coastal, forest_n, port, cluster_size, elevation_gradient). The elevation_gradient captures terrain transitions that correlate with settlement expansion direction. Backtest: avg=93.72 (+0.08). Marginal but consistent.",
		"META-ANALYSIS: After 470k autoloop experiments, the parameter landscape is nearly exhausted. Top 100 parameter sets all score within 0.03 of each other. Structural changes (code modifications) are now the primary vector for improvement. Switching research focus to pipeline stages 5-7.",
		"Calibration model update: Added round 15 and 16 ground truth data. Fine-level prior table expanded from 12,400 to 14,200 entries. Cross-validation score improved from 93.41 to 93.58 just from additional training data. The model continues to benefit from more calibration rounds.",
		"MULTI-AGENT CONSENSUS: Both Gemini and Claude agree that the spatial smoothing stage (stage 7) is the weakest link. Current uniform alpha=0.15 doesn't account for terrain boundaries. Proposed: adaptive alpha based on terrain homogeneity in 3x3 neighborhood. Expected gain: +0.3.",
		"Deploying growth front propagation model... Settlements expand along coastal paths with 78% probability, through plains at 45%, into forest at 12%. The growth front map from observations gives us a directional prior that significantly outperforms the isotropic distance decay.",
	];

	let llmIdx = $derived(Math.floor(fastTick / 120) % llmResponses.length);
	let llmChars = $derived(Math.min((fastTick % 120) * 3, llmResponses[llmIdx].length));
	let llmText = $derived(llmResponses[llmIdx].substring(0, llmChars));

	// === CODE SNIPPETS (fast rotating) ===
	const codeLines = [
		'pred = prior * w + empirical * sqrt(count)',
		'sigmoid_decay = 1/(1 + exp(beta*(d-gamma)))',
		'kl = sum(gt * log(gt / pred))',
		'score = 100 * exp(-3 * weighted_kl)',
		'coastal_mask = adjacent_to_ocean(terrain)',
		'cluster = settlement_density(grid, r=3)',
		'temp = T_low + (T_high-T_low) * entropy',
		'floor_nonzero = max(pred, 0.005)',
		'fk = (terrain, dist, coastal, forest_n, port)',
		'ratio = observed[c] / expected[c]',
		'pred[mountain] = [0,0,0,0,0,1]',
		'pred[ocean] = [1,0,0,0,0,0]',
		'smooth = alpha * neighbor_avg + (1-alpha) * pred',
		'prior = cal.prior_for(feature_key)',
		'harness.evaluate(params)  # ~70ms',
		'growth_front = propagate(settlements, terrain)',
		'survival = base_surv ** (1 / (pop + 0.5))',
		'obs_overlay = merge_viewports(obs_list)',
		'sett_surv = track_alive_dead(obs, init)',
		'adaptive_alpha = homogeneity(grid, 3)',
		'conf_interval = 1 - exp(-count / tau)',
		'regime = classify(sett_pct, pop_growth)',
		'forest_decay = exp(-0.3 * forest_dist)',
		'port_bonus = 1.15 if has_port else 1.0',
		'elevation_grad = sobel(terrain_height)',
		'dirichlet_params = fit_alpha(gt_data)',
		'calibrated = softmax(logits / T_high)',
		'wbf_merge(boxes, scores, iou=0.55)',
		'ensemble = mean([m1, m2, m3], axis=0)',
		'submit_tensor(pred, round_id, seed)',
	];
	let codeIdx = $derived(Math.floor(tickCount / 12) % codeLines.length);
	let codeChars = $derived(tickCount % 12);
	let currentCode = $derived(codeLines[codeIdx].substring(0, codeChars * 5));

	// === PARAMETER JITTER (fake real-time updates) ===
	let jitterParams = $derived({
		prior_w: (data.params?.prior_w ?? 0.9718) + (Math.sin(tickCount * 0.3) * 0.0003),
		T_high: (data.params?.T_high ?? 1.0352) + (Math.cos(tickCount * 0.2) * 0.0002),
		floor: (data.params?.floor ?? 0.00461) + (Math.sin(tickCount * 0.5) * 0.000008),
		exp_damp: (data.params?.exp_damp ?? 0.3847) + (Math.cos(tickCount * 0.4) * 0.0003),
		smooth_alpha: (data.params?.smooth_alpha ?? 0.1523) + (Math.sin(tickCount * 0.35) * 0.0002),
		base_power: (data.params?.base_power ?? 1.0124) + (Math.cos(tickCount * 0.25) * 0.0001),
	});

	// === EXPERIMENT COUNTER ANIMATION ===
	let displayedExperiments = $state(0);
	$effect(() => {
		const target = data.autoloop?.total_experiments ?? 0;
		if (target > displayedExperiments) {
			const step = Math.max(1, Math.floor((target - displayedExperiments) / 20));
			const interval = setInterval(() => {
				displayedExperiments = Math.min(displayedExperiments + step, target);
				if (displayedExperiments >= target) clearInterval(interval);
			}, 50);
			return () => clearInterval(interval);
		}
	});

	// === HEX DATA STREAM ===
	function randomHex(len: number): string {
		const chars = '0123456789abcdef';
		let result = '';
		// Use tickCount as a seed for deterministic-looking randomness
		for (let i = 0; i < len; i++) {
			const idx = ((tickCount * 7 + i * 13) * 31) % 16;
			result += chars[idx];
		}
		return result;
	}

	// === FAKE LOG ENTRIES ===
	const logTemplates = [
		'[AUTOLOOP] iter={iter} score={score} delta=+{delta} ACCEPTED',
		'[AUTOLOOP] iter={iter} score={score2} delta=-0.003 rejected',
		'[AUTOLOOP] iter={iter} prior_w={pw} T_high={th} ACCEPTED +{delta}',
		'[AUTOLOOP] iter={iter} new best! avg={score} boom=95.2 nonboom=92.8',
		'[AUTOLOOP] streak=847 no improvement, widening search radius',
		'[DAEMON]   round_check: R17 active, 42/50 queries used',
		'[DAEMON]   submit seed 0: score=95.14 (predicted 94.8)',
		'[DAEMON]   submit seed 3: score=93.87 regime=BOOM',
		'[DAEMON]   sync_best_params: avg 93.41 -> 93.87 (+0.46)',
		'[DAEMON]   calibration: downloaded R16 ground truth (5 seeds)',
		'[DAEMON]   autoloop restarted with new calibration data',
		'[HARNESS]  evaluate: 68ms  rounds=16  seeds=5  cells=128000',
		'[HARNESS]  cache hit: 94.2% (pre-computed terrain arrays)',
		'[HARNESS]  batch eval: 50 params in 3.4s (68ms/eval)',
		'[GEMINI]   proposal #{prop}: adaptive_forest_decay',
		'[GEMINI]   proposal #{prop}: coastal_growth_propagation',
		'[GEMINI]   proposal #{prop}: settlement_survival_model',
		'[GEMINI]   backtest: avg=93.72 (+0.08) ACCEPTED',
		'[GEMINI]   backtest: avg=93.95 (+0.31) NEW RECORD!',
		'[GEMINI]   proposal #{prop}: terrain_boundary_smoothing',
		'[MULTI]    flash analysis: 1.8s  direction: spatial_smoothing',
		'[MULTI]    opus code gen: adaptive_alpha_smoothing.py (42 lines)',
		'[MULTI]    backtest: avg=93.64 (+0.02) marginal, rejected',
		'[MULTI]    backtest: avg=94.12 (+0.50) BREAKTHROUGH!',
		'[MULTI]    consensus: both models agree on stage 7 weakness',
		'[CALIBR]   loaded 16 rounds, 80000 cells, 1600 settlements',
		'[CALIBR]   fine-level priors: 14,200 feature key entries',
		'[CALIBR]   cross-val score: 93.58 (was 93.41, +0.17)',
		'[PREDICT]  pipeline: 9 stages, 40x40x6 tensor, 12ms total',
		'[PREDICT]  stage 5 (global mult): 6 class ratios computed',
		'[PREDICT]  stage 7 (smoothing): 847 cells adjusted',
		'[SUBMIT]   R17 seed 2: 200 dynamic cells, entropy=1.82',
		'[SUBMIT]   R17 all seeds submitted, waiting for scores...',
		'[EXPLORE]  viewport (12,12): 15x15 grid, 3 settlements found',
		'[EXPLORE]  coverage: 97.4% (389/400 cells observed)',
	];

	let logLines = $derived(() => {
		const lines: string[] = [];
		for (let i = 0; i < 10; i++) {
			const tmplIdx = (tickCount + i * 3) % logTemplates.length;
			let line = logTemplates[tmplIdx];
			line = line
				.replace('{iter}', String(displayedExperiments + i))
				.replace('{score}', (93.4 + Math.sin((tickCount + i) * 0.1) * 0.5).toFixed(3))
				.replace('{score2}', (93.1 + Math.cos((tickCount + i) * 0.15) * 0.4).toFixed(3))
				.replace('{delta}', (Math.abs(Math.sin(tickCount * 0.2)) * 0.15 + 0.01).toFixed(3))
				.replace('{prop}', String(Math.floor(tickCount / 20) + i))
				.replace('{pw}', (0.970 + Math.sin(tickCount * 0.1) * 0.003).toFixed(4))
				.replace('{th}', (1.028 + Math.cos(tickCount * 0.15) * 0.005).toFixed(4));
			lines.push(line);
		}
		return lines;
	});

</script>

<div class="h-full flex flex-col overflow-hidden relative">
	<!-- Header -->
	<div class="flex items-center justify-between px-3 py-2 flex-shrink-0 z-10 relative">
		<div class="flex items-center gap-4">
			<div>
				<h1 class="text-sm text-neon-cyan neon-text tracking-[0.2em] uppercase">System Architecture</h1>
				<p class="text-[11px] text-cyber-muted">Real-time component visualization &middot; Astar Island Automation</p>
			</div>
		</div>
		<div class="flex items-center gap-3">
			<!-- Live code stream ticker -->
			<div class="glass px-3 py-1.5 text-xs font-mono max-w-[280px] overflow-hidden" style="border-color: rgba(255, 0, 255, 0.2);">
				<span class="text-neon-magenta/60">&gt;&gt;&gt; </span>
				<span class="text-cyber-fg">{currentCode}</span>
				<span class="animate-pulse-glow text-neon-cyan">|</span>
			</div>

			<!-- Live experiment counter -->
			<div class="glass px-3 py-1.5 text-center" style="border-color: rgba(0, 255, 240, 0.2);">
				<div class="text-xs text-cyber-muted uppercase tracking-wider">Experiments</div>
				<div class="text-sm font-bold text-neon-cyan tabular-nums">{displayedExperiments.toLocaleString()}</div>
			</div>

			<!-- Score with jitter -->
			{#if data.params?.score_avg}
				<div class="glass px-3 py-1.5 text-center" style="border-color: rgba(255, 215, 0, 0.2);">
					<div class="text-xs text-cyber-muted uppercase tracking-wider">Best Score</div>
					<div class="text-sm font-bold neon-text-gold tabular-nums" style="color: var(--color-neon-gold);">
						{data.params.score_avg.toFixed(3)}
					</div>
				</div>
			{/if}

			<!-- Hex stream -->
			<div class="glass px-2 py-1.5 text-xs font-mono text-neon-cyan/30 w-[90px] overflow-hidden" style="border-color: rgba(0, 255, 240, 0.08);">
				<div>0x{randomHex(8)}</div>
				<div>0x{randomHex(8)}</div>
			</div>

			<div class="text-[11px] text-cyber-muted flex items-center gap-1.5">
				<div class="w-1.5 h-1.5 rounded-full bg-score-great animate-pulse-glow"></div>
				LIVE
			</div>
		</div>
	</div>

	<!-- Simulation overlay bar -->
	{#if simulating && simStatusText}
		<div class="absolute top-14 left-1/2 -translate-x-1/2 z-50 glass px-6 py-3 max-w-[700px] text-center transition-all duration-500"
			style="border-color: {simScore > 0 ? 'rgba(0, 230, 118, 0.5)' : 'rgba(0, 255, 240, 0.3)'}; box-shadow: 0 0 30px {simScore > 0 ? 'rgba(0, 230, 118, 0.15)' : 'rgba(0, 255, 240, 0.1)'};">
			<div class="text-[13px] font-mono {simScore > 0 ? 'text-score-great font-bold' : 'text-cyber-fg'}" style="{simScore > 0 ? 'text-shadow: 0 0 12px rgba(0, 230, 118, 0.5)' : ''}">
				{simStatusText}
			</div>
			{#if simScore > 0}
				<div class="flex items-center justify-center gap-3 mt-2">
					{#each simSeedScores as ss, i}
						<div class="text-xs">
							<span class="text-cyber-muted">S{i}:</span>
							<span class="text-score-great font-bold" style="text-shadow: 0 0 8px rgba(0,230,118,0.4)">{ss.toFixed(2)}</span>
						</div>
					{/each}
					<div class="text-sm font-bold text-neon-gold neon-text-gold ml-2">AVG: {simScore.toFixed(2)}</div>
				</div>
			{/if}
		</div>
	{/if}

	<!-- Main content: Canvas + side panels -->
	<div class="flex-1 relative min-h-0 z-10">
		<FlowCanvas
			{flowState}
			{selectedNodeId}
			onSelectNode={(id) => { selectedNodeId = id; }}
		/>

		<!-- Detail panel overlay -->
		{#if selectedNodeId}
			<NodeDetail
				nodeId={selectedNodeId}
				{flowState}
				{sparklineData}
				onclose={() => { selectedNodeId = null; }}
			/>
		{/if}

		<!-- ==========================================
		     FLOATING CYBERPUNK OVERLAYS
		     ========================================== -->

		<!-- LLM Stream Panel (top-right) -->
		<div class="absolute top-3 right-3 z-20 glass p-3 w-[320px] opacity-50 hover:opacity-95 transition-opacity" style="border-color: rgba(255, 0, 255, 0.15); max-height: 220px; overflow: hidden;">
			<div class="flex items-center gap-2 mb-2">
				<div class="w-1.5 h-1.5 rounded-full bg-neon-magenta animate-pulse-glow"></div>
				<span class="text-[11px] text-neon-magenta/70 uppercase tracking-wider">Gemini Stream</span>
			</div>
			<div class="text-[11px] text-cyber-fg/80 leading-relaxed font-mono">
				{llmText}<span class="animate-pulse-glow text-neon-magenta">&#x2588;</span>
			</div>
		</div>

		<!-- Live Log Stream (bottom-left above legend) -->
		<div class="absolute bottom-[140px] left-3 z-20 glass p-2 w-[460px] opacity-40 hover:opacity-90 transition-opacity" style="border-color: rgba(0, 255, 240, 0.1);">
			<div class="flex items-center gap-2 mb-1.5">
				<div class="w-1.5 h-1.5 rounded-full bg-neon-cyan animate-pulse-glow"></div>
				<span class="text-xs text-neon-cyan/60 uppercase tracking-wider">System Log</span>
			</div>
			<div class="space-y-0.5 font-mono text-xs">
				{#each logLines() as line}
					<div class="truncate {line.includes('ACCEPTED') ? 'text-score-great/70' : line.includes('rejected') ? 'text-score-bad/40' : line.includes('GEMINI') || line.includes('MULTI') ? 'text-neon-magenta/50' : 'text-cyber-muted/50'}">
						{line}
					</div>
				{/each}
			</div>
		</div>

		<!-- Parameter Telemetry (bottom-right) -->
		<div class="absolute bottom-3 right-[60px] z-20 glass p-2 opacity-35 hover:opacity-90 transition-opacity" style="border-color: rgba(255, 215, 0, 0.1);">
			<div class="text-xs text-neon-gold/60 uppercase tracking-wider mb-1">Param Telemetry</div>
			<div class="grid grid-cols-2 gap-x-4 gap-y-0.5 font-mono text-[11px]">
				<div class="text-cyber-muted">prior_w</div>
				<div class="text-cyber-fg tabular-nums">{jitterParams.prior_w.toFixed(6)}</div>
				<div class="text-cyber-muted">T_high</div>
				<div class="text-cyber-fg tabular-nums">{jitterParams.T_high.toFixed(6)}</div>
				<div class="text-cyber-muted">floor</div>
				<div class="text-cyber-fg tabular-nums">{jitterParams.floor.toFixed(6)}</div>
				<div class="text-cyber-muted">exp_damp</div>
				<div class="text-cyber-fg tabular-nums">{jitterParams.exp_damp.toFixed(6)}</div>
				<div class="text-cyber-muted">smooth_a</div>
				<div class="text-cyber-fg tabular-nums">{jitterParams.smooth_alpha.toFixed(6)}</div>
				<div class="text-cyber-muted">base_pow</div>
				<div class="text-cyber-fg tabular-nums">{jitterParams.base_power.toFixed(6)}</div>
			</div>
		</div>

		<!-- Floating score badges -->
		{#if data.autoloop}
			<div class="absolute top-3 left-[220px] z-20 flex gap-2">
				<div class="glass px-2 py-1 text-xs" style="border-color: rgba(0, 230, 118, 0.2);">
					<span class="text-cyber-muted">AVG</span>
					<span class="text-score-great font-bold ml-1 tabular-nums">{(data.autoloop.best_score ?? 0).toFixed(2)}</span>
				</div>
				<div class="glass px-2 py-1 text-xs" style="border-color: rgba(255, 138, 101, 0.2);">
					<span class="text-cyber-muted">BOOM</span>
					<span class="text-neon-orange font-bold ml-1 tabular-nums">{(data.autoloop.best_boom ?? 0).toFixed(2)}</span>
				</div>
				<div class="glass px-2 py-1 text-xs" style="border-color: rgba(0, 255, 240, 0.2);">
					<span class="text-cyber-muted">NON-BOOM</span>
					<span class="text-neon-cyan font-bold ml-1 tabular-nums">{(data.autoloop.best_nonboom ?? 0).toFixed(2)}</span>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	/* Extra animations for this page */
	@keyframes scan {
		0% { transform: translateY(-100%); }
		100% { transform: translateY(100vh); }
	}
</style>
