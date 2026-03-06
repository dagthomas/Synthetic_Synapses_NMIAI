<script>
	import Grid from '$lib/components/Grid.svelte';
	import Grid3D from '$lib/components/Grid3D.svelte';

	let { data } = $props();
	let run = $derived(data.run);
	let rounds = $derived(data.rounds);

	const CELL = 28;
	const CELLW = Math.round(CELL * 4 / 3); // 4:3 wide cells
	const ICON_OFF = (CELLW - CELL) / 2;     // center icon in wider cell
	const BOT_COLORS = [
		'#00FF41', '#0DF0E3', '#FF0055', '#F39C12', '#A200FF',
		'#FFEA00', '#FF3300', '#0066FF', '#FFFFFF', '#FF00AA',
	];

	// Build lookup sets
	const wallSet = new Set(run.walls.map(w => `${w[0]},${w[1]}`));
	const shelfSet = new Set(run.shelves.map(s => `${s[0]},${s[1]}`));
	const itemMap = new Map();
	for (const item of run.items) {
		const key = `${item.position[0]},${item.position[1]}`;
		if (!itemMap.has(key)) itemMap.set(key, []);
		itemMap.get(key).push(item);
	}

	// Unique grocery types on this map (sorted, with count)
	const groceryTypes = (() => {
		const counts = new Map();
		for (const item of run.items) {
			counts.set(item.type, (counts.get(item.type) || 0) + 1);
		}
		return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([type, count]) => ({ type, count }));
	})();

	// State
	let viewMode = $state('crt'); // 'crt' or '3d'
	let currentRound = $state(0);
	let playing = $state(false);
	let speed = $state(5);
	let selectedBot = $state(null);
	let intervalId = $state(null);

	let roundData = $derived(rounds[currentRound] || null);
	let bots = $derived(roundData?.bots || []);
	let orders = $derived(roundData?.orders || []);
	let actions = $derived(roundData?.actions || []);
	let events = $derived(roundData?.events || []);
	let score = $derived(rounds[currentRound]?.score ?? 0);

	let botPositions = $derived(new Map(bots.map(b => [`${b.position[0]},${b.position[1]}`, b])));

	// Active/preview order item types for grid highlighting
	let activeTypes = $derived(new Set(
		orders.filter(o => o.status === 'active').flatMap(o => o.items_required)
	));
	let previewTypes = $derived(new Set(
		orders.filter(o => o.status === 'preview').flatMap(o => o.items_required)
	));

	// Build cumulative event history up to current round
	let eventHistory = $derived.by(() => {
		const history = [];
		for (let r = 0; r <= currentRound; r++) {
			const rd = rounds[r];
			if (!rd?.events) continue;
			for (const evt of rd.events) {
				history.push({ ...evt, round: rd.round_number });
			}
		}
		return history;
	});

	function play() {
		if (playing) return;
		playing = true;
		intervalId = setInterval(() => {
			if (currentRound < rounds.length - 1) {
				currentRound++;
			} else {
				pause();
			}
		}, 1000 / speed);
	}

	function pause() {
		playing = false;
		if (intervalId) {
			clearInterval(intervalId);
			intervalId = null;
		}
	}

	function stepForward() {
		pause();
		if (currentRound < rounds.length - 1) currentRound++;
	}

	function stepBack() {
		pause();
		if (currentRound > 0) currentRound--;
	}

	function setRound(r) {
		pause();
		currentRound = r;
	}

	function togglePlay() {
		if (playing) pause();
		else play();
	}

	function setSpeed(s) {
		speed = s;
		if (playing) {
			pause();
			play();
		}
	}

	function getBotAction(botId) {
		const a = actions.find(a => a.bot === botId);
		return a ? a.action : 'wait';
	}

	function getDeliveredMask(order) {
		const available = [...(order.items_delivered || [])];
		return order.items_required.map(item => {
			const idx = available.indexOf(item);
			if (idx !== -1) { available.splice(idx, 1); return true; }
			return false;
		});
	}

	function getItemTypeName(t) {
		return t.charAt(0).toUpperCase() + t.slice(1);
	}

	const ITEM_CLR = {
		milk: '#dfe6e9', bread: '#ffeaa7', eggs: '#fab1a0', butter: '#fdcb6e',
		cheese: '#f39c12', pasta: '#e17055', rice: '#dfe6e9', juice: '#8aa8b8',
		yogurt: '#a29bfe', cereal: '#e67e22', flour: '#b2bec3', sugar: '#dfe6e9',
		coffee: '#6d4c2a', tea: '#4dbd6a', oil: '#fdcb6e', salt: '#b2bec3',
		cream: '#dfe6e9', oats: '#d4a76a',
		apples: '#33FF00', bananas: '#556655', carrots: '#AA0000', lettuce: '#AA6677',
		onions: '#665544', peppers: '#FF6600', tomatoes: '#DDAA00',
	};

	const ITEM_ABBR = {
		milk: 'Mk', bread: 'Br', eggs: 'Eg', butter: 'Bu',
		cheese: 'Ch', pasta: 'Pa', rice: 'Ri', juice: 'Ju',
		yogurt: 'Yo', cereal: 'Ce', flour: 'Fl', sugar: 'Su',
		coffee: 'Co', tea: 'Te', oil: 'Oi', salt: 'Sa',
		apples: 'Ap', bananas: 'Ba', carrots: 'Ca', lettuce: 'Le',
		onions: 'On', peppers: 'Pe', tomatoes: 'To',
	};

	function getItemAbbr(t) { return ITEM_ABBR[t] || t.slice(0, 2).toUpperCase(); }
	function getItemColor(t) { return ITEM_CLR[t] || '#aaa'; }
	function getCyberIcon(t, size = 20) { const svgs = isNightmare ? NIGHTMARE_SVGS : CYBER_SVGS; return svgs[t] ? svgs[t].replace(/width="28"/, `width="${size}"`).replace(/height="28"/, `height="${size}"`) : ''; }

	const isNightmare = run.difficulty === 'nightmare';

	function getBotSvg(botId, color, size = 24) {
		if (isNightmare) {
			return `<svg width="${size}" height="${size}" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg" style="--goat-color: ${color};">
				<style>
					@keyframes floatGoat{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
					@keyframes eyeGlow{0%,100%{fill:var(--goat-color);filter:drop-shadow(0 0 2px var(--goat-color))}50%{fill:#FF0000;filter:drop-shadow(0 0 4px #FF0000)}}
					@keyframes breatheSnout{0%,100%{transform:scaleX(1)}50%{transform:scaleX(1.05)}}
					@keyframes shadowPulse{0%,100%{opacity:.2;transform:scale(1)}50%{opacity:.4;transform:scale(1.2)}}
					.fG{animation:floatGoat 3s infinite ease-in-out;transform-origin:center}
					.eG{animation:eyeGlow 1.5s infinite}
					.bS{animation:breatheSnout 2.5s infinite ease-in-out;transform-origin:14px 15px}
					.sP{animation:shadowPulse 3s infinite ease-in-out;transform-origin:14px 24px}
				</style>
				<rect width="28" height="28" fill="#050202" rx="4"/>
				<ellipse cx="14" cy="24" rx="7" ry="2" fill="#330000" class="sP"/>
				<g class="fG">
					<path d="M14 5 L16 11 L22 11 L17 15 L19 21 L14 17 L9 21 L11 15 L6 11 L12 11 Z" fill="none" stroke="#FF0000" stroke-width="0.5" opacity="0.2"/>
					<path d="M11 9 C8 4, 3 3, 2 8 C3 9, 6 8, 9 11 Z" fill="#110505" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M17 9 C20 4, 25 3, 26 8 C25 9, 22 8, 19 11 Z" fill="#110505" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M9 11.5 L3 15 L8 16 Z" fill="#110505" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M19 11.5 L25 15 L20 16 Z" fill="#110505" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
					<path class="bS" d="M9 11 L19 11 L16 20 L12 20 Z" fill="#0A0303" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
					<path d="M10 13 L12 14 L10 15 Z" fill="#FF0000" class="eG"/>
					<path d="M18 13 L16 14 L18 15 Z" fill="#FF0000" class="eG"/>
					<path d="M13 18 L14 19 L15 18" fill="none" stroke="${color}" stroke-width="1" stroke-linecap="round"/>
					<text x="14" y="25" text-anchor="middle" dominant-baseline="central" font-size="4.5" font-weight="900" fill="${color}" font-family="monospace">${botId}</text>
				</g>
			</svg>`;
		}
		return `<svg width="${size}" height="${size}" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg" style="--bot-color: ${color};">
			<style>
				@keyframes sbHover{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
				@keyframes sbScan{0%,10%{transform:translateX(-2px)}40%,60%{transform:translateX(2px)}90%,100%{transform:translateX(-2px)}}
				@keyframes sbExhaust{0%,100%{opacity:.8;transform:scaleY(1)}50%{opacity:.3;transform:scaleY(.5)}}
				@keyframes sbNode{0%,100%{fill:var(--bot-color)}50%{fill:#FF0055}}
				.sbH{animation:sbHover 3s infinite ease-in-out}
				.sbS{animation:sbScan 2s infinite ease-in-out}
				.sbE{animation:sbExhaust .1s infinite;transform-origin:top;transform-box:fill-box}
				.sbN{animation:sbNode 1.5s infinite steps(2,start)}
			</style>
			<rect width="28" height="28" fill="#0D1117" rx="4"/>
			<g class="sbH">
				<path d="M12 19V22M14 19V23M16 19V22" stroke="#0DF0E3" stroke-width="1.2" stroke-linecap="round" class="sbE"/>
				<line x1="14" y1="9" x2="14" y2="4" stroke="${color}" stroke-width="1.2"/>
				<circle cx="14" cy="4" r="1.5" fill="${color}" class="sbN"/>
				<rect x="5.5" y="11" width="2" height="4" fill="#161B22" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
				<rect x="20.5" y="11" width="2" height="4" fill="#161B22" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
				<path d="M9 9L19 9L21 15L7 15Z" fill="#161B22" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
				<path d="M7 15L11 19H17L21 15" fill="#161B22" stroke="${color}" stroke-width="1.2" stroke-linejoin="round"/>
				<rect x="10" y="11.5" width="8" height="3" fill="#0D1117" stroke="${color}" stroke-width="1"/>
				<rect x="13" y="12" width="2" height="2" fill="#FF0055" class="sbS"/>
				<text x="14" y="18" text-anchor="middle" dominant-baseline="central" font-size="5" font-weight="900" fill="${color}" font-family="monospace">${botId}</text>
			</g>
		</svg>`;
	}

	// Nightmare satanic SVG icons
	const NIGHTMARE_SVGS = {
		milk: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes bloodDrop{0%{transform:translateY(0) scale(1);opacity:1}80%{transform:translateY(8px) scale(0.8);opacity:1}100%{transform:translateY(10px) scale(0);opacity:0}}@keyframes shadow{0%,100%{opacity:.2}50%{opacity:.4}}.s{animation:shadow 3s infinite}.d{animation:bloodDrop 1.5s infinite linear}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="24" rx="7" ry="1.5" fill="#300" class="s"/><path d="M 8 7 C 8 7, 8 13, 14 16 C 20 13, 20 7, 20 7 Z" fill="#1A1111" stroke="#AA8844" stroke-width="1.2"/><path d="M 13 15.5 V 21 M 10 21 H 18" stroke="#AA8844" stroke-width="1.2" stroke-linecap="round"/><ellipse cx="14" cy="7" rx="6" ry="2" fill="#800" stroke="#AA8844" stroke-width="1.2"/><circle cx="14" cy="9" r="1.5" fill="#FF0000" class="d"/><path d="M 12 7 V 10" stroke="#FF0000" stroke-width="1.5" stroke-linecap="round"/></svg>`,
		bread: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes blink{0%,45%,55%,100%{transform:scaleY(1)}50%{transform:scaleY(0)}}@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.03)}}.b{animation:breathe 2.5s infinite ease-in-out;transform-origin:center}.e{animation:blink 4s infinite;transform-origin:14px 14px}</style><rect width="28" height="28" fill="#050202" rx="4"/><g class="b"><rect x="7" y="6" width="14" height="16" fill="#331A1A" stroke="#220505" stroke-width="1.2" rx="1"/><path d="M 10 6 V 22 M 18 6 V 22" stroke="#552222" stroke-width="1"/><ellipse cx="14" cy="14" rx="4" ry="2.5" fill="#110505" stroke="#880000" stroke-width="1.2"/><ellipse cx="14" cy="14" rx="1.5" ry="2.5" fill="#FFCC00" class="e"/><circle cx="14" cy="14" r="0.8" fill="#000" class="e"/><path d="M 7 10 H 9 M 7 18 H 9 M 19 10 H 21 M 19 18 H 21" stroke="#000" stroke-width="1.2"/></g></svg>`,
		eggs: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes twitch{0%,100%{transform:translate(0,0)}10%,30%{transform:translate(-1px,1px)}20%,40%{transform:translate(1px,-1px)}50%{transform:translate(0,0)}}.t{animation:twitch 3s infinite steps(2,start)}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><g class="t"><circle cx="11" cy="16" r="4.5" fill="#EEDDDD" stroke="#800" stroke-width="1"/><circle cx="17" cy="15" r="4" fill="#EEDDDD" stroke="#800" stroke-width="1"/><circle cx="14" cy="11" r="5" fill="#EEDDDD" stroke="#800" stroke-width="1"/><circle cx="10" cy="16" r="1.5" fill="#000"/><circle cx="18" cy="14" r="1" fill="#000"/><circle cx="14" cy="10" r="1.8" fill="#000"/><path d="M 14 6 C 13 8, 11 9, 10 10 M 17 11 C 18 12, 19 12, 20 13 M 9 14 C 8 15, 7 15, 6 16" stroke="#C00" stroke-width="0.5" fill="none"/></g></svg>`,
		butter: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes stab{0%,100%{transform:translate(4px,-4px);opacity:0}20%,80%{transform:translate(0,0);opacity:1}}.p{animation:stab 2s infinite ease-in}</style><rect width="28" height="28" fill="#050202" rx="4"/><path d="M 14 6 C 12 6, 11 8, 11 10 C 11 11, 12 12, 14 12 C 16 12, 17 11, 17 10 C 17 8, 16 6, 14 6 Z M 11 12 H 17 V 18 H 11 Z M 11 12 L 8 15 M 17 12 L 20 15 M 12 18 L 12 22 M 16 18 L 16 22" fill="none" stroke="#A86" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/><line x1="12" y1="9" x2="13" y2="10" stroke="#000" stroke-width="1"/><line x1="13" y1="9" x2="12" y2="10" stroke="#000" stroke-width="1"/><line x1="15" y1="9" x2="16" y2="10" stroke="#000" stroke-width="1"/><line x1="16" y1="9" x2="15" y2="10" stroke="#000" stroke-width="1"/><line x1="13" y1="15" x2="17" y2="15" stroke="#000" stroke-width="1" stroke-dasharray="1 1"/><line x1="18" y1="10" x2="14" y2="14" stroke="#F00" stroke-width="1" class="p"/><circle cx="18" cy="10" r="1.5" fill="#F00" class="p"/></svg>`,
		cheese: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes jaw{0%,100%{transform:translateY(0)}50%{transform:translateY(2px)}}@keyframes eye{0%,100%{opacity:.2}50%{opacity:1}}.j{animation:jaw 1.5s infinite steps(2,start);transform-origin:center}.e{animation:eye 3s infinite}</style><rect width="28" height="28" fill="#050202" rx="4"/><path d="M 9 10 C 9 6, 19 6, 19 10 C 19 14, 17 16, 17 16 H 11 C 11 16, 9 14, 9 10 Z" fill="#DDB" stroke="#332" stroke-width="1.2"/><path d="M 12 17 V 19 H 16 V 17 Z" fill="#DDB" stroke="#332" stroke-width="1.2" class="j"/><line x1="14" y1="17" x2="14" y2="19" stroke="#332" stroke-width="1" class="j"/><ellipse cx="12" cy="11" rx="1.5" ry="2" fill="#050202"/><ellipse cx="16" cy="11" rx="1.5" ry="2" fill="#050202"/><path d="M 14 14 L 13 15 H 15 Z" fill="#050202"/><circle cx="12" cy="11" r="0.5" fill="#F00" class="e"/><path d="M 12 6 L 14 9 M 18 8 L 16 10" stroke="#050202" stroke-width="1"/></svg>`,
		pasta: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes writhe{0%,100%{stroke-dashoffset:0}50%{stroke-dashoffset:4}}@keyframes tongue{0%,100%{opacity:0;transform:scaleX(0)}50%{opacity:1;transform:scaleX(1)}}.w{animation:writhe 2s infinite linear;stroke-dasharray:6 2}.t{animation:tongue 1s infinite steps(2,start);transform-origin:left}</style><rect width="28" height="28" fill="#050202" rx="4"/><path class="w" d="M 8 10 C 12 6, 16 14, 20 10 C 24 6, 12 20, 8 16 C 4 12, 10 20, 14 20" fill="none" stroke="#263" stroke-width="2.5" stroke-linecap="round"/><circle cx="8" cy="10" r="1.5" fill="#132"/><circle cx="14" cy="20" r="1.5" fill="#132"/><path d="M 8 10 L 5 9 L 4 10 M 5 9 L 6 7" fill="none" stroke="#F00" stroke-width="0.8" class="t"/><circle cx="8" cy="10" r="0.5" fill="#F00"/><circle cx="14" cy="20" r="0.5" fill="#F00"/></svg>`,
		rice: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes squirm1{0%,100%{transform:rotate(0)}50%{transform:rotate(15deg)}}@keyframes squirm2{0%,100%{transform:rotate(0)}50%{transform:rotate(-15deg)}}.s1{animation:squirm1 1s infinite ease-in-out;transform-origin:center}.s2{animation:squirm2 1.2s infinite ease-in-out;transform-origin:center}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="22" rx="7" ry="2" fill="#300"/><g fill="#DDCCBB" stroke="#443322" stroke-width="1"><rect x="10" y="10" width="8" height="3" rx="1.5" class="s1"/><rect x="12" y="14" width="7" height="3" rx="1.5" class="s2"/><rect x="7" y="15" width="6" height="3" rx="1.5" transform="rotate(45 10 16)" class="s1"/><rect x="15" y="8" width="6" height="3" rx="1.5" transform="rotate(-30 18 9)" class="s2"/><rect x="11" y="18" width="7" height="3" rx="1.5" transform="rotate(-15 14 19)" class="s1"/></g></svg>`,
		juice: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(2px)}}@keyframes bubble{0%{transform:translateY(0);opacity:1}100%{transform:translateY(-8px);opacity:0}}.b{animation:bob 3s infinite ease-in-out}.bu1{animation:bubble 2s infinite linear 0s}.bu2{animation:bubble 2s infinite linear 1s}</style><rect width="28" height="28" fill="#050202" rx="4"/><path d="M 8 7 H 20 V 23 H 8 Z" fill="#0A1A10" stroke="#335544" stroke-width="1.2"/><path d="M 7 6 H 21 V 8 H 7 Z" fill="#112211" stroke="#335544" stroke-width="1.2"/><g class="b"><ellipse cx="14" cy="14" rx="4" ry="5" fill="#667755" stroke="#223322" stroke-width="1"/><path d="M 12 13 L 13 14 M 16 13 L 15 14" stroke="#111" stroke-width="1"/><line x1="13" y1="17" x2="15" y2="17" stroke="#111" stroke-width="1"/></g><circle cx="10" cy="18" r="1" fill="#44FF88" class="bu1"/><circle cx="18" cy="20" r="1" fill="#44FF88" class="bu2"/><circle cx="12" cy="21" r="0.5" fill="#44FF88" class="bu1"/></svg>`,
		yogurt: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes beat{0%,100%{transform:scale(1)}15%{transform:scale(1.1)}30%{transform:scale(1)}}.ht{animation:beat 1s infinite;transform-origin:14px 14px}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><g class="ht"><path d="M 14 19 C 14 19, 8 14, 8 10 C 8 7, 11 6, 14 9 C 17 6, 20 7, 20 10 C 20 14, 14 19, 14 19 Z" fill="#800" stroke="#400" stroke-width="1.2"/><path d="M 14 9 V 5 M 12 7 V 4 M 16 8 V 6" stroke="#400" stroke-width="1.5" stroke-linecap="round"/><path d="M 14 9 C 12 12, 12 15, 14 19" fill="none" stroke="#500" stroke-width="1"/></g></svg>`,
		cereal: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}.f1{animation:float 2s infinite ease-in-out 0s}.f2{animation:float 2.2s infinite ease-in-out 0.5s}.f3{animation:float 1.8s infinite ease-in-out 1s}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path d="M 6 14 C 6 20, 22 20, 22 14 Z" fill="#222" stroke="#444" stroke-width="1.2"/><ellipse cx="14" cy="14" rx="8" ry="2" fill="#111"/><g fill="#DDD" stroke="#222" stroke-width="0.8"><rect x="10" y="11" width="4" height="2" rx="1" class="f1" transform="rotate(20 12 12)"/><rect x="15" y="10" width="4" height="2" rx="1" class="f2" transform="rotate(-30 17 11)"/><circle cx="13" cy="14" r="1.5" class="f3"/><circle cx="16" cy="13" r="1" class="f1"/><path class="f2" d="M 13 9 C 13 8, 15 8, 15 9 V 11 C 15 12, 13 12, 13 11 Z" transform="rotate(45 14 10)"/></g></svg>`,
		flour: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes drift{0%{transform:translate(0,0);opacity:1}100%{transform:translate(-4px,-8px) scale(2);opacity:0}}.a1{animation:drift 2s infinite linear 0s}.a2{animation:drift 2s infinite linear 0.7s}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path d="M 12 6 H 16 L 15 8 C 18 10, 19 15, 17 20 H 11 C 9 15, 10 10, 13 8 Z" fill="#2A2A30" stroke="#111" stroke-width="1.2"/><path d="M 10 14 C 14 16, 18 12, 18 14" fill="none" stroke="#111" stroke-width="1"/><circle cx="14" cy="6" r="1" fill="#777" class="a1"/><circle cx="15" cy="5" r="0.8" fill="#555" class="a2"/><circle cx="13" cy="4" r="1.2" fill="#999" class="a1" style="animation-delay: 1.2s"/></svg>`,
		sugar: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes glint{0%,80%{opacity:0}90%{opacity:1}100%{opacity:0}}@keyframes drip{0%{transform:translateY(0);opacity:1}100%{transform:translateY(6px);opacity:0}}.g{animation:glint 2s infinite linear}.d{animation:drip 1.5s infinite ease-in}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path d="M 14 2 L 15 6 L 14 18 L 13 6 Z" fill="#99A" stroke="#556" stroke-width="1" stroke-linejoin="round"/><path d="M 11 18 H 17 V 19 H 11 Z" fill="#DA4"/><path d="M 13 19 H 15 V 24 L 14 25 L 13 24 Z" fill="#311" stroke="#DA4" stroke-width="0.8"/><circle cx="14" cy="21" r="1" fill="#F00"/><path d="M 14 2 L 14 18" stroke="#FFF" stroke-width="0.5" class="g"/><circle cx="14" cy="18" r="1" fill="#800" class="d"/></svg>`,
		cream: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes tw{0%,90%{transform:rotate(0)}95%{transform:rotate(-5deg)}100%{transform:rotate(5deg)}}.tw{animation:tw 2s infinite;transform-origin:left bottom}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><g transform="rotate(-15 14 14)"><path d="M 6 16 C 6 12, 10 12, 12 12 H 16 V 20 H 12 C 10 20, 6 20, 6 16 Z" fill="#887777" stroke="#443333" stroke-width="1.2"/><path d="M 16 13 H 20 C 21 13, 21 15, 20 15 H 16 Z" fill="#887777" stroke="#443333" stroke-width="1" class="tw"/><path d="M 16 15 H 22 C 23 15, 23 17, 22 17 H 16 Z" fill="#887777" stroke="#443333" stroke-width="1" class="tw" style="animation-delay:0.1s"/><path d="M 16 17 H 21 C 22 17, 22 19, 21 19 H 16 Z" fill="#887777" stroke="#443333" stroke-width="1" class="tw" style="animation-delay:0.2s"/><path d="M 13 18 L 15 22 C 16 23, 14 24, 13 22 L 11 19 Z" fill="#887777" stroke="#443333" stroke-width="1"/><ellipse cx="6" cy="16" rx="1.5" ry="4" fill="#A00"/></g><circle cx="8" cy="21" r="1" fill="#A00"/><circle cx="5" cy="19" r="1" fill="#A00"/></svg>`,
		oats: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes pulseAura{0%,100%{opacity:.3;transform:scale(1)}50%{opacity:.6;transform:scale(1.1)}}@keyframes spinPenta{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}.a{animation:pulseAura 3s infinite ease-in-out;transform-origin:center}.sp{animation:spinPenta 10s infinite linear;transform-origin:center}</style><rect width="28" height="28" fill="#050202" rx="4"/><circle cx="14" cy="14" r="10" fill="none" stroke="#F00" stroke-width="2" class="a"/><g class="sp"><circle cx="14" cy="14" r="8" fill="none" stroke="#800" stroke-width="1"/><path d="M 14 6 L 16.5 20 L 4.5 11 H 23.5 L 11.5 20 Z" fill="none" stroke="#F00" stroke-width="1"/><circle cx="14" cy="14" r="2" fill="#800"/></g><circle cx="14" cy="4" r="1" fill="#FA0"/><circle cx="4.5" cy="11" r="1" fill="#FA0"/><circle cx="23.5" cy="11" r="1" fill="#FA0"/><circle cx="8.5" cy="22" r="1" fill="#FA0"/><circle cx="19.5" cy="22" r="1" fill="#FA0"/></svg>`,
		apples: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes wriggle{0%,100%{transform:rotate(-5deg)}50%{transform:rotate(10deg)}}@keyframes glow{0%,100%{filter:drop-shadow(0 0 2px #3F0)}50%{filter:drop-shadow(0 0 5px #3F0)}}.w{animation:wriggle 1.5s infinite ease-in-out;transform-origin:10px 14px}.g{animation:glow 2s infinite}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path d="M 14 8 C 21 8, 22 14, 20 19 C 19 22, 15 22, 14 20 C 13 22, 9 22, 8 19 C 6 14, 7 8, 14 8 Z" fill="#2A0505" stroke="#110202" stroke-width="1.2"/><path d="M 14 8 Q 15 4 18 5" fill="none" stroke="#111" stroke-width="1.5"/><ellipse cx="10" cy="14" rx="2" ry="3" fill="#050202"/><g class="w"><path d="M 10 14 Q 5 10 7 18" fill="none" stroke="#3F0" stroke-width="2" stroke-linecap="round" class="g"/><circle cx="7" cy="18" r="1.5" fill="#3F0" class="g"/><circle cx="7.5" cy="18" r="0.5" fill="#000"/></g></svg>`,
		bananas: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes fingerTwitch{0%,90%{transform:rotate(0)}95%{transform:rotate(-4deg)}100%{transform:rotate(2deg)}}.ft{animation:fingerTwitch 2.5s infinite;transform-origin:18px 20px}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><g class="ft"><path d="M 17 20 Q 12 18 8 10 L 6 12 Q 10 21 16 22 Z" fill="#4A554A" stroke="#223322" stroke-width="1"/><path d="M 18 20 Q 15 15 12 7 L 10 8 Q 14 18 17 22 Z" fill="#556655" stroke="#223322" stroke-width="1"/><path d="M 19 20 Q 19 14 18 6 L 16 6 Q 17 15 18 22 Z" fill="#334433" stroke="#223322" stroke-width="1"/><path d="M 8 10 L 5 7 L 6 12 Z M 12 7 L 10 3 L 10 8 Z M 18 6 L 19 2 L 16 6 Z" fill="#111"/><path d="M 15 18 H 21 V 23 H 15 Z" fill="#600" stroke="#300" stroke-width="1"/><path d="M 16 19 H 20 M 16 21 H 20" stroke="#200" stroke-width="1"/></g></svg>`,
		carrots: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes drip{0%{transform:translateY(0);opacity:1}80%{transform:translateY(8px);opacity:1}100%{transform:translateY(10px);opacity:0}}.d1{animation:drip 1.5s infinite linear}.d2{animation:drip 2s infinite linear 0.5s}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path d="M 8 6 H 12 L 10 20 Z" fill="#444" stroke="#222" stroke-width="1"/><path d="M 16 4 H 20 L 18 22 Z" fill="#444" stroke="#222" stroke-width="1"/><path d="M 12 8 H 16 L 14 24 Z" fill="#555" stroke="#222" stroke-width="1"/><path d="M 8 6 H 12 V 8 H 8 Z M 16 4 H 20 V 6 H 16 Z M 12 8 H 16 V 10 H 12 Z" fill="#222"/><path d="M 9 14 L 11 20 M 17 14 L 19 22 M 13 16 L 15 24" stroke="#A00" stroke-width="1.5"/><circle cx="10" cy="20" r="1.5" fill="#A00" class="d1"/><circle cx="18" cy="22" r="1.5" fill="#A00" class="d2"/><circle cx="14" cy="24" r="1.5" fill="#A00" class="d1" style="animation-delay: 1s"/></svg>`,
		lettuce: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes throb{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}.t{animation:throb 1.2s infinite ease-in-out;transform-origin:14px 14px}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path d="M 14 22 C 6 22, 4 16, 6 10 C 8 6, 20 6, 22 10 C 24 16, 22 22, 14 22 Z" fill="#1A2A1A" stroke="#051105" stroke-width="1.5"/><g class="t"><ellipse cx="14" cy="14" rx="6" ry="5" fill="#A67" stroke="#423" stroke-width="1.2"/><path d="M 14 9 V 19 M 11 10 C 13 12, 9 14, 11 16 M 17 10 C 15 12, 19 14, 17 16" fill="none" stroke="#634" stroke-width="1.2" stroke-linecap="round"/><circle cx="14" cy="14" r="5" fill="none" stroke="#F00" stroke-width="0.5" opacity="0.5"/></g></svg>`,
		onions: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes swing{0%,100%{transform:rotate(-5deg)}50%{transform:rotate(5deg)}}.s{animation:swing 2.5s infinite ease-in-out;transform-origin:14px 0px}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><g class="s"><line x1="14" y1="0" x2="14" y2="8" stroke="#543" stroke-width="1" stroke-dasharray="2 1"/><ellipse cx="14" cy="14" rx="5" ry="6" fill="#654" stroke="#321" stroke-width="1.2"/><path d="M 12 12 L 16 12" stroke="#210" stroke-width="1.5"/><path d="M 11 11 L 13 13 M 15 11 L 17 13 M 13 11 L 11 13 M 17 11 L 15 13" stroke="#111" stroke-width="1"/><path d="M 12 17 L 16 17" stroke="#111" stroke-width="1"/><line x1="12.5" y1="16" x2="12.5" y2="18" stroke="#111" stroke-width="1"/><line x1="14" y1="16" x2="14" y2="18" stroke="#111" stroke-width="1"/><line x1="15.5" y1="16" x2="15.5" y2="18" stroke="#111" stroke-width="1"/><path d="M 9 14 Q 7 18 8 22 M 19 14 Q 21 18 20 22 M 11 20 Q 14 24 17 20" fill="none" stroke="#111" stroke-width="1"/></g></svg>`,
		peppers: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes beatHeat{0%,100%{transform:scale(1);fill:#A00}15%{transform:scale(1.1);fill:#F00}30%{transform:scale(1);fill:#A00}}@keyframes flame{0%,100%{transform:scaleY(1);opacity:.8}50%{transform:scaleY(1.3);opacity:1}}.b{animation:beatHeat 1.2s infinite;transform-origin:14px 16px}.f{animation:flame .8s infinite alternate;transform-origin:14px 8px}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><path class="f" d="M 14 4 Q 12 8 14 10 Q 16 8 14 4 Z" fill="#FA0"/><g class="b"><path d="M 14 20 C 14 20, 9 14, 9 10 C 9 7, 13 7, 14 10 C 15 7, 19 7, 19 10 C 19 14, 14 20, 14 20 Z" fill="#A00" stroke="#400" stroke-width="1.2"/><path d="M 9 12 Q 14 15 19 11 M 10 16 Q 14 18 17 14" fill="none" stroke="#333" stroke-width="1.5"/><circle cx="14" cy="14" r="1.5" fill="#333"/><path d="M 14 14 L 12 12 M 14 14 L 16 16" stroke="#333" stroke-width="1.5"/></g></svg>`,
		tomatoes: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes dart{0%,10%,100%{transform:translate(0,0)}20%,40%{transform:translate(-2px,1px)}50%,70%{transform:translate(2px,-1px)}80%,90%{transform:translate(0,-2px)}}@keyframes pulseVein{0%,100%{stroke:#600}50%{stroke:#F00}}.d{animation:dart 3s infinite steps(2,start)}.v{animation:pulseVein 1.5s infinite}</style><rect width="28" height="28" fill="#050202" rx="4"/><ellipse cx="14" cy="23" rx="7" ry="2" fill="#300"/><ellipse cx="14" cy="14" rx="7" ry="6.5" fill="#EEDDDD" stroke="#511" stroke-width="1.2"/><path class="v" d="M 7 14 Q 10 12 11 14 M 21 14 Q 18 16 17 14 M 14 7 Q 16 10 14 11 M 14 21 Q 12 18 14 17" fill="none" stroke="#600" stroke-width="1"/><g class="d"><circle cx="14" cy="14" r="3" fill="#DA0"/><ellipse cx="14" cy="14" rx="1" ry="2.5" fill="#000"/><circle cx="15" cy="13" r="0.5" fill="#FFF"/></g></svg>`,
	};

	// Cyberpunk SVG icons — exact designs with embedded <style> animations
	const CYBER_SVGS = {
		milk: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes dashFlow{to{stroke-dashoffset:-12}}@keyframes blinkNode{0%,100%{opacity:1}50%{opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15;transform:scale(1)}50%{opacity:.25;transform:scale(1.1)}}.shadow{animation:shadowPulse 3s infinite ease-in-out;transform-origin:14px 23px}.flow{stroke-dasharray:4 4;animation:dashFlow 1.5s linear infinite}.dot{animation:blinkNode 1s steps(2,start) infinite}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M10 11 L14 7 L18 11 V21 C18 21.5 17.5 22 17 22 H11 C10.5 22 10 21.5 10 21 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round" opacity=".3"/><path d="M10 11 L14 7 L18 11 V21 C18 21.5 17.5 22 17 22 H11 C10.5 22 10 21.5 10 21 Z" fill="none" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round" class="flow"/><path d="M9 11 H19" stroke="#00FF41" stroke-width="1.2" stroke-linecap="round"/><rect x="12" y="14" width="4" height="4" fill="none" stroke="#0DF0E3" stroke-width="1"/><circle cx="14" cy="16" r="1" fill="#FF0055" class="dot"/></svg>`,
		bread: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes neonFlicker{0%,18%,22%,25%,53%,57%,100%{opacity:1}20%,24%,55%{opacity:.2}}@keyframes flowUp{to{stroke-dashoffset:-10}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.flicker{animation:neonFlicker 4s infinite}.score{stroke-dasharray:2 2;animation:flowUp 1.5s linear infinite}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path class="flicker" d="M 6 15 C 6 10, 22 10, 22 15 L 21 21 C 21 21.5, 20.5 22, 20 22 H 8 C 7.5 22, 7 21.5, 7 21 Z" fill="#161B22" stroke="#FF0055" stroke-width="1.2"/><path class="score" d="M 10 13 V 20 M 14 13 V 20 M 18 13 V 20" stroke="#0DF0E3" stroke-width="1.2" stroke-linecap="square"/></svg>`,
		eggs: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes nodePulse{0%,100%{fill:#0DF0E3;opacity:.3}50%{fill:#FF0055;opacity:1}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.n1{animation:nodePulse 1.5s infinite 0s}.n2{animation:nodePulse 1.5s infinite .5s}.n3{animation:nodePulse 1.5s infinite 1s}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><ellipse cx="10" cy="15" rx="3" ry="4" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><ellipse cx="14" cy="14" rx="3" ry="4" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><ellipse cx="18" cy="15" rx="3" ry="4" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 5 17 L 6 21 C 6 21.5, 6.5 22, 7 22 H 21 C 21.5 22, 22 21.5, 22 21 L 23 17 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/><circle cx="10" cy="15" r="1.5" class="n1"/><circle cx="14" cy="14" r="1.5" class="n2"/><circle cx="18" cy="15" r="1.5" class="n3"/></svg>`,
		butter: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes scanPass{0%,10%{transform:translateY(8px);opacity:0}15%,85%{opacity:1}90%,100%{transform:translateY(22px);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.scanner{animation:scanPass 2.5s infinite linear}</style><defs><clipPath id="bC"><path d="M 7 14 L 17 11 L 21 13 L 11 16 Z M 7 14 L 11 16 L 11 20 L 7 18 Z M 11 16 L 21 13 L 21 17 L 11 20 Z"/></clipPath></defs><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 7 14 L 17 11 L 21 13 L 11 16 Z M 7 14 L 11 16 L 11 20 L 7 18 Z M 11 16 L 21 13 L 21 17 L 11 20 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round"/><path d="M 7 14 L 13 12.5 L 13 16.5 L 7 18 Z" fill="none" stroke="#0DF0E3" stroke-width="1" stroke-dasharray="1 2"/><line x1="0" y1="0" x2="28" y2="0" stroke="#FF0055" stroke-width="1.5" clip-path="url(#bC)" class="scanner"/></svg>`,
		cheese: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes glitchMove{0%,93%{transform:translate(0,0);opacity:1}94%{transform:translate(-1.5px,0);stroke:#0DF0E3;opacity:.8}96%{transform:translate(1.5px,0);stroke:#00FF41;opacity:.8}98%{transform:translate(0,1.5px);stroke:#FF0055;opacity:.8}100%{transform:translate(0,0);opacity:1}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.glitch{animation:glitchMove 3s infinite steps(1)}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path class="glitch" d="M 7 19 L 21 21 L 18 10 Z" fill="#161B22" stroke="#FF0055" stroke-width="1.2" stroke-linejoin="round"/><rect x="11" y="16" width="2" height="2" fill="#0DF0E3"/><rect x="16" y="16" width="2" height="2" fill="#0DF0E3"/><rect x="14.5" y="12.5" width="1" height="1" fill="#0DF0E3"/></svg>`,
		pasta: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes coreSpin{0%{transform:rotate(0)}10%,50%{transform:rotate(90deg)}60%,100%{transform:rotate(180deg)}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.core{animation:coreSpin 2.5s cubic-bezier(.6,-.28,.735,.045) infinite;transform-origin:14px 14px}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 5 10 L 13 13 L 13 15 L 5 18 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 23 10 L 15 13 L 15 15 L 23 18 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><rect x="12" y="12" width="4" height="4" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2" class="core"/></svg>`,
		rice: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes termBlink{0%,50%{opacity:1}51%,100%{opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.cursor{animation:termBlink .9s infinite}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 9 12 C 7 12, 7 22, 10 22 H 18 C 21 22, 21 12, 19 12 Z" fill="#161B22" stroke="#0DF0E3" stroke-width="1.2"/><path d="M 12 8 L 14 11 L 16 8 Z" fill="none" stroke="#00FF41" stroke-width="1.2"/><path d="M 11 11 H 17" stroke="#00FF41" stroke-width="1.2" stroke-linecap="square"/><text x="12.5" y="19" text-anchor="middle" font-size="6" font-weight="bold" font-family="monospace" fill="#00FF41">[R]</text><rect x="16.5" y="14.5" width="2" height="5" fill="#00FF41" class="cursor"/></svg>`,
		juice: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes chargeLevel{0%{transform:scaleY(0);opacity:.8;fill:#FF0055}50%{fill:#0DF0E3}80%,100%{transform:scaleY(1);opacity:1;fill:#00FF41}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.battery{animation:chargeLevel 3s steps(8) infinite;transform-origin:center bottom}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><rect x="9" y="10" width="10" height="12" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 15 10 V 6 H 17" fill="none" stroke="#FF0055" stroke-width="1.2" stroke-linecap="square"/><rect x="11" y="12" width="6" height="8" fill="#0D1117"/><rect x="11" y="12" width="6" height="8" class="battery"/></svg>`,
		yogurt: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes radarWave{0%{transform:scale(1);opacity:.8;stroke-width:1}100%{transform:scale(1.6);opacity:0;stroke-width:.1}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.wave1{animation:radarWave 2s infinite 0s ease-out;transform-origin:14px 12px}.wave2{animation:radarWave 2s infinite 1s ease-out;transform-origin:14px 12px}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 9 12 L 10 21 C 10 21.5, 11 22, 14 22 C 17 22, 18 21.5, 18 21 L 19 12 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="none" stroke="#0DF0E3" class="wave1"/><ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="none" stroke="#0DF0E3" class="wave2"/><ellipse cx="14" cy="12" rx="5.5" ry="1.5" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/><path d="M 12 16 H 16 M 13 19 H 15" stroke="#0DF0E3" stroke-width="1.2" stroke-dasharray="2 2"/></svg>`,
		cereal: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes floatHex{0%{transform:translateY(0) scale(1);opacity:1}100%{transform:translateY(-8px) scale(.5);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.b1{animation:floatHex 2s infinite linear 0s}.b2{animation:floatHex 2s infinite linear .5s}.b3{animation:floatHex 2s infinite linear 1s}.b4{animation:floatHex 2s infinite linear 1.5s}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><rect x="9" y="8" width="10" height="14" fill="#161B22" stroke="#0DF0E3" stroke-width="1.2"/><rect x="13" y="14" width="2" height="2" fill="#00FF41" class="b1"/><rect x="11" y="15" width="1" height="1" fill="#FF0055" class="b2"/><rect x="16" y="13" width="1" height="1" fill="#FF0055" class="b3"/><rect x="15" y="16" width="1" height="1" fill="#0DF0E3" class="b4"/><rect x="12" y="12" width="1" height="1" fill="#0DF0E3" class="b2"/></svg>`,
		flour: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes zipExtract{to{stroke-dashoffset:-12}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.extract{stroke-dasharray:2 4;animation:zipExtract 1.5s linear infinite}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 9 10 L 19 10 L 19 21 C 19 21.5, 18.5 22, 18 22 L 10 22 C 9.5 22, 9 21.5, 9 21 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 9 10 L 14 13 L 19 10 L 19 7 C 19 6.5, 18.5 6, 18 6 L 10 6 C 9.5 6, 9 6.5, 9 7 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/><path class="extract" d="M 14 14 V 19 M 12 16 H 16 M 12 18 H 16" stroke="#FF0055" stroke-width="1.2" stroke-linecap="round"/></svg>`,
		sugar: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes authCube{0%,100%{transform:scale(1);stroke:#00FF41;fill:#161B22}50%{transform:scale(1.15);stroke:#0DF0E3;fill:#0DF0E3;opacity:.9}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.c1{animation:authCube 1.5s infinite 0s;transform-origin:center;transform-box:fill-box}.c2{animation:authCube 1.5s infinite .5s;transform-origin:center;transform-box:fill-box}.c3{animation:authCube 1.5s infinite 1s;transform-origin:center;transform-box:fill-box}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><rect x="9" y="15" width="6" height="6" stroke-width="1" class="c1"/><rect x="15" y="15" width="6" height="6" stroke-width="1" class="c2"/><rect x="12" y="10" width="6" height="6" stroke-width="1" class="c3"/><circle cx="12" cy="18" r=".5" fill="#00FF41"/><circle cx="18" cy="18" r=".5" fill="#0DF0E3"/><circle cx="15" cy="13" r=".5" fill="#FF0055"/></svg>`,
		coffee: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes codeExhaust{0%{transform:translateY(2px);opacity:0}50%{opacity:1}100%{transform:translateY(-6px);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.s1{animation:codeExhaust 1.5s infinite 0s linear}.s2{animation:codeExhaust 1.5s infinite .75s linear}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 10 12 L 11 20 C 11 21.5, 12 22, 14 22 C 16 22, 17 21.5, 17 20 L 18 12 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 9.5 10 C 9.5 9, 18.5 9, 18.5 10 L 18.5 12 L 9.5 12 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/><path class="s1" d="M 12 8 V 6 H 13 V 4" fill="none" stroke="#FF0055" stroke-width="1" stroke-linejoin="miter"/><path class="s2" d="M 16 8 V 6 H 15 V 4" fill="none" stroke="#0DF0E3" stroke-width="1" stroke-linejoin="miter"/></svg>`,
		tea: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes tagScan{0%{transform:translateX(-1px);opacity:0}50%{opacity:1}100%{transform:translateX(5px);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.scan{animation:tagScan 1.5s infinite linear}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 9 13 V 17 C 9 19, 11 21, 14 21 C 17 21, 19 19, 19 17 V 13 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 19 14 H 21 C 22.5 14, 22.5 17, 21 17 H 19" fill="none" stroke="#0DF0E3" stroke-width="1.2"/><path d="M 14 13 V 9 H 11" fill="none" stroke="#0DF0E3" stroke-width="1" stroke-linecap="square"/><rect x="9" y="7" width="4" height="4" fill="#0D1117" stroke="#00FF41" stroke-width="1"/><line x1="9" y1="7" x2="9" y2="11" stroke="#FF0055" stroke-width="1" class="scan"/></svg>`,
		oil: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes bubbleRise{0%{transform:translateY(6px) scale(.5);opacity:0}20%{opacity:1}100%{transform:translateY(-4px) scale(1.2);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.b1{animation:bubbleRise 2s infinite 0s ease-in}.b2{animation:bubbleRise 2s infinite .6s ease-in}.b3{animation:bubbleRise 2s infinite 1.2s ease-in}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 12 11 C 10 14, 10 21, 10 21 C 10 21.5, 10.5 22, 11 22 H 17 C 17.5 22, 18 21.5, 18 21 C 18 21, 18 14, 16 11 V 8 H 12 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><rect x="12" y="6" width="4" height="2" fill="#0DF0E3"/><rect x="13.5" y="15" width="1" height="1" fill="#0DF0E3" class="b1"/><rect x="12" y="17" width="1" height="1" fill="#0DF0E3" class="b2"/><rect x="15" y="16" width="1" height="1" fill="#0DF0E3" class="b3"/><path d="M 12 11 H 16" stroke="#00FF41" stroke-width="1" stroke-dasharray="2 2"/></svg>`,
		cream: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes dripFall{0%{transform:translateY(0);opacity:1}70%{transform:translateY(6px);opacity:0}100%{transform:translateY(6px);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.d1{animation:dripFall 1.5s infinite 0s linear}.d2{animation:dripFall 1.5s infinite .75s linear}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 11 10 L 8 8 V 10 L 10 12 V 20 C 10 21.5, 10.5 22, 12 22 H 16 C 17.5 22, 18 21.5, 18 20 V 10 Z" fill="#161B22" stroke="#00FF41" stroke-width="1.2" stroke-linejoin="round"/><path d="M 11 10 H 18" stroke="#00FF41" stroke-width="1.2" stroke-linecap="round"/><path d="M 18 13 H 21 V 17 H 18" fill="none" stroke="#0DF0E3" stroke-width="1.2" stroke-linejoin="round"/><line x1="12" y1="18" x2="16" y2="18" stroke="#0DF0E3" stroke-width="1" stroke-dasharray="1 1"/><line x1="12" y1="15" x2="15" y2="15" stroke="#FF0055" stroke-width="1" stroke-dasharray="1 1"/><rect x="7.5" y="11" width="1" height="2" fill="#FF0055" class="d1"/><rect x="7.5" y="12" width="1" height="1" fill="#0DF0E3" class="d2"/></svg>`,
		oats: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes fiberFlow{to{stroke-dashoffset:-8}}@keyframes nodeBlink{0%,100%{fill:#00FF41}50%{fill:#FF0055}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.fiber{animation:fiberFlow 1s linear infinite}.node{animation:nodeBlink 1.5s infinite steps(2,start)}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><path d="M 9 9.5 V 20 C 9 21.5, 19 21.5, 19 20 V 9.5" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path class="fiber" d="M 9 14 C 12 12, 16 16, 19 14" fill="none" stroke="#FF0055" stroke-width="1.2" stroke-dasharray="2 2"/><ellipse cx="14" cy="9.5" rx="5.5" ry="1.5" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><ellipse cx="14" cy="8.5" rx="5.5" ry="1.5" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/><text x="14" y="19.5" text-anchor="middle" font-size="5" font-family="monospace" font-weight="bold" fill="#0DF0E3">[O]</text><circle cx="14" cy="11.5" r=".75" class="node"/></svg>`,
		salt: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 28 28"><style>@keyframes matrixRain{0%{transform:translateY(-3px);opacity:0}20%,80%{opacity:1}100%{transform:translateY(6px);opacity:0}}@keyframes shadowPulse{0%,100%{opacity:.15}50%{opacity:.25}}.shadow{animation:shadowPulse 3s infinite ease-in-out}.rain1{animation:matrixRain 1.5s infinite 0s linear}.rain2{animation:matrixRain 1.5s infinite .4s linear}.rain3{animation:matrixRain 1.5s infinite .8s linear}</style><rect width="28" height="28" fill="#0D1117" rx="4"/><ellipse cx="14" cy="23" rx="6" ry="1.5" fill="#00FF41" class="shadow"/><rect x="11" y="10" width="6" height="12" fill="#161B22" stroke="#00FF41" stroke-width="1.2"/><path d="M 11 10 C 11 7, 17 7, 17 10 Z" fill="#0D1117" stroke="#0DF0E3" stroke-width="1.2"/><rect x="13" y="7.5" width="1" height="1" fill="#FF0055"/><rect x="15" y="8.5" width="1" height="1" fill="#FF0055"/><rect x="12" y="8.5" width="1" height="1" fill="#FF0055"/><line x1="12.5" y1="12" x2="12.5" y2="14" stroke="#0DF0E3" stroke-width="1" class="rain1"/><line x1="14" y1="11" x2="14" y2="13" stroke="#FF0055" stroke-width="1" class="rain2"/><line x1="15.5" y1="13" x2="15.5" y2="15" stroke="#00FF41" stroke-width="1" class="rain3"/></svg>`,
	};

	// Preload SVGs as Image objects for canvas drawing (CRT shader applies to these)
	const ACTIVE_SVGS = isNightmare ? NIGHTMARE_SVGS : CYBER_SVGS;
	let _itemImages = new Map();
	let _itemImagesReady = $state(false);
	if (typeof Image !== 'undefined') {
		let loaded = 0;
		const total = Object.keys(ACTIVE_SVGS).length;
		for (const [type, svg] of Object.entries(ACTIVE_SVGS)) {
			const img = new Image();
			img.onload = () => {
				loaded++;
				if (loaded >= total) {
					_itemImagesReady = true;
					_drawGrid();
				}
			};
			img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
			_itemImages.set(type, img);
		}
	}

	const diffColors = {
		easy: '#39d353',
		medium: '#d29922',
		hard: '#f85149',
		expert: '#da3633',
		nightmare: '#880000',
	};

	// Keyboard controls
	function handleKeydown(e) {
		if (e.key === 'ArrowRight' || e.key === 'l') stepForward();
		else if (e.key === 'ArrowLeft' || e.key === 'h') stepBack();
		else if (e.key === ' ') { e.preventDefault(); togglePlay(); }
		else if (e.key === 'Home') setRound(0);
		else if (e.key === 'End') setRound(rounds.length - 1);
	}

	// ── WebGL CRT Shader (adapted from gingerbeardman/webgl-crt-shader) ──
	let crtCanvas = $state(null);
	let gridWrapper = $state(null);

	const CRT = {
		scanlineIntensity: 0.35,
		scanlineCount: 240,
		brightness: 1.25,
		contrast: 1.1,
		bloomIntensity: 0.35,
		bloomThreshold: 0.25,
		rgbShift: 1.0,
		vignetteStrength: 0.55,
		curvature: 0.1,
		flickerStrength: 0.03,
	};

	const CRT_VERT = `#version 300 es
precision highp float;
const vec2 P[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
const vec2 U[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
out vec2 vUv;
void main(){vUv=U[gl_VertexID];gl_Position=vec4(P[gl_VertexID],0,1);}`;

	const CRT_FRAG = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform float uTime,uScanI,uScanC,uBright,uContrast;
uniform float uBloom,uBloomT,uRGB,uVig,uCurve,uFlick;
in vec2 vUv;
out vec4 fc;
const float PI=3.14159265;
const vec3 LU=vec3(.299,.587,.114);

vec2 curve(vec2 uv,float c){
  vec2 co=uv*2.-1.;
  co*=1.+dot(co,co)*c*.25;
  return co*.5+.5;
}

// pseudo-random from 2D seed
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}

void main(){
  vec2 uv=vUv;

  // Occasional horizontal jitter — brief glitch every ~4s
  float jitGate=step(.992,sin(uTime*1.57));
  uv.x+=jitGate*sin(uv.y*90.+uTime*40.)*.003;

  if(uCurve>.001){
    uv=curve(uv,uCurve);
    if(uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.){fc=vec4(0,0,0,1);return;}
  }
  vec4 px=texture(uTex,uv);

  // Bloom
  if(uBloom>.001){
    float l=dot(px.rgb,LU);
    if(l>uBloomT*.5){
      vec2 o=vec2(.005);
      vec4 bl=px*.4+(texture(uTex,uv+vec2(o.x,0))+texture(uTex,uv-vec2(o.x,0))+
        texture(uTex,uv+vec2(0,o.y))+texture(uTex,uv-vec2(0,o.y)))*.15;
      bl.rgb*=uBright;
      px.rgb+=bl.rgb*uBloom*max(0.,(dot(bl.rgb,LU)-uBloomT)*1.5);
    }
  }

  // RGB chromatic shift
  if(uRGB>.005){
    float s=uRGB*.005;
    px.r+=texture(uTex,vec2(uv.x+s,uv.y)).r*.08;
    px.b+=texture(uTex,vec2(uv.x-s,uv.y)).b*.08;
  }

  // Brightness & contrast
  px.rgb*=uBright;
  px.rgb=(px.rgb-.5)*uContrast+.5;

  float m=1.;

  // Scrolling scanlines
  if(uScanI>.001){
    float scrollY=uv.y+uTime*.015; // slow upward scroll
    m*=1.-abs(sin(scrollY*uScanC*PI))*uScanI;
  }

  // Rolling bright bar (moves down screen every ~8s)
  float barY=fract(uTime*.12);
  float barD=abs(uv.y-barY);
  m+=smoothstep(.025,.0,barD)*.1;

  // Flicker
  if(uFlick>.001) m*=1.+sin(uTime*110.)*uFlick;

  // Vignette
  if(uVig>.001){vec2 vc=uv*2.-1.;float d=max(abs(vc.x),abs(vc.y));m*=1.-d*d*uVig;}

  // Noise grain
  float grain=hash(uv*500.+uTime)*.06-.03;
  px.rgb+=grain;

  px.rgb*=m;
  fc=px;
}`;

	let _crt = null;

	function _crtCompile(gl, type, src) {
		const s = gl.createShader(type);
		gl.shaderSource(s, src);
		gl.compileShader(s);
		if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
			console.error('CRT shader:', gl.getShaderInfoLog(s));
			return null;
		}
		return s;
	}

	$effect(() => {
		if (!crtCanvas) return;
		const gl = crtCanvas.getContext('webgl2', { alpha: false, premultipliedAlpha: false });
		if (!gl) return;

		const vs = _crtCompile(gl, gl.VERTEX_SHADER, CRT_VERT);
		const fs = _crtCompile(gl, gl.FRAGMENT_SHADER, CRT_FRAG);
		if (!vs || !fs) return;

		const prog = gl.createProgram();
		gl.attachShader(prog, vs);
		gl.attachShader(prog, fs);
		gl.linkProgram(prog);
		if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;

		const vao = gl.createVertexArray();
		gl.bindVertexArray(vao);

		const tex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, tex);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

		const unis = {};
		for (const n of ['uTex','uTime','uScanI','uScanC','uBright','uContrast','uBloom','uBloomT','uRGB','uVig','uCurve','uFlick'])
			unis[n] = gl.getUniformLocation(prog, n);

		const scene = document.createElement('canvas');
		const w = run.grid_width * CELLW;
		const h = run.grid_height * CELL;
		scene.width = w;
		scene.height = h;
		crtCanvas.width = w;
		crtCanvas.height = h;
		const ctx = scene.getContext('2d');

		_crt = { gl, prog, vao, tex, unis, scene, ctx };

		let animId;
		function render(ts) {
			if (!_crt) return;
			const t = ts * 0.001;
			// Redraw grid every frame for animated glows
			_drawGrid(t);
			gl.bindTexture(gl.TEXTURE_2D, tex);
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, scene);
			// Subtle organic drift — each on a different slow cycle
			const bloom = CRT.bloomIntensity + Math.sin(t * 0.37) * 0.06 + Math.sin(t * 1.1) * 0.02;
			const bright = CRT.brightness + Math.sin(t * 0.23) * 0.025;
			const flick = CRT.flickerStrength + Math.max(0, Math.sin(t * 0.13) * 0.008);
			const scanI = CRT.scanlineIntensity + Math.sin(t * 0.19) * 0.03;
			const rgb = CRT.rgbShift + Math.sin(t * 0.29) * 0.1;

			gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
			gl.useProgram(prog);
			gl.bindVertexArray(vao);
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gl.TEXTURE_2D, tex);
			gl.uniform1i(unis.uTex, 0);
			gl.uniform1f(unis.uTime, t);
			gl.uniform1f(unis.uScanI, scanI);
			gl.uniform1f(unis.uScanC, CRT.scanlineCount);
			gl.uniform1f(unis.uBright, bright);
			gl.uniform1f(unis.uContrast, CRT.contrast);
			gl.uniform1f(unis.uBloom, bloom);
			gl.uniform1f(unis.uBloomT, CRT.bloomThreshold);
			gl.uniform1f(unis.uRGB, rgb);
			gl.uniform1f(unis.uVig, CRT.vignetteStrength);
			gl.uniform1f(unis.uCurve, CRT.curvature);
			gl.uniform1f(unis.uFlick, flick);
			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
			animId = requestAnimationFrame(render);
		}
		animId = requestAnimationFrame(render);

		return () => {
			cancelAnimationFrame(animId);
			gl.deleteTexture(tex);
			gl.deleteProgram(prog);
			_crt = null;
		};
	});

	// Build dropoff set for multi-dropoff support
	const _dropOffZones = run.drop_off_zones || [run.drop_off];
	const _dropOffSet = new Set(_dropOffZones.map(d => `${d[0]},${d[1]}`));

	function _drawGrid(time = 0) {
		if (!_crt) return;
		const { ctx: c, scene } = _crt;
		const C = CELL, W = CELLW, IO = ICON_OFF, gw = run.grid_width, gh = run.grid_height;
		const nm = isNightmare;
		c.clearRect(0, 0, scene.width, scene.height);

		// Floor
		c.fillStyle = nm ? '#1a1818' : '#1a2230';
		c.fillRect(0, 0, gw * W, gh * C);

		// Cells
		for (let y = 0; y < gh; y++) {
			for (let x = 0; x < gw; x++) {
				const key = `${x},${y}`;
				if (wallSet.has(key)) {
					const edge = (x === 0 || y === 0 || x === gw - 1 || y === gh - 1);
					c.fillStyle = nm ? (edge ? '#2a1010' : '#351515') : (edge ? '#2d333b' : '#373e47');
					c.fillRect(x * W, y * C, W, C);
				} else if (shelfSet.has(key)) {
					c.fillStyle = nm ? '#1a0a0a' : '#0d2818';
					c.fillRect(x * W, y * C, W, C);
					c.strokeStyle = nm ? '#442222' : '#1a4d2e';
					c.lineWidth = 1;
					c.strokeRect(x * W + 0.5, y * C + 0.5, W - 1, C - 1);
				} else {
					// Floor grid lines
					c.strokeStyle = nm ? '#2a2626' : '#161b22';
					c.lineWidth = 0.5;
					c.strokeRect(x * W + 0.5, y * C + 0.5, W - 1, C - 1);
				}
			}
		}

		// Drop-off zones (all of them)
		c.font = `bold ${Math.floor(C * 0.45)}px sans-serif`;
		c.textAlign = 'center';
		c.textBaseline = 'middle';
		for (const [dx, dy] of _dropOffZones) {
			if (nm) {
				// Pentagram dropoff
				c.fillStyle = 'rgba(136,0,0,0.15)';
				c.fillRect(dx * W, dy * C, W, C);
				c.strokeStyle = '#880000';
				c.lineWidth = 2;
				c.strokeRect(dx * W + 1, dy * C + 1, W - 2, C - 2);
				// Draw pentagram (centered icon)
				const pcx = dx * W + W / 2, pcy = dy * C + C / 2, pr = C * 0.35;
				c.beginPath();
				for (let i = 0; i < 5; i++) {
					const a = -Math.PI / 2 + (i * 4 * Math.PI) / 5 + time * 0.3;
					const method = i === 0 ? 'moveTo' : 'lineTo';
					c[method](pcx + pr * Math.cos(a), pcy + pr * Math.sin(a));
				}
				c.closePath();
				c.strokeStyle = '#AA0000';
				c.lineWidth = 1.2;
				c.stroke();
			} else {
				c.fillStyle = 'rgba(57,211,83,0.2)';
				c.fillRect(dx * W, dy * C, W, C);
				c.strokeStyle = '#39d353';
				c.lineWidth = 2;
				c.strokeRect(dx * W + 1, dy * C + 1, W - 2, C - 2);
				c.fillStyle = '#39d353';
				c.fillText('D', dx * W + W / 2, dy * C + C / 2);
			}
		}

		// Spawn
		const [sx, sy] = run.spawn;
		if (nm) {
			c.fillStyle = 'rgba(170,0,51,0.15)';
			c.fillRect(sx * W, sy * C, W, C);
			c.strokeStyle = '#AA0033';
			c.lineWidth = 2;
			c.strokeRect(sx * W + 1, sy * C + 1, W - 2, C - 2);
			c.fillStyle = '#FF0000';
			c.fillText('S', sx * W + W / 2, sy * C + C / 2);
		} else {
			c.fillStyle = 'rgba(88,166,255,0.2)';
			c.fillRect(sx * W, sy * C, W, C);
			c.strokeStyle = '#58a6ff';
			c.lineWidth = 2;
			c.strokeRect(sx * W + 1, sy * C + 1, W - 2, C - 2);
			c.fillStyle = '#58a6ff';
			c.fillText('S', sx * W + W / 2, sy * C + C / 2);
		}

		// Items on shelves + animated active/preview highlights
		for (const [key, items] of itemMap) {
			const [ix, iy] = key.split(',').map(Number);
			const itype = items[0].type;
			const isActive = activeTypes.has(itype);
			const isPreview = !isActive && previewTypes.has(itype);
			// Animated active/preview cell glow (desynced: active peaks when preview fades)
			if (isActive) {
				const wave = time * 2.5;
				const pulse = Math.cos(wave) * 0.5 + 0.5;
				const fillA = 0.08 + pulse * 0.18;
				const strokeA = 0.3 + pulse * 0.5;
				const spread = pulse * 2;
				const ar = nm ? 255 : 250, ag = nm ? 20 : 204, ab = nm ? 147 : 21;
				c.fillStyle = `rgba(${ar}, ${ag}, ${ab}, ${fillA})`;
				c.fillRect(ix * W + IO - spread, iy * C - spread, C + spread * 2, C + spread * 2);
				c.strokeStyle = `rgba(${ar}, ${ag}, ${ab}, ${strokeA})`;
				c.lineWidth = 1.5 + pulse;
				c.strokeRect(ix * W + IO - spread + 0.5, iy * C - spread + 0.5, C + spread * 2 - 1, C + spread * 2 - 1);
			} else if (isPreview) {
				const wave = time * 2.5;
				const pulse = Math.sin(wave) * 0.5 + 0.5;
				const fillA = 0.05 + pulse * 0.12;
				const strokeA = 0.2 + pulse * 0.4;
				const pr = nm ? 255 : 244, pg = nm ? 255 : 114, pb = nm ? 255 : 182;
				c.fillStyle = `rgba(${pr}, ${pg}, ${pb}, ${fillA})`;
				c.fillRect(ix * W + IO, iy * C, C, C);
				c.strokeStyle = `rgba(${pr}, ${pg}, ${pb}, ${strokeA})`;
				c.lineWidth = 1 + pulse * 0.5;
				c.strokeRect(ix * W + IO + 0.5, iy * C + 0.5, C - 1, C - 1);
			}
			// Draw item icon (square, centered in wider cell)
			const img = _itemImages.get(itype);
			if (img && img.complete && img.naturalWidth > 0) {
				c.drawImage(img, ix * W + IO, iy * C, C, C);
			} else {
				const ccx = ix * W + W / 2, ccy = iy * C + C / 2;
				c.fillStyle = nm ? '#880000' : '#00FF41';
				c.font = `bold ${Math.floor(C * 0.26)}px monospace`;
				c.textAlign = 'center'; c.textBaseline = 'middle';
				c.fillText(ITEM_ABBR[itype] || itype.slice(0, 2).toUpperCase(), ccx, ccy + 1);
			}
		}

		// Bots
		for (const bot of bots) {
			const [bx, by] = bot.position;
			const clr = BOT_COLORS[bot.id % BOT_COLORS.length];
			const cx = bx * W + IO, cy = by * C;

			if (nm) {
				// Nightmare goat bot (simplified canvas version)
				// Shadow
				c.fillStyle = 'rgba(51,0,0,0.4)';
				c.beginPath();
				c.ellipse(cx + C / 2, cy + C - 2, C / 3, 2, 0, 0, Math.PI * 2);
				c.fill();
				// Head (trapezoid)
				c.fillStyle = '#0A0303';
				c.strokeStyle = clr;
				c.lineWidth = 1.2;
				c.beginPath();
				c.moveTo(cx + C * 0.32, cy + C * 0.39);
				c.lineTo(cx + C * 0.68, cy + C * 0.39);
				c.lineTo(cx + C * 0.57, cy + C * 0.71);
				c.lineTo(cx + C * 0.43, cy + C * 0.71);
				c.closePath();
				c.fill(); c.stroke();
				// Horns
				c.beginPath();
				c.moveTo(cx + C * 0.39, cy + C * 0.32);
				c.quadraticCurveTo(cx + C * 0.18, cy + C * 0.07, cx + C * 0.07, cy + C * 0.29);
				c.strokeStyle = clr; c.lineWidth = 1.5; c.stroke();
				c.beginPath();
				c.moveTo(cx + C * 0.61, cy + C * 0.32);
				c.quadraticCurveTo(cx + C * 0.82, cy + C * 0.07, cx + C * 0.93, cy + C * 0.29);
				c.stroke();
				// Eyes (red triangles)
				const eyePulse = Math.sin(time * 3 + bot.id) * 0.5 + 0.5;
				c.fillStyle = `rgba(255, ${Math.floor(eyePulse * 80)}, 0, ${0.7 + eyePulse * 0.3})`;
				c.beginPath();
				c.moveTo(cx + C * 0.36, cy + C * 0.46);
				c.lineTo(cx + C * 0.43, cy + C * 0.5);
				c.lineTo(cx + C * 0.36, cy + C * 0.54);
				c.fill();
				c.beginPath();
				c.moveTo(cx + C * 0.64, cy + C * 0.46);
				c.lineTo(cx + C * 0.57, cy + C * 0.5);
				c.lineTo(cx + C * 0.64, cy + C * 0.54);
				c.fill();
				// ID
				c.fillStyle = clr;
				c.font = `bold ${Math.floor(C * 0.18)}px monospace`;
				c.textAlign = 'center'; c.textBaseline = 'middle';
				c.fillText(String(bot.id), cx + C / 2, cy + C * 0.88);
			} else {
				// Cyberpunk drone bot
				c.fillStyle = 'rgba(0,0,0,0.35)';
				c.beginPath();
				c.ellipse(cx + C / 2, cy + C - 2, C / 3, 2, 0, 0, Math.PI * 2);
				c.fill();
				c.fillStyle = '#161B22';
				c.shadowColor = clr;
				c.shadowBlur = 4;
				c.beginPath();
				c.moveTo(cx + C * 0.32, cy + C * 0.32);
				c.lineTo(cx + C * 0.68, cy + C * 0.32);
				c.lineTo(cx + C * 0.75, cy + C * 0.54);
				c.lineTo(cx + C * 0.25, cy + C * 0.54);
				c.closePath();
				c.fill();
				c.beginPath();
				c.moveTo(cx + C * 0.25, cy + C * 0.54);
				c.lineTo(cx + C * 0.39, cy + C * 0.68);
				c.lineTo(cx + C * 0.61, cy + C * 0.68);
				c.lineTo(cx + C * 0.75, cy + C * 0.54);
				c.fill();
				c.shadowBlur = 0;
				c.fillStyle = '#0D1117';
				c.fillRect(cx + C * 0.36, cy + C * 0.41, C * 0.28, C * 0.11);
				c.strokeStyle = clr;
				c.lineWidth = 0.8;
				c.strokeRect(cx + C * 0.36, cy + C * 0.41, C * 0.28, C * 0.11);
				const scanX = Math.sin(time * 2 + bot.id) * C * 0.08;
				c.fillStyle = '#FF0055';
				c.fillRect(cx + C * 0.46 + scanX, cy + C * 0.43, C * 0.08, C * 0.07);
				c.fillStyle = clr;
				c.font = `bold ${Math.floor(C * 0.22)}px monospace`;
				c.textAlign = 'center'; c.textBaseline = 'middle';
				c.fillText(String(bot.id), cx + C / 2, cy + C * 0.62);
			}
			// Inventory dots below
			bot.inventory.forEach((item, i) => {
				c.fillStyle = ITEM_CLR[item] || '#aaa';
				c.beginPath();
				c.arc(
					bx * W + W / 2 - (bot.inventory.length - 1) * 3 + i * 6,
					by * C + C + 3, 2.5, 0, Math.PI * 2
				);
				c.fill();
			});
		}

	}
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="replay-page stagger">
	<div class="top-bar">
		<a href="/" class="back-link">&lt; runs</a>
		<div class="run-info">
			<span class="badge" style="background: {diffColors[run.difficulty]}22; color: {diffColors[run.difficulty]}; border: 1px solid {diffColors[run.difficulty]}44">
				{run.difficulty}
			</span>
			<span class="mono">seed:{run.seed}</span>
			<span class="mono">{run.grid_width}x{run.grid_height}</span>
			<span class="mono">{run.bot_count}bots</span>
		</div>
		<div class="score-display">
			<span class="score-label">score&gt;</span>
			<span class="score-value">{rounds[currentRound]?.score ?? '?'}</span>
			<span class="score-final">/{run.final_score}</span>
		</div>
	</div>

	<div class="main-area">
		<div class="grid-section">
			<div class="view-toggle">
				<button class="view-btn" class:active={viewMode === 'crt'} onclick={() => viewMode = 'crt'}>CRT</button>
				<button class="view-btn" class:active={viewMode === '3d'} onclick={() => viewMode = '3d'}>3D</button>
			</div>

			{#if viewMode === 'crt'}
			<div class="crt-monitor">
				<div class="crt-bezel">
					<div class="crt-screen" bind:this={gridWrapper}>
						<Grid
							width={run.grid_width}
							height={run.grid_height}
							cellSize={CELL}
							{wallSet}
							{shelfSet}
							{itemMap}
							dropOff={run.drop_off}
							dropOffZones={run.drop_off_zones}
							spawn={run.spawn}
							{bots}
							{botPositions}
							botColors={BOT_COLORS}
							{selectedBot}
							onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
							{activeTypes}
							{previewTypes}
							difficulty={run.difficulty}
						/>
					<canvas bind:this={crtCanvas} class="crt-overlay"></canvas>
					</div>
				</div>
			</div>
			{:else}
			<div class="threed-wrapper">
				<Grid3D
					width={run.grid_width}
					height={run.grid_height}
					{wallSet}
					{shelfSet}
					{itemMap}
					dropOff={run.drop_off}
					dropOffZones={run.drop_off_zones}
					spawn={run.spawn}
					{bots}
					botColors={BOT_COLORS}
					{selectedBot}
					onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
					{activeTypes}
					{previewTypes}
					difficulty={run.difficulty}
				/>
			</div>
			{/if}

			<!-- Controls -->
			<div class="controls card">
				<div class="controls-row">
					<button class="ctrl-btn" onclick={() => setRound(0)} title="Start (Home)">|&lt;</button>
					<button class="ctrl-btn" onclick={stepBack} title="Step back (Left/H)">&lt;&lt;</button>
					<button class="ctrl-btn play-btn" class:playing onclick={togglePlay} title="Play/Pause (Space)">
						{#if playing}||{:else}&#9654;{/if}
					</button>
					<button class="ctrl-btn" onclick={stepForward} title="Step forward (Right/L)">&gt;&gt;</button>
					<button class="ctrl-btn" onclick={() => setRound(rounds.length - 1)} title="End (End)">&gt;|</button>
				</div>

				<div class="round-slider">
					<span class="round-label">R:{String(currentRound).padStart(3, '0')}</span>
					<input type="range" min="0" max={rounds.length - 1} bind:value={currentRound} oninput={pause} />
					<span class="round-label">{rounds.length - 1}</span>
				</div>

				<div class="speed-control">
					<span class="speed-label">&gt; spd:</span>
					{#each [1, 2, 5, 10, 30] as s}
						<button class="speed-btn" class:active={speed === s} onclick={() => setSpeed(s)}>{s}x</button>
					{/each}
				</div>
			</div>
		</div>

		<!-- Side panel -->
		<div class="side-panel">
			<!-- Orders -->
			<div class="panel-section card">
				<h3>Orders</h3>
				{#each orders as order}
					{@const deliveredMask = getDeliveredMask(order)}
					<div class="order" class:active-order={order.status === 'active'} class:preview-order={order.status === 'preview'}>
						<div class="order-header">
							<span class="order-status" class:active={order.status === 'active'}>{order.status}</span>
							<span class="mono order-id">{order.id}</span>
						</div>
						<div class="order-items">
							{#each order.items_required as item, i}
								<span class="order-item" class:delivered={deliveredMask[i]}>
									<span class="cyber-icon-wrap">{@html getCyberIcon(item, 22)}</span>
									<span class="item-name">{getItemTypeName(item)}</span>
								</span>
							{/each}
						</div>
						<div class="order-progress">
							{order.items_delivered.length} / {order.items_required.length} delivered
						</div>
					</div>
				{/each}
			</div>

			<!-- Event History -->
			<div class="panel-section card">
				<h3>Event Log ({eventHistory.length})</h3>
				<div class="event-log">
					{#each eventHistory.toReversed() as evt}
						<div class="event" class:current-round={evt.round === currentRound}>
							<span class="evt-round mono">R{evt.round}</span>
							{#if evt.type === 'pickup'}
								<span class="evt-icon pickup-icon">P</span>
								<span class="evt-text">Bot {evt.bot} picked {evt.item_type}</span>
							{:else if evt.type === 'deliver'}
								<span class="evt-icon deliver-icon">D</span>
								<span class="evt-text">Bot {evt.bot} delivered {evt.item_type}</span>
							{:else if evt.type === 'auto_deliver'}
								<span class="evt-icon auto-icon">A</span>
								<span class="evt-text">Bot {evt.bot} auto-delivered {evt.item_type}</span>
							{:else if evt.type === 'order_complete'}
								<span class="evt-icon complete-icon">!</span>
								<span class="evt-text">Order complete! (+5)</span>
							{/if}
						</div>
					{/each}
					{#if eventHistory.length === 0}
						<div class="no-events">No events yet</div>
					{/if}
				</div>
			</div>

			<!-- Map Legend -->
			<div class="panel-section card">
				<h3>Legend</h3>
				<div class="legend-grid">
					<div class="legend-item"><span class="legend-swatch" style="background: #2d333b"></span>Wall</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #0d2818"></span>Shelf</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #39d353"></span>Drop-off</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #58a6ff"></span>Spawn</div>
					<div class="legend-item"><span class="legend-swatch" style="background: #010409"></span>Floor</div>
				</div>
			</div>

			<!-- Groceries on this map -->
			<div class="panel-section card">
				<h3>Groceries on Map ({groceryTypes.length})</h3>
				<div class="grocery-list">
					{#each groceryTypes as g}
						<div class="grocery-entry">
							<span class="cyber-icon-wrap">{@html getCyberIcon(g.type, 22)}</span>
							<span class="grocery-name">{getItemTypeName(g.type)}</span>
							<span class="grocery-count">x{g.count}</span>
						</div>
					{/each}
				</div>
			</div>

			<!-- Bots -->
			<div class="panel-section card">
				<h3>Bots</h3>
				{#each bots as bot}
					<button
						class="bot-row"
						class:selected={selectedBot === bot.id}
						onclick={() => selectedBot = selectedBot === bot.id ? null : bot.id}
						style="--bot-color: {BOT_COLORS[bot.id % BOT_COLORS.length]}"
					>
						<div class="bot-header">
							<span class="cyber-icon-wrap">{@html getBotSvg(bot.id, BOT_COLORS[bot.id % BOT_COLORS.length], 24)}</span>
							<span class="bot-pos mono">({bot.position[0]}, {bot.position[1]})</span>
							<span class="bot-action">{getBotAction(bot.id)}</span>
						</div>
						<div class="bot-inv">
							{#if bot.inventory.length === 0}
								<span class="empty-inv">empty</span>
							{:else}
								{#each bot.inventory as item}
									<span class="inv-item">
										<span class="cyber-icon-wrap sm">{@html getCyberIcon(item, 18)}</span>
										<span class="item-name">{getItemTypeName(item)}</span>
									</span>
								{/each}
							{/if}
						</div>
					</button>
				{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.replay-page { display: flex; flex-direction: column; gap: 1rem; }

	/* ── Top bar — terminal header ── */
	.top-bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		flex-wrap: wrap;
		gap: 1rem;
		padding: 0.6rem 1rem;
		background: #010409;
		border: 1px solid var(--border);
		border-left: 3px solid var(--accent);
		border-radius: 0;
	}
	.back-link { font-size: 0.8rem; color: var(--accent); }
	.back-link:hover { color: var(--accent-light); }
	.run-info { display: flex; align-items: center; gap: 0.75rem; font-size: 0.8rem; }
	.badge {
		padding: 0.15rem 0.5rem;
		border-radius: 0;
		font-size: 0.72rem;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}
	.mono { font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted); }
	.score-display { display: flex; align-items: baseline; gap: 0.35rem; }
	.score-label { font-size: 0.72rem; color: var(--accent); text-transform: none; font-weight: 700; }
	.score-value {
		font-size: 1.75rem;
		font-weight: 800;
		color: var(--green);
		font-family: var(--font-mono);
		text-shadow: 0 0 12px rgba(57, 211, 83, 0.3);
	}
	.score-final { font-size: 0.8rem; color: var(--text-muted); }

	.main-area {
		display: grid;
		grid-template-columns: 1fr 420px;
		gap: 1rem;
		align-items: start;
	}

	/* ── Grid + CRT Monitor ── */
	.grid-section { display: flex; flex-direction: column; gap: 1rem; }

	.view-toggle {
		display: flex;
		gap: 0;
		align-self: flex-start;
	}
	.view-btn {
		padding: 0.25rem 0.75rem;
		background: transparent;
		color: var(--text-muted);
		border: 1px solid var(--border);
		font-size: 0.7rem;
		font-family: var(--font-mono);
		font-weight: 700;
		letter-spacing: 0.08em;
		cursor: pointer;
		transition: all 0.15s ease;
	}
	.view-btn:first-child { border-right: none; }
	.view-btn.active {
		background: rgba(57, 211, 83, 0.1);
		color: var(--accent);
		border-color: var(--accent);
		text-shadow: 0 0 6px rgba(57, 211, 83, 0.4);
	}
	.view-btn:hover:not(.active) {
		color: var(--accent);
		border-color: var(--accent);
	}

	.threed-wrapper {
		border: 3px solid #111;
		border-radius: 6px;
		background: #050505;
		padding: 4px;
		box-shadow: 0 0 20px rgba(0,0,0,0.8);
	}

	.crt-monitor {
		position: relative;
		background: #050505;
		border: 3px solid #111;
		border-radius: 6px;
		padding: 8px;
		box-shadow: 0 0 20px rgba(0,0,0,0.8), inset 0 0 30px rgba(0,0,0,0.5);
	}
	.crt-bezel {
		position: relative;
		border: 2px solid #1a1a1a;
		border-radius: 4px;
		background: #010409;
		overflow: hidden;
		box-shadow: inset 0 0 40px rgba(57, 211, 83, 0.03);
	}
	.crt-screen {
		position: relative;
		padding: 0.5rem;
	}
	.crt-overlay {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		pointer-events: none;
		z-index: 2;
	}

	/* ── Controls — hacker terminal ── */
	.controls {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		align-items: center;
		background: #010409;
		border: 1px solid var(--border);
		border-radius: 0;
		padding: 0.75rem 1rem;
	}
	.controls-row { display: flex; gap: 0.4rem; align-items: center; }
	.ctrl-btn {
		width: 38px;
		height: 32px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: transparent;
		color: var(--accent);
		border: 1px solid var(--accent);
		border-radius: 0;
		font-family: var(--font-mono);
		font-size: 0.72rem;
		font-weight: 700;
		letter-spacing: -0.05em;
		cursor: pointer;
		transition: all 0.1s ease;
		text-shadow: 0 0 4px rgba(57, 211, 83, 0.3);
	}
	.ctrl-btn:hover {
		background: rgba(57, 211, 83, 0.12);
		box-shadow: 0 0 8px rgba(57, 211, 83, 0.2), inset 0 0 8px rgba(57, 211, 83, 0.06);
		text-shadow: 0 0 8px rgba(57, 211, 83, 0.6);
	}
	.ctrl-btn:active {
		background: rgba(57, 211, 83, 0.25);
		box-shadow: 0 0 12px rgba(57, 211, 83, 0.4);
	}
	.play-btn {
		width: 52px;
		height: 36px;
		font-size: 0.9rem;
		border-width: 2px;
		box-shadow: 0 0 6px rgba(57, 211, 83, 0.15);
	}
	.play-btn.playing {
		border-color: var(--red);
		color: var(--red);
		text-shadow: 0 0 4px rgba(248, 81, 73, 0.4);
		box-shadow: 0 0 6px rgba(248, 81, 73, 0.15);
	}
	.play-btn.playing:hover {
		background: rgba(248, 81, 73, 0.12);
		box-shadow: 0 0 10px rgba(248, 81, 73, 0.3);
		text-shadow: 0 0 8px rgba(248, 81, 73, 0.6);
	}

	.round-slider {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		width: 100%;
	}
	.round-label {
		font-size: 0.72rem;
		color: var(--accent);
		font-family: var(--font-mono);
		white-space: nowrap;
		min-width: 52px;
		text-shadow: 0 0 4px rgba(57, 211, 83, 0.2);
	}
	input[type="range"] {
		flex: 1;
		height: 2px;
		-webkit-appearance: none;
		background: var(--accent);
		opacity: 0.3;
		border-radius: 0;
		outline: none;
	}
	input[type="range"]::-webkit-slider-thumb {
		-webkit-appearance: none;
		width: 10px;
		height: 16px;
		border-radius: 0;
		background: var(--accent);
		cursor: pointer;
		box-shadow: 0 0 8px rgba(57, 211, 83, 0.5);
	}

	.speed-control { display: flex; align-items: center; gap: 0.3rem; }
	.speed-label {
		font-size: 0.68rem;
		color: var(--accent);
		font-family: var(--font-mono);
		margin-right: 0.15rem;
		opacity: 0.7;
	}
	.speed-btn {
		padding: 0.15rem 0.45rem;
		background: transparent;
		color: var(--text-muted);
		border: 1px solid var(--border);
		border-radius: 0;
		font-size: 0.68rem;
		font-family: var(--font-mono);
		cursor: pointer;
		transition: all 0.1s ease;
	}
	.speed-btn:hover {
		color: var(--accent);
		border-color: var(--accent);
		text-shadow: 0 0 4px rgba(57, 211, 83, 0.3);
	}
	.speed-btn.active {
		background: rgba(57, 211, 83, 0.1);
		color: var(--accent);
		border-color: var(--accent);
		box-shadow: 0 0 6px rgba(57, 211, 83, 0.2);
		text-shadow: 0 0 6px rgba(57, 211, 83, 0.4);
	}

	/* ── Side panel ── */
	.side-panel {
		display: flex;
		flex-direction: column;
		gap: 0;
		max-height: calc(100vh - 150px);
		overflow-y: auto;
	}

	.panel-section {
		background: #010409;
		border: 1px solid var(--border);
		border-radius: 0;
		padding: 0.75rem;
		border-bottom: none;
	}
	.panel-section:last-child {
		border-bottom: 1px solid var(--border);
	}

	.panel-section h3 {
		font-size: 0.7rem;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--accent);
		margin-bottom: 0.5rem;
		padding-bottom: 0.3rem;
		border-bottom: 1px solid rgba(57, 211, 83, 0.15);
	}

	/* ── Orders — no border, separator lines ── */
	.order {
		padding: 0.5rem 0;
		border-radius: 0;
		margin-bottom: 0;
		border: none;
		border-bottom: 1px solid rgba(48, 54, 61, 0.5);
	}
	.order:last-child { border-bottom: none; }
	.active-order { background: none; border-left: 2px solid var(--yellow); padding-left: 0.5rem; }
	.preview-order { background: none; border-left: 2px solid var(--pink); padding-left: 0.5rem; opacity: 0.7; }
	.order-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; }
	.order-status {
		font-size: 0.6rem;
		text-transform: uppercase;
		font-weight: 700;
		padding: 0.05rem 0.3rem;
		border-radius: 0;
		background: none;
		color: var(--pink);
		letter-spacing: 0.08em;
	}
	.order-status.active { background: none; color: var(--yellow); }
	.order-id { font-size: 0.68rem; color: var(--text-muted); }
	.order-items { display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.3rem; }
	.order-item {
		display: inline-flex;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.62rem;
		padding: 0.1rem 0.25rem 0.1rem 0.1rem;
		background: rgba(22, 27, 34, 0.6);
		border: 1px solid var(--border);
		border-radius: 0;
		color: var(--text-muted);
	}
	.order-item.delivered {
		border-color: rgba(57, 211, 83, 0.3);
		background: rgba(57, 211, 83, 0.06);
	}
	.order-item.delivered .item-name { color: var(--green); text-decoration: line-through; }
	.order-item.delivered .item-icon { opacity: 0.5; }
	.item-icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 20px;
		height: 20px;
		border-radius: 50%;
		font-size: 0.55rem;
		font-weight: 800;
		color: #0d1117;
		flex-shrink: 0;
		letter-spacing: -0.03em;
	}
	.item-icon.sm {
		width: 16px;
		height: 16px;
		font-size: 0.48rem;
	}
	.cyber-icon-wrap {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		border-radius: 2px;
		overflow: hidden;
	}
	.cyber-icon-wrap.sm {
		transform: scale(0.9);
	}
	:global(.cyber-icon-wrap svg) {
		display: block;
	}
	.item-name { font-size: 0.65rem; }
	.order-progress { font-size: 0.65rem; color: var(--text-muted); }

	/* ── Bots ── */
	.bot-row {
		display: block;
		width: 100%;
		text-align: left;
		padding: 0.4rem;
		background: transparent;
		border: none;
		border-bottom: 1px solid rgba(48, 54, 61, 0.4);
		border-radius: 0;
		margin-bottom: 0;
		color: var(--text);
	}
	.bot-row:last-child { border-bottom: none; }
	.bot-row:hover { background: rgba(57, 211, 83, 0.04); }
	.bot-row.selected { border-left: 2px solid var(--bot-color); background: rgba(57, 211, 83, 0.04); }
	.bot-header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem; }
	.bot-marker {
		width: 20px;
		height: 20px;
		border-radius: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 0.65rem;
		font-weight: 700;
		color: #0d1117;
	}
	.bot-pos { font-size: 0.72rem; color: var(--text-muted); }
	.bot-action {
		margin-left: auto;
		font-size: 0.68rem;
		color: var(--accent);
		font-family: var(--font-mono);
	}
	.bot-inv { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-left: 1.5rem; }
	.empty-inv { font-size: 0.68rem; color: var(--text-muted); }
	.inv-item {
		display: inline-flex;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.62rem;
		padding: 0.1rem 0.25rem 0.1rem 0.1rem;
		background: rgba(22, 27, 34, 0.5);
		border: 1px solid var(--border);
		border-radius: 0;
		color: var(--accent);
	}

	/* ── Event log ── */
	.event-log {
		max-height: 300px;
		overflow-y: auto;
	}
	.event {
		font-size: 0.72rem;
		padding: 0.2rem 0.3rem;
		display: flex;
		align-items: center;
		gap: 0.4rem;
		border-radius: 0;
		border-bottom: 1px solid rgba(48, 54, 61, 0.3);
	}
	.event:last-child { border-bottom: none; }
	.event.current-round {
		background: rgba(57, 211, 83, 0.06);
		border-left: 2px solid var(--accent);
	}
	.evt-round {
		font-size: 0.62rem;
		color: var(--text-muted);
		min-width: 28px;
	}
	.evt-text { flex: 1; }
	.evt-icon {
		width: 16px;
		height: 16px;
		border-radius: 0;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 0.55rem;
		font-weight: 700;
		color: #0d1117;
		flex-shrink: 0;
	}
	.pickup-icon { background: var(--blue); }
	.deliver-icon { background: var(--green); }
	.auto-icon { background: var(--orange); }
	.complete-icon { background: var(--red); }
	.no-events { font-size: 0.72rem; color: var(--text-muted); padding: 0.5rem 0; }

	/* ── Legend ── */
	.legend-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.3rem; }
	.legend-item { display: flex; align-items: center; gap: 0.4rem; font-size: 0.7rem; color: var(--text-muted); }
	.legend-swatch { width: 14px; height: 14px; border-radius: 0; border: 1px solid var(--border); }

	/* ── Groceries ── */
	.grocery-list {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0 0;
		border-collapse: collapse;
	}
	.grocery-list > :nth-child(odd) {
		border-right: 2px solid rgba(57, 211, 83, 0.15);
		padding-right: 0.35rem;
	}
	.grocery-list > :nth-child(even) {
		padding-left: 0.35rem;
	}
	.grocery-entry {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		padding: 0.25rem 0.2rem;
		border: none;
	}
	.grocery-name { font-size: 0.65rem; color: var(--text-muted); flex: 1; }
	.grocery-count { font-size: 0.6rem; color: var(--text-muted); opacity: 0.6; font-family: var(--font-mono); }

	@media (max-width: 900px) {
		.main-area { grid-template-columns: 1fr; }
		.side-panel { max-height: none; }
	}
</style>
