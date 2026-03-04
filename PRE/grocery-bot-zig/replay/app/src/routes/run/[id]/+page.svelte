<script>
	import Grid from '$lib/components/Grid.svelte';

	let { data } = $props();
	let run = $derived(data.run);
	let rounds = $derived(data.rounds);

	const CELL = 28;
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

	// Shelf items for overlay (same as Grid.svelte shelfItems)
	let shelfItems = $derived.by(() => {
		const result = [];
		for (const [key, items] of itemMap) {
			const [x, y] = key.split(',').map(Number);
			result.push({ x, y, items, type: items[0].type });
		}
		return result;
	});

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
	};

	const ITEM_ABBR = {
		milk: 'Mk', bread: 'Br', eggs: 'Eg', butter: 'Bu',
		cheese: 'Ch', pasta: 'Pa', rice: 'Ri', juice: 'Ju',
		yogurt: 'Yo', cereal: 'Ce', flour: 'Fl', sugar: 'Su',
		coffee: 'Co', tea: 'Te', oil: 'Oi', salt: 'Sa',
	};

	function getItemAbbr(t) { return ITEM_ABBR[t] || t.slice(0, 2).toUpperCase(); }
	function getItemColor(t) { return ITEM_CLR[t] || '#aaa'; }
	function getCyberIcon(t, size = 20) { return CYBER_SVGS[t] ? CYBER_SVGS[t].replace(/width="28"/, `width="${size}"`).replace(/height="28"/, `height="${size}"`) : ''; }

	function getBotSvg(botId, color, size = 24) {
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

	// Preload cyberpunk SVGs as Image objects for canvas drawing (CRT shader applies to these)
	let _itemImages = new Map();
	let _itemImagesReady = $state(false);
	if (typeof Image !== 'undefined') {
		let loaded = 0;
		const total = Object.keys(CYBER_SVGS).length;
		for (const [type, svg] of Object.entries(CYBER_SVGS)) {
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
	let _crtDirty = true;

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
		const w = run.grid_width * CELL;
		const h = run.grid_height * CELL;
		scene.width = w;
		scene.height = h;
		crtCanvas.width = w;
		crtCanvas.height = h;
		const ctx = scene.getContext('2d');

		_crt = { gl, prog, vao, tex, unis, scene, ctx };
		_crtDirty = true;
		_drawGrid(); // initial capture

		let animId;
		function render(ts) {
			if (!_crt) return;
			if (_crtDirty && scene.width > 0) {
				gl.bindTexture(gl.TEXTURE_2D, tex);
				gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, scene);
				_crtDirty = false;
			}
			const t = ts * 0.001;
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

	function _drawGrid() {
		if (!_crt) return;
		const { ctx: c, scene } = _crt;
		const C = CELL, gw = run.grid_width, gh = run.grid_height;
		c.clearRect(0, 0, scene.width, scene.height);

		// Floor
		c.fillStyle = '#010409';
		c.fillRect(0, 0, gw * C, gh * C);

		// Cells
		for (let y = 0; y < gh; y++) {
			for (let x = 0; x < gw; x++) {
				const key = `${x},${y}`;
				if (wallSet.has(key)) {
					c.fillStyle = (x === 0 || y === 0 || x === gw - 1 || y === gh - 1) ? '#2d333b' : '#373e47';
					c.fillRect(x * C, y * C, C, C);
				} else if (shelfSet.has(key)) {
					c.fillStyle = '#0d2818';
					c.fillRect(x * C, y * C, C, C);
					c.strokeStyle = '#1a4d2e';
					c.lineWidth = 1;
					c.strokeRect(x * C + 0.5, y * C + 0.5, C - 1, C - 1);
				}
			}
		}

		// Drop-off
		const [dx, dy] = run.drop_off;
		c.fillStyle = 'rgba(57,211,83,0.2)';
		c.fillRect(dx * C, dy * C, C, C);
		c.strokeStyle = '#39d353';
		c.lineWidth = 2;
		c.strokeRect(dx * C + 1, dy * C + 1, C - 2, C - 2);
		c.fillStyle = '#39d353';
		c.font = `bold ${Math.floor(C * 0.45)}px sans-serif`;
		c.textAlign = 'center';
		c.textBaseline = 'middle';
		c.fillText('D', dx * C + C / 2, dy * C + C / 2);

		// Spawn
		const [sx, sy] = run.spawn;
		c.fillStyle = 'rgba(88,166,255,0.2)';
		c.fillRect(sx * C, sy * C, C, C);
		c.strokeStyle = '#58a6ff';
		c.lineWidth = 2;
		c.strokeRect(sx * C + 1, sy * C + 1, C - 2, C - 2);
		c.fillStyle = '#58a6ff';
		c.fillText('S', sx * C + C / 2, sy * C + C / 2);

		// Items on shelves (cyberpunk SVG icons) + active/preview highlights
		for (const [key, items] of itemMap) {
			const [ix, iy] = key.split(',').map(Number);
			const t = items[0].type;
			// Active/preview cell highlight
			if (activeTypes.has(t)) {
				c.fillStyle = 'rgba(57, 211, 83, 0.15)';
				c.fillRect(ix * C + 1, iy * C + 1, C - 2, C - 2);
				c.strokeStyle = 'rgba(57, 211, 83, 0.5)';
				c.lineWidth = 1.5;
				c.strokeRect(ix * C + 1.5, iy * C + 1.5, C - 3, C - 3);
			} else if (previewTypes.has(t)) {
				c.fillStyle = 'rgba(210, 153, 34, 0.12)';
				c.fillRect(ix * C + 1, iy * C + 1, C - 2, C - 2);
				c.strokeStyle = 'rgba(210, 153, 34, 0.4)';
				c.lineWidth = 1.5;
				c.strokeRect(ix * C + 1.5, iy * C + 1.5, C - 3, C - 3);
			}
			// Draw item icon (static, but CRT shader applies scanlines/bloom)
			const img = _itemImages.get(t);
			if (img && img.complete && img.naturalWidth > 0) {
				c.drawImage(img, ix * C, iy * C, C, C);
			} else {
				// Fallback text
				const ccx = ix * C + C / 2, ccy = iy * C + C / 2;
				c.fillStyle = '#00FF41'; c.font = `bold ${Math.floor(C * 0.26)}px monospace`;
				c.textAlign = 'center'; c.textBaseline = 'middle';
				c.fillText(ITEM_ABBR[t] || t.slice(0, 2).toUpperCase(), ccx, ccy + 1);
			}
		}

		// Bots (Cyber Drone style)
		for (const bot of bots) {
			const [bx, by] = bot.position;
			const clr = BOT_COLORS[bot.id % BOT_COLORS.length];
			const cx = bx * C, cy = by * C;
			// Shadow
			c.fillStyle = 'rgba(0,0,0,0.35)';
			c.beginPath();
			c.ellipse(cx + C / 2, cy + C - 2, C / 3, 2, 0, 0, Math.PI * 2);
			c.fill();
			// Angular chassis (no border)
			c.fillStyle = '#161B22';
			c.shadowColor = clr;
			c.shadowBlur = 4;
			// Top hull
			c.beginPath();
			c.moveTo(cx + C * 0.32, cy + C * 0.32);
			c.lineTo(cx + C * 0.68, cy + C * 0.32);
			c.lineTo(cx + C * 0.75, cy + C * 0.54);
			c.lineTo(cx + C * 0.25, cy + C * 0.54);
			c.closePath();
			c.fill();
			// Bottom hull
			c.beginPath();
			c.moveTo(cx + C * 0.25, cy + C * 0.54);
			c.lineTo(cx + C * 0.39, cy + C * 0.68);
			c.lineTo(cx + C * 0.61, cy + C * 0.68);
			c.lineTo(cx + C * 0.75, cy + C * 0.54);
			c.fill();
			c.shadowBlur = 0;
			// Visor
			c.fillStyle = '#0D1117';
			c.fillRect(cx + C * 0.36, cy + C * 0.41, C * 0.28, C * 0.11);
			c.strokeStyle = clr;
			c.lineWidth = 0.8;
			c.strokeRect(cx + C * 0.36, cy + C * 0.41, C * 0.28, C * 0.11);
			// Scanner dot
			c.fillStyle = '#FF0055';
			c.fillRect(cx + C * 0.46, cy + C * 0.43, C * 0.08, C * 0.07);
			// ID
			c.fillStyle = clr;
			c.font = `bold ${Math.floor(C * 0.22)}px monospace`;
			c.textAlign = 'center';
			c.textBaseline = 'middle';
			c.fillText(String(bot.id), cx + C / 2, cy + C * 0.62);
			// Inventory dots below
			bot.inventory.forEach((item, i) => {
				c.fillStyle = ITEM_CLR[item] || '#aaa';
				c.beginPath();
				c.arc(
					bx * C + C / 2 - (bot.inventory.length - 1) * 3 + i * 6,
					by * C + C + 3, 2.5, 0, Math.PI * 2
				);
				c.fill();
			});
		}

		_crtDirty = true;
	}

	// Re-draw when round changes (track all reactive deps used in _drawGrid)
	$effect(() => {
		const _ = currentRound;
		const _b = bots; // track bot positions
		const _at = activeTypes; // track active order types
		const _pt = previewTypes; // track preview order types
		if (!_crt) return;
		_drawGrid();
	});
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
							spawn={run.spawn}
							{bots}
							{botPositions}
							botColors={BOT_COLORS}
							{selectedBot}
							onSelectBot={(id) => selectedBot = selectedBot === id ? null : id}
							{activeTypes}
							{previewTypes}
						/>
						<!-- Highlight overlay for active/preview order items -->
					<div class="item-overlay">
						{#each shelfItems as si}
							{@const isActive = activeTypes.has(si.type)}
							{@const isPreview = !isActive && previewTypes.has(si.type)}
							{#if isActive || isPreview}
								<div
									class="item-cell"
									class:item-active={isActive}
									class:item-preview={isPreview}
									style="left: {si.x / run.grid_width * 100}%; top: {si.y / run.grid_height * 100}%; width: {100 / run.grid_width}%; height: {100 / run.grid_height}%;"
								></div>
							{/if}
						{/each}
					</div>
					<canvas bind:this={crtCanvas} class="crt-overlay"></canvas>
					</div>
				</div>
			</div>

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
	.item-overlay {
		position: absolute;
		inset: 0.5rem;
		pointer-events: none;
		z-index: 3;
	}
	.item-cell {
		position: absolute;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 2px;
	}
	.item-cell.item-active {
		box-shadow: 0 0 8px rgba(250, 204, 21, 0.7), inset 0 0 4px rgba(250, 204, 21, 0.2);
		animation: itemActivePulse 1.5s ease-in-out infinite;
	}
	.item-cell.item-preview {
		box-shadow: 0 0 6px rgba(244, 114, 182, 0.5), inset 0 0 3px rgba(244, 114, 182, 0.15);
		animation: itemPreviewPulse 2s ease-in-out infinite;
	}
	@keyframes itemActivePulse {
		0%, 100% { box-shadow: 0 0 6px rgba(250, 204, 21, 0.5), inset 0 0 3px rgba(250, 204, 21, 0.15); }
		50% { box-shadow: 0 0 12px rgba(250, 204, 21, 0.8), inset 0 0 6px rgba(250, 204, 21, 0.3); }
	}
	@keyframes itemPreviewPulse {
		0%, 100% { box-shadow: 0 0 4px rgba(244, 114, 182, 0.35), inset 0 0 2px rgba(244, 114, 182, 0.1); }
		50% { box-shadow: 0 0 8px rgba(244, 114, 182, 0.6), inset 0 0 4px rgba(244, 114, 182, 0.2); }
	}
	:global(.item-cell svg) {
		display: block;
		width: 100% !important;
		height: 100% !important;
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
