// Day/night cycle: time-of-day → light colors, sun position, sky/fog, moon position
// Optimized: zero allocations per call — reuses a single cached state object.

import * as THREE from 'three';

export interface DayNightState {
	sunPosition: THREE.Vector3;
	sunColor: THREE.Color;
	sunIntensity: number;
	ambientColor: THREE.Color;
	ambientIntensity: number;
	hemiSkyColor: THREE.Color;
	hemiGroundColor: THREE.Color;
	hemiIntensity: number;
	skyColor: THREE.Color;
	fogColor: THREE.Color;
	fogDensity: number;
	moonPosition: THREE.Vector3;
	nightFade: number;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
	return t * t * (3 - 2 * t);
}

function lerpHex(a: number, b: number, t: number, out: THREE.Color): void {
	_ca.set(a);
	_cb.set(b);
	out.copy(_ca).lerp(_cb, t);
}

// Reusable scratch colors (never exposed)
const _ca = new THREE.Color();
const _cb = new THREE.Color();

// Single cached state — returned every call, mutated in-place
const _state: DayNightState = {
	sunPosition: new THREE.Vector3(),
	sunColor: new THREE.Color(),
	sunIntensity: 0,
	ambientColor: new THREE.Color(),
	ambientIntensity: 0,
	hemiSkyColor: new THREE.Color(),
	hemiGroundColor: new THREE.Color(),
	hemiIntensity: 0,
	skyColor: new THREE.Color(),
	fogColor: new THREE.Color(),
	fogDensity: 0,
	moonPosition: new THREE.Vector3(),
	nightFade: 0
};

export function computeDayNight(hour: number): DayNightState {
	const h = ((hour % 24) + 24) % 24;

	// Sun orbit
	const sunAngle = ((h - 6) / 12) * Math.PI;
	_state.sunPosition.set(80 * Math.cos(sunAngle), 60 * Math.sin(sunAngle), 0);

	// Moon orbit
	const moonAngle = sunAngle + Math.PI;
	_state.moonPosition.set(70 * Math.cos(moonAngle), 50 * Math.sin(moonAngle), 0);

	// Phase blending
	const nightFade = Math.min(1, Math.max(0, 1 - smoothstep(5, 7, h) + smoothstep(17, 19, h)));
	const dawnFade = smoothstep(5, 6.5, h) * (1 - smoothstep(7, 8, h));
	const duskFade = smoothstep(17, 18, h) * (1 - smoothstep(18.5, 19.5, h));
	const dayFade = smoothstep(7, 8, h) * (1 - smoothstep(17, 18, h));
	_state.nightFade = nightFade;

	// Sun color (mutate in-place)
	if (dawnFade > 0.01) {
		lerpHex(0xffb347, 0xfff8e1, smoothstep(6, 8, h), _state.sunColor);
	} else if (duskFade > 0.01) {
		lerpHex(0xfff8e1, 0xff7043, smoothstep(17, 19, h), _state.sunColor);
	} else if (dayFade > 0.5) {
		_state.sunColor.set(0xfff8e1);
	} else {
		_state.sunColor.set(0x8899cc); // cool bluish-white moonlight for specular
	}

	// Night uses sunLight as bright moonlight for specular highlights
	_state.sunIntensity = dayFade * 1.2 + dawnFade * 0.6 + duskFade * 0.6 + nightFade * 0.55;

	lerpHex(0x4466aa, 0x606080, 1 - nightFade, _state.ambientColor);
	_state.ambientIntensity = 0.35 + dayFade * 0.15 + (dawnFade + duskFade) * 0.15 + nightFade * 0.25;

	lerpHex(0x2a3870, 0x87ceeb, 1 - nightFade * 0.7, _state.hemiSkyColor);
	lerpHex(0x182030, 0x5d4e37, 1 - nightFade * 0.6, _state.hemiGroundColor);
	_state.hemiIntensity = 0.30 + dayFade * 0.35 + (dawnFade + duskFade) * 0.2 + nightFade * 0.22;

	// Sky color
	if (nightFade > 0.8) {
		_state.skyColor.set(0x0a0a2a);
	} else if (dawnFade > 0.1) {
		lerpHex(0x0a0a2a, 0xff9966, dawnFade, _state.skyColor);
		_state.skyColor.lerp(_cb.set(0x87ceeb), dayFade);
	} else if (duskFade > 0.1) {
		lerpHex(0x87ceeb, 0xff6b35, duskFade, _state.skyColor);
		_state.skyColor.lerp(_cb.set(0x0a0a2a), nightFade);
	} else {
		lerpHex(0x0a0a2a, 0x87ceeb, dayFade, _state.skyColor);
	}

	_state.fogColor.copy(_state.skyColor);
	_state.fogDensity = 0.006 + nightFade * 0.003;

	return _state;
}
