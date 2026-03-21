// Day/night cycle: time-of-day → light colors, sun position, sky/fog, moon position

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

function lerpColor(a: number, b: number, t: number): THREE.Color {
	const ca = new THREE.Color(a);
	const cb = new THREE.Color(b);
	return ca.lerp(cb, t);
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
	return t * t * (3 - 2 * t);
}

export function computeDayNight(hour: number): DayNightState {
	// Normalize hour to 0-24
	const h = ((hour % 24) + 24) % 24;

	// Sun orbit: rises at 6, sets at 18
	const sunAngle = ((h - 6) / 12) * Math.PI;
	const sunX = 40 * Math.cos(sunAngle);
	const sunY = 30 * Math.sin(sunAngle);
	const sunZ = 10;

	// Moon orbit: opposite to sun, slightly offset in Z
	const moonAngle = sunAngle + Math.PI;
	const moonX = 35 * Math.cos(moonAngle);
	const moonY = 25 * Math.sin(moonAngle);
	const moonZ = -8;

	// Phase blending factors
	const nightFade = Math.min(1, Math.max(0, 1 - smoothstep(5, 7, h) + smoothstep(17, 19, h)));
	const dawnFade = smoothstep(5, 6.5, h) * (1 - smoothstep(7, 8, h));
	const duskFade = smoothstep(17, 18, h) * (1 - smoothstep(18.5, 19.5, h));
	const dayFade = smoothstep(7, 8, h) * (1 - smoothstep(17, 18, h));

	// Sun color
	let sunColor: THREE.Color;
	if (dawnFade > 0.01) {
		sunColor = lerpColor(0xffb347, 0xfff8e1, smoothstep(6, 8, h));
	} else if (duskFade > 0.01) {
		sunColor = lerpColor(0xfff8e1, 0xff7043, smoothstep(17, 19, h));
	} else if (dayFade > 0.5) {
		sunColor = new THREE.Color(0xfff8e1);
	} else {
		sunColor = new THREE.Color(0x334466);
	}

	// Moonlight provides visible illumination at night
	const sunIntensity = dayFade * 1.2 + dawnFade * 0.6 + duskFade * 0.6 + nightFade * 0.20;

	const ambientColor = lerpColor(0x2a3366, 0x606080, 1 - nightFade);
	const ambientIntensity = 0.35 + dayFade * 0.15 + (dawnFade + duskFade) * 0.15 + nightFade * 0.10;

	const hemiSkyColor = lerpColor(0x1a2850, 0x87ceeb, 1 - nightFade * 0.7);
	const hemiGroundColor = lerpColor(0x101820, 0x5d4e37, 1 - nightFade * 0.6);
	const hemiIntensity = 0.30 + dayFade * 0.35 + (dawnFade + duskFade) * 0.2 + nightFade * 0.12;

	let skyColor: THREE.Color;
	if (nightFade > 0.8) {
		skyColor = new THREE.Color(0x0a0a2a);
	} else if (dawnFade > 0.1) {
		skyColor = lerpColor(0x0a0a2a, 0xff9966, dawnFade).lerp(new THREE.Color(0x87ceeb), dayFade);
	} else if (duskFade > 0.1) {
		skyColor = lerpColor(0x87ceeb, 0xff6b35, duskFade).lerp(new THREE.Color(0x0a0a2a), nightFade);
	} else {
		skyColor = lerpColor(0x0a0a2a, 0x87ceeb, dayFade);
	}

	const fogColor = skyColor.clone();
	const fogDensity = 0.006 + nightFade * 0.003; // less fog at night so you can see the fires

	return {
		sunPosition: new THREE.Vector3(sunX, sunY, sunZ),
		sunColor,
		sunIntensity,
		ambientColor,
		ambientIntensity,
		hemiSkyColor,
		hemiGroundColor,
		hemiIntensity,
		skyColor,
		fogColor,
		fogDensity,
		moonPosition: new THREE.Vector3(moonX, moonY, moonZ),
		nightFade
	};
}
