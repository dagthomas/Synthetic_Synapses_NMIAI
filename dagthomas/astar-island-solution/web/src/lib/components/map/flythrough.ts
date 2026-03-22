/**
 * Eagle-like FPV flythrough with dramatic swoops.
 *
 * Swoops = climb high → nose-dive toward a feature → pull up at the last
 * moment → soar back up. The path bakes these altitude oscillations into
 * the spline so the Catmull-Rom interpolation produces smooth, continuous
 * curves rather than abrupt changes.
 *
 * Banking roll, forward-facing camera, gentle lateral sway.
 */
import * as THREE from 'three';
import { findClusters, type Cluster } from './clusters';
import { TerrainCode } from '$lib/types';

export interface FlythroughSystem {
	active: boolean;
	update(camera: THREE.PerspectiveCamera, dt: number): void;
	start(camera: THREE.PerspectiveCamera): void;
	stop(): void;
	dispose(): void;
}

function buildFlightPath(
	grid: number[][],
	heightFn: (x: number, z: number) => number
): THREE.Vector3[] {
	const rows = grid.length;
	const cols = grid[0].length;
	const ox = -cols / 2;
	const oz = -rows / 2;
	const mapR = Math.max(cols, rows) / 2;

	const clusters = findClusters(grid);
	const pts: THREE.Vector3[] = [];

	const mountains  = clusters.filter(c => c.terrainType === TerrainCode.MOUNTAIN).sort((a, b) => b.size - a.size);
	const settlements = clusters.filter(c => c.terrainType === TerrainCode.SETTLEMENT).sort((a, b) => b.size - a.size);
	const forests    = clusters.filter(c => c.terrainType === TerrainCode.FOREST).sort((a, b) => b.size - a.size);
	const ruins      = clusters.filter(c => c.terrainType === TerrainCode.RUIN);
	const ports      = clusters.filter(c => c.terrainType === TerrainCode.PORT);

	const w = (c: Cluster) => new THREE.Vector3(c.centerX + ox + 0.5, 0, c.centerY + oz + 0.5);
	const h = (x: number, z: number) => heightFn(x, z);

	// Orbit helper
	const orb = (c: THREE.Vector3, a: number, r: number, alt: number) =>
		new THREE.Vector3(c.x + Math.cos(a) * r, alt, c.z + Math.sin(a) * r);

	// Swoop helper — creates 3 points: climb → dive → pull-up
	// approach: high point before the feature
	// nadir:    low point right at/over the feature
	// exit:     climbing away
	function swoop(
		target: THREE.Vector3,
		groundH: number,
		approachAngle: number,
		approachR: number,
		exitAngle: number,
		exitR: number,
		peakAlt: number,   // altitude of climb/exit
		nadirAlt: number   // altitude of the low swoop point
	) {
		// 1. Climb before the swoop
		pts.push(orb(target, approachAngle, approachR, groundH + peakAlt));
		// 2. Dive down — close to the feature, low altitude
		pts.push(orb(target, (approachAngle + exitAngle) / 2, approachR * 0.35, groundH + nadirAlt));
		// 3. Pull up and soar away
		pts.push(orb(target, exitAngle, exitR, groundH + peakAlt * 0.85));
	}

	// =========================================================
	// Flight plan — each section has a dramatic swoop
	// =========================================================

	// --- Opening: high panoramic entry ---
	pts.push(new THREE.Vector3(ox + cols * 0.85, 12, oz + rows * 0.85));
	pts.push(new THREE.Vector3(ox + cols * 0.65, 9, oz + rows * 0.65));

	// --- Mountain 1: eagle dive past the peak ---
	if (mountains.length > 0) {
		const m = w(mountains[0]);
		const mh = h(m.x, m.z);
		// Wide approach arc
		pts.push(orb(m, -0.8, 7, mh + 6));
		// Swoop: climb high, dive past the summit, pull up
		swoop(m, mh, 0.5, 6, 2.5, 6, 7, 1.0);
	}

	// --- Settlement 1: low buzz swoop ---
	if (settlements.length > 0) {
		const s = w(settlements[0]);
		const sh = h(s.x, s.z);
		// Transition cruise
		pts.push(orb(s, -1.2, 7, sh + 5));
		// Swoop down through the settlement
		swoop(s, sh, -0.3, 5, 1.8, 5, 5, 0.6);
	}

	// --- Forest: canopy skim swoop ---
	if (forests.length > 0) {
		const f = w(forests[0]);
		const fh = h(f.x, f.z);
		pts.push(new THREE.Vector3(f.x - 6, fh + 5, f.z - 2));
		// Dive into the treetops
		pts.push(new THREE.Vector3(f.x - 1, fh + 0.8, f.z));
		// Skim along the canopy
		pts.push(new THREE.Vector3(f.x + 2, fh + 1.0, f.z + 1));
		// Pull up out of the forest
		pts.push(new THREE.Vector3(f.x + 5, fh + 5, f.z + 3));
	}

	// --- Ruins: contemplative swoop ---
	if (ruins.length > 0) {
		const r = w(ruins[0]);
		const rh = h(r.x, r.z);
		swoop(r, rh, -0.5, 5, 2.0, 5, 4, 0.8);
	}

	// --- Port: water-level swoop ---
	if (ports.length > 0) {
		const p = w(ports[0]);
		const ph = h(p.x, p.z);
		pts.push(new THREE.Vector3(p.x - 6, ph + 5, p.z - 1));
		// Dive down to water level
		pts.push(new THREE.Vector3(p.x - 1, ph + 0.3, p.z + 1));
		// Skim the water surface
		pts.push(new THREE.Vector3(p.x + 2, ph + 0.4, p.z + 2));
		// Soar back up
		pts.push(new THREE.Vector3(p.x + 6, ph + 6, p.z));
	}

	// --- Mountain 2: dramatic banking swoop ---
	if (mountains.length > 1) {
		const m2 = w(mountains[1]);
		const m2h = h(m2.x, m2.z);
		pts.push(orb(m2, -1.0, 7, m2h + 5));
		swoop(m2, m2h, 0.3, 5, 2.8, 6, 8, 1.5);
	}

	// --- Settlement 2: tight low orbit swoop ---
	if (settlements.length > 1) {
		const s2 = w(settlements[1]);
		const s2h = h(s2.x, s2.z);
		pts.push(orb(s2, -0.8, 6, s2h + 4));
		swoop(s2, s2h, 0.2, 4, 2.2, 5, 5, 0.5);
	}

	// --- Forest 2: dive and weave ---
	if (forests.length > 1) {
		const f2 = w(forests[1]);
		const f2h = h(f2.x, f2.z);
		pts.push(new THREE.Vector3(f2.x - 5, f2h + 5, f2.z));
		pts.push(new THREE.Vector3(f2.x, f2h + 0.9, f2.z + 1));
		pts.push(new THREE.Vector3(f2.x + 5, f2h + 5.5, f2.z - 1));
	}

	// --- Grand finale: sweeping climb + panoramic swoop ---
	pts.push(new THREE.Vector3(ox + cols * 0.2, 4, oz + rows * 0.2));
	// Big climb
	pts.push(new THREE.Vector3(ox + cols * 0.4, 12, oz + rows * 0.4));
	// Final swoop toward start
	pts.push(new THREE.Vector3(ox + cols * 0.6, 5, oz + rows * 0.6));
	pts.push(new THREE.Vector3(ox + cols * 0.75, 10, oz + rows * 0.75));

	// Fallback
	if (pts.length < 8) {
		for (let i = 0; i < 12; i++) {
			const a = (i / 12) * Math.PI * 2;
			const alt = 3 + Math.sin(a * 3) * 4; // swooping altitude
			pts.push(new THREE.Vector3(
				Math.cos(a) * mapR * 0.6,
				alt,
				Math.sin(a) * mapR * 0.6
			));
		}
	}

	return pts;
}

export function createFlythrough(
	grid: number[][],
	heightFn: (x: number, z: number) => number
): FlythroughSystem {
	const pathPoints = buildFlightPath(grid, heightFn);

	// Low tension for buttery smooth curves
	const curve = new THREE.CatmullRomCurve3(
		pathPoints, true, 'catmullrom', 0.15
	);

	const totalLength = curve.getLength();
	const SPEED = 3.0;
	const loopDuration = totalLength / SPEED;

	let t = 0;
	let active = false;

	const _smoothLook = new THREE.Vector3();
	const _tangent = new THREE.Vector3();
	const _up = new THREE.Vector3(0, 1, 0);
	const _right = new THREE.Vector3();
	const _prevTangent = new THREE.Vector3(0, 0, -1);
	let smoothRoll = 0;
	let smoothPitch = 0;

	return {
		get active() { return active; },
		set active(v: boolean) { active = v; },

		update(camera: THREE.PerspectiveCamera, dt: number) {
			if (!active) return;

			t += dt / loopDuration;
			if (t > 1) t -= 1;

			// Position on spline
			const pos = curve.getPointAt(t);

			// Micro-undulation on top of the baked swoops — breathing feel
			const breathPhase = t * Math.PI * 20;
			pos.y += Math.sin(breathPhase) * 0.06;

			// Subtle lateral sway
			curve.getTangentAt(t, _tangent);
			_right.crossVectors(_tangent, _up).normalize();
			pos.addScaledVector(_right, Math.sin(breathPhase * 0.6) * 0.04);

			camera.position.copy(pos);

			// Look forward — 4% ahead on spline
			const lookT = (t + 0.04) % 1;
			const lookPos = curve.getPointAt(lookT);

			// During dives (negative tangent.y), look further down to enhance swoop feel
			const pitch = _tangent.y;
			const targetPitchOffset = pitch < -0.1 ? pitch * 1.5 : pitch * 0.3;
			smoothPitch += (targetPitchOffset - smoothPitch) * Math.min(1, dt * 3);
			lookPos.y += smoothPitch;

			// Smooth look interpolation
			if (_smoothLook.lengthSq() < 0.001) {
				_smoothLook.copy(lookPos);
			} else {
				_smoothLook.lerp(lookPos, Math.min(1, dt * 2.5));
			}

			camera.lookAt(_smoothLook);

			// Banking roll — stronger into swoops and turns
			curve.getTangentAt(t, _tangent);
			const crossTurn = _prevTangent.x * _tangent.z - _prevTangent.z * _tangent.x;
			const turnMagnitude = Math.abs(crossTurn);

			// More aggressive roll during high-speed swoops (steep tangent)
			const swoopFactor = 1 + Math.abs(pitch) * 3;
			const targetRoll = -crossTurn * 10 * swoopFactor;
			const maxRoll = 0.40; // ~23 degrees
			const clampedRoll = Math.max(-maxRoll, Math.min(maxRoll, targetRoll));
			smoothRoll += (clampedRoll - smoothRoll) * Math.min(1, dt * 2.5);
			camera.rotation.z = smoothRoll;

			_prevTangent.copy(_tangent);
		},

		start(camera: THREE.PerspectiveCamera) {
			active = true;
			t = 0;
			smoothRoll = 0;
			smoothPitch = 0;
			camera.position.copy(pathPoints[0]);
			_smoothLook.copy(pathPoints[1] || pathPoints[0]);
			_prevTangent.set(0, 0, -1);
			camera.lookAt(_smoothLook);
			camera.rotation.z = 0;
		},

		stop() {
			active = false;
		},

		dispose() {
			active = false;
		}
	};
}
