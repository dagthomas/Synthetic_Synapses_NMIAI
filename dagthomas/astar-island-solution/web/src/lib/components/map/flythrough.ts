/**
 * Eagle-like FPV flythrough with dramatic swoops.
 *
 * Each start() generates a fresh random flight path by shuffling terrain
 * features and randomizing approach angles, distances, and altitudes.
 * No two flights are the same.
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

/** Fisher-Yates shuffle (in-place) */
function shuffle<T>(arr: T[]): T[] {
	for (let i = arr.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[arr[i], arr[j]] = [arr[j], arr[i]];
	}
	return arr;
}

/** Ensure minimum distance between consecutive waypoints */
function enforceMinDist(pts: THREE.Vector3[], minDist: number): THREE.Vector3[] {
	if (pts.length < 2) return pts;
	const out = [pts[0]];
	for (let i = 1; i < pts.length; i++) {
		if (pts[i].distanceTo(out[out.length - 1]) >= minDist) {
			out.push(pts[i]);
		}
	}
	return out;
}

function buildRandomFlightPath(
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

	// Collect all interesting features
	const features: { pos: THREE.Vector3; groundH: number; type: string }[] = [];

	for (const c of clusters) {
		const pos = new THREE.Vector3(c.centerX + ox + 0.5, 0, c.centerY + oz + 0.5);
		const groundH = heightFn(pos.x, pos.z);
		let type = 'other';
		if (c.terrainType === TerrainCode.MOUNTAIN) type = 'mountain';
		else if (c.terrainType === TerrainCode.SETTLEMENT) type = 'settlement';
		else if (c.terrainType === TerrainCode.FOREST) type = 'forest';
		else if (c.terrainType === TerrainCode.RUIN) type = 'ruin';
		else if (c.terrainType === TerrainCode.PORT) type = 'port';
		else continue;
		features.push({ pos, groundH, type });
	}

	// Shuffle and pick 6-10 features for this flight
	shuffle(features);
	const count = Math.min(features.length, 6 + Math.floor(Math.random() * 5));
	const selected = features.slice(0, count);

	const rAngle = () => Math.random() * Math.PI * 2;
	const rRange = (lo: number, hi: number) => lo + Math.random() * (hi - lo);

	const orb = (c: THREE.Vector3, a: number, r: number, alt: number) =>
		new THREE.Vector3(c.x + Math.cos(a) * r, alt, c.z + Math.sin(a) * r);

	// Swoop: climb → dive → pull-up (3 points)
	function swoop(target: THREE.Vector3, groundH: number) {
		const approachAngle = rAngle();
		const exitAngle = approachAngle + rRange(1.5, 3.0);
		const approachR = rRange(4, 8);
		const exitR = rRange(4, 8);
		const peakAlt = rRange(2.5, 4.5);
		const nadirAlt = rRange(0.3, 0.8);

		pts.push(orb(target, approachAngle, approachR, groundH + peakAlt));
		pts.push(orb(target, (approachAngle + exitAngle) / 2, approachR * 0.35, groundH + nadirAlt));
		pts.push(orb(target, exitAngle, exitR, groundH + peakAlt * 0.85));
	}

	// Canopy skim: 4 points weaving low over the feature
	function skim(target: THREE.Vector3, groundH: number) {
		const angle = rAngle();
		const dx = Math.cos(angle);
		const dz = Math.sin(angle);
		const spread = rRange(4, 7);

		pts.push(new THREE.Vector3(target.x - dx * spread, groundH + rRange(2, 3.5), target.z - dz * spread));
		pts.push(new THREE.Vector3(target.x - dx * 1.5, groundH + rRange(0.4, 0.7), target.z - dz * 1.5));
		pts.push(new THREE.Vector3(target.x + dx * 1.5, groundH + rRange(0.4, 0.8), target.z + dz * 1.5));
		pts.push(new THREE.Vector3(target.x + dx * spread, groundH + rRange(2.5, 4), target.z + dz * spread));
	}

	// Random entry point from map edge
	const entryAngle = rAngle();
	pts.push(new THREE.Vector3(
		Math.cos(entryAngle) * mapR * 0.8,
		rRange(3, 5),
		Math.sin(entryAngle) * mapR * 0.8
	));

	// Visit each selected feature with a random flight pattern
	for (const feat of selected) {
		// Transition: add a cruise point between features for spacing
		const lastPt = pts[pts.length - 1];
		const mid = new THREE.Vector3().lerpVectors(lastPt, feat.pos, 0.5);
		mid.y = feat.groundH + rRange(2, 4);
		// Offset mid sideways for less linear paths
		mid.x += rRange(-3, 3);
		mid.z += rRange(-3, 3);
		pts.push(mid);

		// Choose flight pattern based on feature type + randomness
		const pattern = Math.random();
		if (feat.type === 'forest' || (feat.type === 'ruin' && pattern < 0.5)) {
			skim(feat.pos, feat.groundH);
		} else {
			swoop(feat.pos, feat.groundH);
		}
	}

	// Closing: curve back toward entry for smooth loop
	const closingAngle = entryAngle + rRange(0.5, 1.5);
	pts.push(new THREE.Vector3(
		Math.cos(closingAngle) * mapR * 0.6,
		rRange(3, 5),
		Math.sin(closingAngle) * mapR * 0.6
	));

	// Enforce minimum 3-unit distance between consecutive waypoints
	const filtered = enforceMinDist(pts, 3);

	// Fallback if not enough points
	if (filtered.length < 8) {
		const fallback: THREE.Vector3[] = [];
		for (let i = 0; i < 12; i++) {
			const a = (i / 12) * Math.PI * 2 + Math.random() * 0.3;
			const alt = 3 + Math.sin(a * 3) * 4;
			fallback.push(new THREE.Vector3(
				Math.cos(a) * mapR * 0.6,
				alt,
				Math.sin(a) * mapR * 0.6
			));
		}
		return fallback;
	}

	return filtered;
}

export function createFlythrough(
	grid: number[][],
	heightFn: (x: number, z: number) => number
): FlythroughSystem {
	let pathPoints = buildRandomFlightPath(grid, heightFn);
	let curve = new THREE.CatmullRomCurve3(pathPoints, true, 'catmullrom', 0.08);
	let loopDuration = curve.getLength() / 1.8;

	let t = 0;
	let active = false;

	const _smoothLook = new THREE.Vector3();
	const _tangent = new THREE.Vector3();
	const _up = new THREE.Vector3(0, 1, 0);
	const _right = new THREE.Vector3();
	const _prevTangent = new THREE.Vector3(0, 0, -1);
	const _rollQuat = new THREE.Quaternion();
	const _fwd = new THREE.Vector3();
	let smoothRoll = 0;
	let smoothPitch = 0;

	return {
		get active() { return active; },
		set active(v: boolean) { active = v; },

		update(camera: THREE.PerspectiveCamera, dt: number) {
			if (!active) return;

			t += dt / loopDuration;
			if (t > 1) {
				// Generate a new random path for the next loop
				t -= 1;
				pathPoints = buildRandomFlightPath(grid, heightFn);
				curve = new THREE.CatmullRomCurve3(pathPoints, true, 'catmullrom', 0.08);
				loopDuration = curve.getLength() / 1.8;
				smoothRoll = 0;
				smoothPitch = 0;
				_prevTangent.set(0, 0, -1);
			}

			// Position on spline
			const pos = curve.getPointAt(t);

			// Very gentle breathing — barely perceptible
			const breathPhase = t * Math.PI * 10;
			pos.y += Math.sin(breathPhase) * 0.03;

			// Minimal sway — like wind
			curve.getTangentAt(t, _tangent);
			_right.crossVectors(_tangent, _up).normalize();
			pos.addScaledVector(_right, Math.sin(breathPhase * 0.4) * 0.015);

			camera.position.copy(pos);

			// Smooth the tangent direction (always faces forward — never backward)
			if (_smoothLook.lengthSq() < 0.001) {
				_smoothLook.copy(_tangent);
			} else {
				_smoothLook.lerp(_tangent, Math.min(1, dt * 4.0));
			}
			_smoothLook.normalize();

			// Look along smoothed tangent
			const lookTarget = pos.clone().addScaledVector(_smoothLook, 5);

			// Gentle pitch — follow terrain slope
			const pitch = _tangent.y;
			const targetPitchOffset = pitch < -0.1 ? pitch * 0.5 : pitch * 0.15;
			smoothPitch += (targetPitchOffset - smoothPitch) * Math.min(1, dt * 1.5);
			lookTarget.y += smoothPitch;

			// Force world-up — prevents flip
			camera.up.set(0, 1, 0);
			camera.lookAt(lookTarget);

			// Gentle banking — like a glider, not a fighter jet
			curve.getTangentAt(t, _tangent);
			const crossTurn = _prevTangent.x * _tangent.z - _prevTangent.z * _tangent.x;

			const swoopFactor = 1 + Math.abs(pitch) * 1.5;
			const targetRoll = -crossTurn * 3 * swoopFactor;
			const maxRoll = 0.18; // ~10 degrees — subtle, graceful
			const clampedRoll = Math.max(-maxRoll, Math.min(maxRoll, targetRoll));
			smoothRoll += (clampedRoll - smoothRoll) * Math.min(1, dt * 0.8);

			// Apply roll as quaternion rotation around camera's forward vector
			camera.getWorldDirection(_fwd);
			_rollQuat.setFromAxisAngle(_fwd, smoothRoll);
			camera.quaternion.premultiply(_rollQuat);

			_prevTangent.copy(_tangent);
		},

		start(camera: THREE.PerspectiveCamera) {
			// Generate fresh random path each time
			pathPoints = buildRandomFlightPath(grid, heightFn);
			curve = new THREE.CatmullRomCurve3(pathPoints, true, 'catmullrom', 0.08);
			loopDuration = curve.getLength() / 1.8;

			active = true;
			t = 0;
			smoothRoll = 0;
			smoothPitch = 0;
			camera.up.set(0, 1, 0);
			camera.position.copy(pathPoints[0]);
			// Initialize smoothLook as forward tangent direction
			curve.getTangentAt(0, _smoothLook);
			_prevTangent.copy(_smoothLook);
			const initLook = pathPoints[0].clone().addScaledVector(_smoothLook, 5);
			camera.lookAt(initLook);
		},

		stop() {
			active = false;
		},

		dispose() {
			active = false;
		}
	};
}
