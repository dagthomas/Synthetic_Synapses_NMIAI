/**
 * Cinematic flythrough — roams around the map perimeter looking inward,
 * with occasional glances toward sun/moon/aurora (5% of the time).
 * Smooth wide arcs, never reaches the grid edge, always keeps terrain in view.
 */
import * as THREE from 'three';

export interface FlythroughSystem {
	active: boolean;
	update(camera: THREE.PerspectiveCamera, dt: number): void;
	start(camera: THREE.PerspectiveCamera): void;
	transitionToNewPath(newGrid: number[][], newHeightFn: (x: number, z: number) => number): void;
	stop(): void;
	dispose(): void;
}

const FLIGHT_SPEED = 1.8;
const MIN_CLEARANCE = 1.8;
const MAX_ROLL = 0.16;
const LOOK_AHEAD = 10;
const LOOK_DOWN = 4.0;
const SPLINE_TENSION = 0.45;

function seededRng(seed: number) {
	let s = seed;
	return () => {
		s = (s * 1664525 + 1013904223) & 0x7fffffff;
		return s / 0x7fffffff;
	};
}

function buildPath(
	grid: number[][],
	heightFn: (x: number, z: number) => number,
	startPos?: THREE.Vector3,
	startDir?: THREE.Vector3
): THREE.Vector3[] {
	const rows = grid.length;
	const cols = grid[0].length;
	const halfW = cols / 2, halfH = rows / 2;
	const rng = seededRng(Date.now() & 0xffff);

	// Orbit radius: roam around the mid-ring of the map (40-70% from center)
	const innerR = Math.min(halfW, halfH) * 0.35;
	const outerR = Math.min(halfW, halfH) * 0.65;

	const pts: THREE.Vector3[] = [];
	const orbitDir = rng() > 0.5 ? 1 : -1;
	const baseAngle = rng() * Math.PI * 2;
	const baseAlt = 5 + rng() * 2;

	// Bridge from current position
	if (startPos && startDir) {
		pts.push(startPos.clone());
		const bridge = startPos.clone().addScaledVector(startDir, 5);
		bridge.y = Math.max(bridge.y, heightFn(bridge.x, bridge.z) + MIN_CLEARANCE);
		pts.push(bridge);
	}

	// ~16 points around a wobbly orbit — mostly mid-ring, occasionally dipping inward
	const numPoints = 14 + Math.floor(rng() * 4);

	for (let i = 0; i < numPoints; i++) {
		const t = i / numPoints;
		const angle = baseAngle + t * Math.PI * 2 * orbitDir;

		// Radius wobbles between inner and outer ring
		const wobble = Math.sin(angle * 3 + rng() * 2) * 0.5 + 0.5; // 0-1
		const r = innerR + wobble * (outerR - innerR);

		// Occasional inward dip (every ~4th point, 25% chance)
		const dipInward = rng() < 0.15;
		const actualR = dipInward ? r * 0.4 : r;

		let x = Math.cos(angle) * actualR;
		let z = Math.sin(angle) * actualR;

		// Clamp to grid
		x = Math.max(-halfW * 0.8, Math.min(halfW * 0.8, x));
		z = Math.max(-halfH * 0.8, Math.min(halfH * 0.8, z));

		const groundH = heightFn(
			Math.max(-halfW + 0.5, Math.min(halfW - 0.5, x)),
			Math.max(-halfH + 0.5, Math.min(halfH - 0.5, z))
		);

		// Altitude: dramatic variation — high swoops + low passes
		const altWave = Math.sin(t * Math.PI * 3) * 3.0; // bigger swings
		const distFactor = actualR / outerR;
		const lowPass = dipInward ? -2.5 : 0; // dip low when cutting through center
		const y = Math.max(groundH + MIN_CLEARANCE, baseAlt + altWave + distFactor * 3 + lowPass);

		pts.push(new THREE.Vector3(x, y, z));
	}

	return pts;
}

export function createFlythrough(
	initialGrid: number[][],
	initialHeightFn: (x: number, z: number) => number
): FlythroughSystem {
	let _grid = initialGrid;
	let _heightFn = initialHeightFn;
	let pathPoints = buildPath(_grid, _heightFn);
	let curve = new THREE.CatmullRomCurve3(pathPoints, true, 'catmullrom', SPLINE_TENSION);
	let loopDuration = curve.getLength() / FLIGHT_SPEED;

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

	// Sky glance state: occasionally look up toward sun/moon/aurora
	let skyGlanceActive = false;
	let skyGlanceTimer = 8 + Math.random() * 15; // first glance after 8-23s
	let skyGlanceProgress = 0;
	const SKY_GLANCE_DURATION = 3.0; // seconds looking up
	const SKY_GLANCE_CHANCE = 0.05; // 5% check each cycle

	function regeneratePath(fromPos?: THREE.Vector3, fromDir?: THREE.Vector3) {
		pathPoints = buildPath(_grid, _heightFn, fromPos, fromDir);
		curve = new THREE.CatmullRomCurve3(pathPoints, true, 'catmullrom', SPLINE_TENSION);
		loopDuration = curve.getLength() / FLIGHT_SPEED;
		t = 0;
	}

	return {
		get active() { return active; },
		set active(v: boolean) { active = v; },

		update(camera: THREE.PerspectiveCamera, dt: number) {
			if (!active) return;

			t += dt / loopDuration;
			if (t > 1) {
				const pos = curve.getPointAt(0.999);
				const dir = _smoothLook.clone().normalize();
				regeneratePath(pos, dir);
				smoothRoll *= 0.3;
			}

			const pos = curve.getPointAt(t);

			// Subtle breathing
			pos.y += Math.sin(t * Math.PI * 10) * 0.015;

			// Terrain clearance
			curve.getTangentAt(t, _tangent);
			const groundH = _heightFn(pos.x, pos.z);
			if (pos.y < groundH + MIN_CLEARANCE) {
				pos.y += (groundH + MIN_CLEARANCE - pos.y + 0.5) * dt * 4;
			}

			camera.position.copy(pos);

			// Smooth look direction
			if (_smoothLook.lengthSq() < 0.001) {
				_smoothLook.copy(_tangent);
			} else {
				_smoothLook.lerp(_tangent, Math.min(1, dt * 1.8));
			}
			_smoothLook.normalize();

			// Dynamic look angle: high = look down more, low = look ahead
			const groundH2 = _heightFn(pos.x, pos.z);
			const heightAboveGround = pos.y - groundH2;
			// Map height 2-12 to lookDown 2-6 (less aggressive at high altitude)
			const dynamicLookDown = 2.0 + Math.min(4.0, Math.max(0, heightAboveGround - 2) * 0.5);

			const lookTarget = pos.clone().addScaledVector(_smoothLook, LOOK_AHEAD);

			// Bias look target toward center of map (looking inward)
			const inwardBias = 0.3;
			lookTarget.x -= lookTarget.x * inwardBias;
			lookTarget.z -= lookTarget.z * inwardBias;
			lookTarget.y -= dynamicLookDown;

			// Sky glance: occasionally tilt up toward sky
			skyGlanceTimer -= dt;
			if (!skyGlanceActive && skyGlanceTimer <= 0) {
				if (Math.random() < SKY_GLANCE_CHANCE) {
					skyGlanceActive = true;
					skyGlanceProgress = 0;
				}
				skyGlanceTimer = 5 + Math.random() * 20;
			}

			if (skyGlanceActive) {
				skyGlanceProgress += dt / SKY_GLANCE_DURATION;
				if (skyGlanceProgress >= 1) {
					skyGlanceActive = false;
				} else {
					// Smooth ease: up then back down
					const glanceT = Math.sin(skyGlanceProgress * Math.PI); // 0→1→0
					// Tilt look target upward (toward sky/sun/moon)
					lookTarget.y += glanceT * 15; // strong upward lift
					// Slight outward look for drama
					lookTarget.x += lookTarget.x * glanceT * 0.3;
					lookTarget.z += lookTarget.z * glanceT * 0.3;
				}
			}

			// Pitch
			const pitch = _tangent.y;
			smoothPitch += (pitch * 0.2 - smoothPitch) * Math.min(1, dt * 1.0);
			lookTarget.y += smoothPitch;

			camera.up.set(0, 1, 0);
			camera.lookAt(lookTarget);

			// Gentle banking
			curve.getTangentAt(t, _tangent);
			const crossTurn = _prevTangent.x * _tangent.z - _prevTangent.z * _tangent.x;
			const clampedRoll = Math.max(-MAX_ROLL, Math.min(MAX_ROLL, -crossTurn * 3.0));
			smoothRoll += (clampedRoll - smoothRoll) * Math.min(1, dt * 0.7);

			camera.getWorldDirection(_fwd);
			_rollQuat.setFromAxisAngle(_fwd, smoothRoll);
			camera.quaternion.premultiply(_rollQuat);

			_prevTangent.copy(_tangent);
		},

		start(camera: THREE.PerspectiveCamera) {
			regeneratePath();
			active = true;
			smoothRoll = 0;
			smoothPitch = 0;
			skyGlanceActive = false;
			skyGlanceTimer = 8 + Math.random() * 15;
			camera.up.set(0, 1, 0);
			camera.position.copy(pathPoints[0]);
			curve.getTangentAt(0, _smoothLook);
			_prevTangent.copy(_smoothLook);
			camera.lookAt(pathPoints[0].clone().addScaledVector(_smoothLook, 5));
		},

		transitionToNewPath(newGrid: number[][], newHeightFn: (x: number, z: number) => number) {
			const currentPos = curve.getPointAt(Math.min(t, 0.999));
			const currentDir = _smoothLook.clone().normalize();
			_grid = newGrid;
			_heightFn = newHeightFn;
			regeneratePath(currentPos, currentDir);
			_prevTangent.copy(currentDir);
		},

		stop() { active = false; },
		dispose() { active = false; }
	};
}
