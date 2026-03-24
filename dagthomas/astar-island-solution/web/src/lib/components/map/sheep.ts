// Sheep flock system — herding behavior with GLB skeletal animation
// 2-3 flocks that graze (idle/walk) and occasionally run + jump together

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkeleton } from 'three/addons/utils/SkeletonUtils.js';
import { mulberry32 } from './prng';

const SHEEP_MODEL_PATH = 'roam/Sheep_2.glb';
const FLOCK_COUNT = 3;
const SHEEP_PER_FLOCK = [5, 4, 3];
const SHEEP_SCALE = 0.076;
const BOUNDARY_RADIUS = 14;
const MODEL_FORWARD_OFFSET = 0; // GLB model faces +Z (forward along movement direction)

// Herd behavior
const SEPARATION_RANGE = 1.4;
const COHESION_RANGE = 4.0;
const SEPARATION_FORCE = 0.6;
const COHESION_FORCE = 0.008;
const ALIGNMENT_FORCE = 0.03;
const WANDER_FORCE = 0.15;
const BOUNDARY_FORCE = 0.4;
const MAX_WALK_SPEED = 0.8;
const MAX_RUN_SPEED = 2.5;
const WALK_TURN_SPEED = 3.0;
const RUN_TURN_SPEED = 4.0;
const IDLE_TURN_SPEED = 1.5;
const GRAZE_DURATION_MIN = 3;
const GRAZE_DURATION_MAX = 8;
const WALK_DURATION_MIN = 4;
const WALK_DURATION_MAX = 10;
const RUN_DURATION_MIN = 2;
const RUN_DURATION_MAX = 5;

type SheepBehavior = 'idle' | 'walk' | 'run';
type AnimName = 'idle' | 'walk' | 'run' | 'jump' | 'eat';

interface SheepState {
	model: THREE.Group;
	mixer: THREE.AnimationMixer;
	actions: Partial<Record<AnimName, THREE.AnimationAction>>;
	currentAnim: AnimName;
	px: number; pz: number;
	vx: number; vz: number;
	behavior: SheepBehavior;
	individualTimer: number; // individual behavior variation
	flockId: number;
	facing: number;
	jumpCooldown: number;
	returningFromJump: boolean;
}

interface Flock {
	centerX: number;
	centerZ: number;
	targetX: number;
	targetZ: number;
	behavior: SheepBehavior;
	behaviorTimer: number;
	wanderAngle: number;
}

export interface SheepSystem {
	update(dt: number, timeOfDay: number): void;
	dispose(): void;
	group: THREE.Group;
}

interface CachedSheep {
	scene: THREE.Group;
	animations: THREE.AnimationClip[];
}

const loader = new GLTFLoader();
let cachedSheep: CachedSheep | null = null;

async function loadSheepModel(): Promise<CachedSheep | null> {
	if (cachedSheep) return cachedSheep;
	try {
		const gltf = await loader.loadAsync(`/${SHEEP_MODEL_PATH}`);
		gltf.scene.traverse((child) => {
			if (child instanceof THREE.Mesh) {
				child.castShadow = true;
				child.receiveShadow = true;
				if (child.material && (child.material as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
					const mat = child.material as THREE.MeshStandardMaterial;
					mat.roughness = 1.0;
					mat.metalness = 0.0;
				}
			}
		});
		cachedSheep = { scene: gltf.scene, animations: gltf.animations || [] };
		return cachedSheep;
	} catch (e) {
		console.warn('Failed to load sheep model:', e);
		return null;
	}
}

function findClip(animations: THREE.AnimationClip[], pattern: RegExp): THREE.AnimationClip | null {
	return animations.find(c => pattern.test(c.name)) || null;
}

/** Switch animation: fade from current to target over duration */
function switchAnim(s: SheepState, target: AnimName, duration = 0.4) {
	if (s.currentAnim === target) return;
	const newAction = s.actions[target];
	if (!newAction) return;

	const oldAction = s.actions[s.currentAnim];

	const speed = 1.0;
	newAction.setEffectiveTimeScale(speed);

	// If same underlying clip (idle reused as walk/run), just change speed
	if (oldAction && oldAction === newAction) {
		s.currentAnim = target;
		return;
	}

	// Different clip — proper crossfade
	newAction.enabled = true;
	newAction.setEffectiveWeight(1);
	newAction.time = 0;
	newAction.play();

	if (oldAction && oldAction !== newAction) {
		oldAction.enabled = true;
		newAction.crossFadeFrom(oldAction, duration, false);
	}

	s.currentAnim = target;
}

/** Shortest signed angle difference, wrapped to [-PI, PI] */
function angleDiff(from: number, to: number): number {
	let d = to - from;
	d = d - Math.floor((d + Math.PI) / (2 * Math.PI)) * 2 * Math.PI;
	return d;
}

export async function createSheepSystem(
	heightFn: (x: number, z: number) => number
): Promise<SheepSystem> {
	const group = new THREE.Group();
	const rng = mulberry32(4242);

	const cached = await loadSheepModel();
	if (!cached || cached.animations.length === 0) {
		console.warn('Sheep: no model or no animations');
		return { group, update() {}, dispose() {} };
	}

	// Match clips — use Idle_Eating as idle (not Idle which is laying down)
	const eatClip = findClip(cached.animations, /Idle_Eating/i) || findClip(cached.animations, /eat/i);
	const idleClip = eatClip || findClip(cached.animations, /idle/i); // prefer eating over laying
	const walkClip = findClip(cached.animations, /walk/i) || idleClip;
	const runClip = findClip(cached.animations, /run/i) || walkClip;
	const jumpClip = findClip(cached.animations, /Jump_Start/i) || findClip(cached.animations, /jump/i);

	// Initialize flocks
	const INIT_BEHAVIORS: SheepBehavior[] = ['walk', 'idle', 'walk'];
	const flocks: Flock[] = [];
	for (let f = 0; f < FLOCK_COUNT; f++) {
		const angle = (f / FLOCK_COUNT) * Math.PI * 2 + rng() * 0.5;
		const radius = 4 + rng() * 6;
		const cx = Math.cos(angle) * radius;
		const cz = Math.sin(angle) * radius;
		const initBehavior = INIT_BEHAVIORS[f % INIT_BEHAVIORS.length];
		const wanderAngle = rng() * Math.PI * 2;
		const wanderDist = 3 + rng() * 4;
		flocks.push({
			centerX: cx, centerZ: cz,
			targetX: initBehavior !== 'idle' ? cx + Math.cos(wanderAngle) * wanderDist : cx,
			targetZ: initBehavior !== 'idle' ? cz + Math.sin(wanderAngle) * wanderDist : cz,
			behavior: initBehavior,
			behaviorTimer: 2 + rng() * 4,
			wanderAngle,
		});
	}

	// Create sheep
	const sheep: SheepState[] = [];
	for (let f = 0; f < FLOCK_COUNT; f++) {
		const count = SHEEP_PER_FLOCK[f] || 3;
		for (let i = 0; i < count; i++) {
			const clone = cloneSkeleton(cached.scene) as THREE.Group;
			clone.scale.setScalar(SHEEP_SCALE);

			const mixer = new THREE.AnimationMixer(clone);
			const actions: Partial<Record<AnimName, THREE.AnimationAction>> = {};

			if (idleClip) {
				const a = mixer.clipAction(idleClip);
				a.setLoop(THREE.LoopRepeat, Infinity);
				actions.idle = a;
			}
			if (walkClip) {
				const a = mixer.clipAction(walkClip);
				a.setLoop(THREE.LoopRepeat, Infinity);
				actions.walk = a;
			}
			if (runClip) {
				const a = mixer.clipAction(runClip);
				a.setLoop(THREE.LoopRepeat, Infinity);
				actions.run = a;
			}
			if (jumpClip) {
				const a = mixer.clipAction(jumpClip);
				a.setLoop(THREE.LoopOnce, 1);
				a.clampWhenFinished = true;
				actions.jump = a;
			}
			if (eatClip) {
				const a = mixer.clipAction(eatClip);
				a.setLoop(THREE.LoopRepeat, Infinity);
				actions.eat = a;
			}

			// Initial position
			const offsetAngle = rng() * Math.PI * 2;
			const offsetDist = 0.5 + rng() * 2.0;
			const px = flocks[f].centerX + Math.cos(offsetAngle) * offsetDist;
			const pz = flocks[f].centerZ + Math.sin(offsetAngle) * offsetDist;
			const facing = rng() * Math.PI * 2;

			clone.position.set(px, heightFn(px, pz), pz);
			clone.rotation.y = facing + MODEL_FORWARD_OFFSET;
			group.add(clone);

			// Start correct animation immediately
			const flockBehavior = flocks[f].behavior;
			const startAnim: AnimName = flockBehavior === 'run' ? 'run' : flockBehavior === 'walk' ? 'walk' : 'idle';
			const actualStart = actions[startAnim] ? startAnim : (actions.idle ? 'idle' : 'walk');
			const startAction = actions[actualStart as AnimName];
			if (startAction) {
				startAction.enabled = true;
				startAction.setEffectiveWeight(1);
				startAction.setEffectiveTimeScale(1);
				startAction.play();
			}

			// Stagger animation timing
			mixer.update(rng() * 3);

			// Listen for jump finished → return to run
			mixer.addEventListener('finished', () => {
				// After jump ends, go back to run
				const s = sheep.find(sh => sh.mixer === mixer);
				if (s && s.returningFromJump) {
					s.returningFromJump = false;
					if (s.behavior === 'run' && s.actions.run) {
						switchAnim(s, 'run', 0.2);
					}
				}
			});

			sheep.push({
				model: clone,
				mixer,
				actions,
				currentAnim: actualStart as AnimName,
				px, pz,
				vx: 0, vz: 0,
				behavior: flockBehavior,
				individualTimer: 1 + rng() * 4, // individual behavior switch timer
				flockId: f,
				facing,
				jumpCooldown: 3 + rng() * 5,
				returningFromJump: false,
			});
		}
	}

	function pickNewFlockBehavior(flock: Flock): void {
		const roll = rng();
		if (roll < 0.45) {
			flock.behavior = 'idle';
			flock.behaviorTimer = GRAZE_DURATION_MIN + rng() * (GRAZE_DURATION_MAX - GRAZE_DURATION_MIN);
		} else if (roll < 0.85) {
			flock.behavior = 'walk';
			flock.behaviorTimer = WALK_DURATION_MIN + rng() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
			flock.wanderAngle += (rng() - 0.5) * Math.PI;
			const wanderDist = 3 + rng() * 5;
			flock.targetX = flock.centerX + Math.cos(flock.wanderAngle) * wanderDist;
			flock.targetZ = flock.centerZ + Math.sin(flock.wanderAngle) * wanderDist;
			const dist = Math.sqrt(flock.targetX ** 2 + flock.targetZ ** 2);
			if (dist > BOUNDARY_RADIUS) {
				flock.targetX *= BOUNDARY_RADIUS / dist;
				flock.targetZ *= BOUNDARY_RADIUS / dist;
			}
		} else {
			flock.behavior = 'run';
			flock.behaviorTimer = RUN_DURATION_MIN + rng() * (RUN_DURATION_MAX - RUN_DURATION_MIN);
			flock.wanderAngle += (rng() - 0.5) * Math.PI * 1.5;
			const wanderDist = 5 + rng() * 6;
			flock.targetX = flock.centerX + Math.cos(flock.wanderAngle) * wanderDist;
			flock.targetZ = flock.centerZ + Math.sin(flock.wanderAngle) * wanderDist;
			const dist = Math.sqrt(flock.targetX ** 2 + flock.targetZ ** 2);
			if (dist > BOUNDARY_RADIUS) {
				flock.targetX *= BOUNDARY_RADIUS / dist;
				flock.targetZ *= BOUNDARY_RADIUS / dist;
			}
		}
	}

	return {
		group,

		update(dt: number, _timeOfDay: number) {
			const clampedDt = Math.min(dt, 0.05);

			// Update flock behaviors
			for (const flock of flocks) {
				flock.behaviorTimer -= clampedDt;
				if (flock.behaviorTimer <= 0) {
					pickNewFlockBehavior(flock);
				}
				const dx = flock.targetX - flock.centerX;
				const dz = flock.targetZ - flock.centerZ;
				const dist = Math.sqrt(dx * dx + dz * dz);
				if (dist > 0.1) {
					const moveSpeed = flock.behavior === 'run' ? 1.5 : flock.behavior === 'walk' ? 0.5 : 0;
					flock.centerX += (dx / dist) * moveSpeed * clampedDt;
					flock.centerZ += (dz / dist) * moveSpeed * clampedDt;
				}
			}

			// Update individual sheep
			for (const s of sheep) {
				const flock = flocks[s.flockId];

				// Individual behavior timer — sheep don't all switch at once
				s.individualTimer -= clampedDt;
				if (s.individualTimer <= 0) {
					if (flock.behavior === 'run') {
						// Run is urgent — all follow quickly
						s.behavior = 'run';
						s.individualTimer = 0.5 + rng() * 1.0;
					} else {
						// Walk/idle: individual variation
						const roll = rng();
						if (flock.behavior === 'walk') {
							// 70% walk with flock, 20% idle (graze), 10% stay current
							s.behavior = roll < 0.7 ? 'walk' : roll < 0.9 ? 'idle' : s.behavior;
						} else {
							// Flock idle: 60% graze, 30% wander-walk, 10% stay
							s.behavior = roll < 0.6 ? 'idle' : roll < 0.9 ? 'walk' : s.behavior;
						}
						s.individualTimer = 2 + rng() * 5;
					}
				}

				const maxSpeed = s.behavior === 'run' ? MAX_RUN_SPEED : MAX_WALK_SPEED;

				// Animation: match behavior (unless mid-jump)
				if (!s.returningFromJump) {
					const targetAnim: AnimName = s.behavior === 'run' ? 'run' : s.behavior === 'walk' ? 'walk' : (eatClip ? 'eat' : 'idle');
					switchAnim(s, targetAnim);
				}

				// Movement physics
				if (s.behavior !== 'idle') {
					let fx = 0, fz = 0;

					// Cohesion: toward flock center
					fx += (flock.centerX - s.px) * COHESION_FORCE;
					fz += (flock.centerZ - s.pz) * COHESION_FORCE;

					// Toward flock target
					const toTargetX = flock.targetX - s.px;
					const toTargetZ = flock.targetZ - s.pz;
					const tDist = Math.sqrt(toTargetX * toTargetX + toTargetZ * toTargetZ);
					if (tDist > 0.5) {
						fx += (toTargetX / tDist) * WANDER_FORCE;
						fz += (toTargetZ / tDist) * WANDER_FORCE;
					}

					// Separation
					for (const other of sheep) {
						if (other === s) continue;
						const dx = s.px - other.px, dz = s.pz - other.pz;
						const d2 = dx * dx + dz * dz;
						if (d2 < SEPARATION_RANGE * SEPARATION_RANGE && d2 > 0.001) {
							const d = Math.sqrt(d2);
							fx += (dx / d) * SEPARATION_FORCE;
							fz += (dz / d) * SEPARATION_FORCE;
						}
					}

					// Alignment with flock mates
					let avgVx = 0, avgVz = 0, neighbors = 0;
					for (const other of sheep) {
						if (other === s || other.flockId !== s.flockId) continue;
						const dx = s.px - other.px, dz = s.pz - other.pz;
						if (dx * dx + dz * dz < COHESION_RANGE * COHESION_RANGE) {
							avgVx += other.vx;
							avgVz += other.vz;
							neighbors++;
						}
					}
					if (neighbors > 0) {
						fx += (avgVx / neighbors - s.vx) * ALIGNMENT_FORCE;
						fz += (avgVz / neighbors - s.vz) * ALIGNMENT_FORCE;
					}

					// Boundary
					const bDist = Math.sqrt(s.px * s.px + s.pz * s.pz);
					if (bDist > BOUNDARY_RADIUS - 2) {
						fx -= (s.px / bDist) * BOUNDARY_FORCE;
						fz -= (s.pz / bDist) * BOUNDARY_FORCE;
					}

					// Apply forces
					s.vx += fx * clampedDt * 3;
					s.vz += fz * clampedDt * 3;

					// Damping
					s.vx *= 0.95;
					s.vz *= 0.95;

					// Speed clamp
					const speed = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
					if (speed > maxSpeed) {
						s.vx = (s.vx / speed) * maxSpeed;
						s.vz = (s.vz / speed) * maxSpeed;
					}

					// Move
					s.px += s.vx * clampedDt;
					s.pz += s.vz * clampedDt;

					// Smooth facing toward movement direction
					if (speed > 0.05) {
						const targetFacing = Math.atan2(s.vx, s.vz);
						const turnSpeed = s.behavior === 'run' ? RUN_TURN_SPEED : WALK_TURN_SPEED;
						const diff = angleDiff(s.facing, targetFacing);
						s.facing += diff * Math.min(1, clampedDt * turnSpeed);
					}
				} else {
					// Idle: decelerate
					s.vx *= 0.9;
					s.vz *= 0.9;
				}

				// Random jump during run
				if (s.behavior === 'run' && s.actions.jump && !s.returningFromJump) {
					s.jumpCooldown -= clampedDt;
					if (s.jumpCooldown <= 0) {
						s.returningFromJump = true;
						switchAnim(s, 'jump', 0.15);
						s.jumpCooldown = 3 + rng() * 6;
					}
				}

				// Apply position and rotation to model
				const groundY = heightFn(s.px, s.pz);
				s.model.position.set(s.px, Math.max(groundY, 0.0), s.pz);
				s.model.rotation.y = s.facing + MODEL_FORWARD_OFFSET;

				// Always update mixer (animations need ticking even when idle)
				s.mixer.update(clampedDt);
			}
		},

		dispose() {
			for (const s of sheep) {
				s.mixer.stopAllAction();
			}
			group.traverse((obj) => {
				if (obj instanceof THREE.Mesh) {
					obj.geometry?.dispose();
					const mat = obj.material;
					if (mat) {
						const mats = Array.isArray(mat) ? mat : [mat];
						for (const m of mats) m.dispose();
					}
				}
			});
		},
	};
}
