// Sheep flock system — herding behavior with GLB skeletal animation
// 2-3 flocks that graze (idle/walk) and occasionally run + jump together

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkeleton } from 'three/addons/utils/SkeletonUtils.js';
import { mulberry32 } from './prng';

const SHEEP_MODEL_PATH = 'roam/Sheep.glb';
const FLOCK_COUNT = 3;
const SHEEP_PER_FLOCK = [5, 4, 3]; // varied flock sizes
const SHEEP_SCALE = 0.04;
const BOUNDARY_RADIUS = 14;

// Herd behavior constants
const SEPARATION_RANGE = 1.4;
const COHESION_RANGE = 4.0;
const SEPARATION_FORCE = 0.6;
const COHESION_FORCE = 0.008;
const ALIGNMENT_FORCE = 0.03;
const WANDER_FORCE = 0.15;
const BOUNDARY_FORCE = 0.4;
const MAX_WALK_SPEED = 0.8;
const MAX_RUN_SPEED = 2.5;
const GRAZE_DURATION_MIN = 3;
const GRAZE_DURATION_MAX = 8;
const WALK_DURATION_MIN = 4;
const WALK_DURATION_MAX = 10;
const RUN_DURATION_MIN = 2;
const RUN_DURATION_MAX = 5;

type SheepBehavior = 'idle' | 'walk' | 'run';

interface SheepState {
	model: THREE.Group;
	mixer: THREE.AnimationMixer;
	actions: Record<string, THREE.AnimationAction>;
	currentAnim: string;
	px: number; pz: number;
	vx: number; vz: number;
	behavior: SheepBehavior;
	behaviorTimer: number;
	flockId: number;
	facing: number; // rotation Y
	jumpCooldown: number;
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
				// Make matte/woolly looking
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

function crossFadeTo(sheep: SheepState, animName: string, duration = 0.3) {
	if (sheep.currentAnim === animName) return;
	const newAction = sheep.actions[animName];
	const oldAction = sheep.actions[sheep.currentAnim];
	if (!newAction) return;

	newAction.reset();
	newAction.setEffectiveWeight(1);
	newAction.play();

	if (oldAction) {
		oldAction.crossFadeTo(newAction, duration, true);
	}

	sheep.currentAnim = animName;
}

export async function createSheepSystem(
	heightFn: (x: number, z: number) => number
): Promise<SheepSystem> {
	const group = new THREE.Group();
	const rng = mulberry32(4242);

	const cached = await loadSheepModel();
	if (!cached || cached.animations.length === 0) {
		return { group, update() {}, dispose() { /* no-op */ } };
	}

	// Find animation clips
	const idleClip = findClip(cached.animations, /idle/i);
	const walkClip = findClip(cached.animations, /walk/i);
	const runClip = findClip(cached.animations, /run/i);
	const jumpClip = findClip(cached.animations, /jump/i);

	// Initialize flocks with spread-out positions — stagger initial behaviors
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
			const actions: Record<string, THREE.AnimationAction> = {};

			if (idleClip) { actions.idle = mixer.clipAction(idleClip); actions.idle.setEffectiveWeight(0); }
			if (walkClip) { actions.walk = mixer.clipAction(walkClip); actions.walk.setEffectiveWeight(0); }
			if (runClip) { actions.run = mixer.clipAction(runClip); actions.run.setEffectiveWeight(0); }
			if (jumpClip) {
				actions.jump = mixer.clipAction(jumpClip);
				actions.jump.setEffectiveWeight(0);
				actions.jump.setLoop(THREE.LoopOnce, 1);
				actions.jump.clampWhenFinished = true;
			}

			// Spread around flock center
			const offsetAngle = rng() * Math.PI * 2;
			const offsetDist = 0.5 + rng() * 2.0;
			const px = flocks[f].centerX + Math.cos(offsetAngle) * offsetDist;
			const pz = flocks[f].centerZ + Math.sin(offsetAngle) * offsetDist;

			clone.position.set(px, heightFn(px, pz), pz);
			group.add(clone);

			// Start with flock's behavior
			const flockBehavior = flocks[f].behavior;
			const startAnim = flockBehavior === 'run' ? 'run' : flockBehavior === 'walk' ? 'walk' : 'idle';
			const actualStart = actions[startAnim] ? startAnim : (idleClip ? 'idle' : 'walk');
			if (actions[actualStart]) {
				actions[actualStart].reset().setEffectiveWeight(1).play();
			}

			// Vary animation timing so they're not in sync
			mixer.update(rng() * 2);

			sheep.push({
				model: clone,
				mixer,
				actions,
				currentAnim: actualStart,
				px, pz,
				vx: 0, vz: 0,
				behavior: flockBehavior,
				behaviorTimer: 0, // immediately active
				flockId: f,
				facing: rng() * Math.PI * 2,
				jumpCooldown: 3 + rng() * 5,
			});
		}
	}

	function pickNewFlockBehavior(flock: Flock): void {
		const roll = rng();
		if (roll < 0.45) {
			// Graze (idle)
			flock.behavior = 'idle';
			flock.behaviorTimer = GRAZE_DURATION_MIN + rng() * (GRAZE_DURATION_MAX - GRAZE_DURATION_MIN);
		} else if (roll < 0.85) {
			// Walk to new spot
			flock.behavior = 'walk';
			flock.behaviorTimer = WALK_DURATION_MIN + rng() * (WALK_DURATION_MAX - WALK_DURATION_MIN);
			flock.wanderAngle += (rng() - 0.5) * Math.PI;
			const wanderDist = 3 + rng() * 5;
			flock.targetX = flock.centerX + Math.cos(flock.wanderAngle) * wanderDist;
			flock.targetZ = flock.centerZ + Math.sin(flock.wanderAngle) * wanderDist;
			// Clamp to boundary
			const dist = Math.sqrt(flock.targetX ** 2 + flock.targetZ ** 2);
			if (dist > BOUNDARY_RADIUS) {
				flock.targetX *= BOUNDARY_RADIUS / dist;
				flock.targetZ *= BOUNDARY_RADIUS / dist;
			}
		} else {
			// Run (spooked or playful)
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

		update(dt: number, timeOfDay: number) {
			const clampedDt = Math.min(dt, 0.05);

			// Update flock behaviors
			for (const flock of flocks) {
				flock.behaviorTimer -= clampedDt;
				if (flock.behaviorTimer <= 0) {
					pickNewFlockBehavior(flock);
				}
				// Move flock center toward target
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
				const maxSpeed = flock.behavior === 'run' ? MAX_RUN_SPEED : MAX_WALK_SPEED;

				// Transition individual behavior to match flock
				if (s.behavior !== flock.behavior) {
					s.behavior = flock.behavior;
					s.behaviorTimer = 0.2 + rng() * 0.8; // stagger transitions
				}

				if (s.behaviorTimer > 0) {
					s.behaviorTimer -= clampedDt;
				} else {
					// Apply animation based on behavior
					const targetAnim = s.behavior === 'run' ? 'run' : s.behavior === 'walk' ? 'walk' : 'idle';
					crossFadeTo(s, targetAnim);
				}

				if (s.behavior !== 'idle') {
					// Herding forces
					let fx = 0, fz = 0;

					// Cohesion: move toward flock center
					const toCenterX = flock.centerX - s.px;
					const toCenterZ = flock.centerZ - s.pz;
					fx += toCenterX * COHESION_FORCE;
					fz += toCenterZ * COHESION_FORCE;

					// Move toward flock target
					const toTargetX = flock.targetX - s.px;
					const toTargetZ = flock.targetZ - s.pz;
					const tDist = Math.sqrt(toTargetX ** 2 + toTargetZ ** 2);
					if (tDist > 0.5) {
						fx += (toTargetX / tDist) * WANDER_FORCE;
						fz += (toTargetZ / tDist) * WANDER_FORCE;
					}

					// Separation from other sheep
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

					// Alignment with nearby flock members
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

					// Boundary avoidance
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

					// Face movement direction
					if (speed > 0.1) {
						const targetFacing = Math.atan2(s.vx, s.vz);
						// Smooth rotation
						let diff = targetFacing - s.facing;
						while (diff > Math.PI) diff -= Math.PI * 2;
						while (diff < -Math.PI) diff += Math.PI * 2;
						s.facing += diff * Math.min(1, clampedDt * 5);
					}
				} else {
					// Idle: slow to stop
					s.vx *= 0.9;
					s.vz *= 0.9;
				}

				// Random jump during run
				if (s.behavior === 'run' && s.actions.jump) {
					s.jumpCooldown -= clampedDt;
					if (s.jumpCooldown <= 0) {
						// Play jump once, then return to run
						const jumpAction = s.actions.jump;
						jumpAction.reset().setEffectiveWeight(1).play();
						const runAction = s.actions.run;
						if (runAction) {
							runAction.crossFadeTo(jumpAction, 0.15, true);
						}
						s.currentAnim = 'jump';

						// Return to run after jump finishes
						const jumpDuration = s.actions.jump.getClip().duration;
						setTimeout(() => {
							if (s.behavior === 'run') {
								crossFadeTo(s, 'run', 0.2);
							}
						}, (jumpDuration - 0.2) * 1000);

						s.jumpCooldown = 3 + rng() * 6;
					}
				}

				// Update position
				const y = heightFn(s.px, s.pz);
				s.model.position.set(s.px, y, s.pz);
				s.model.rotation.y = s.facing;

				// Update animation mixer
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
