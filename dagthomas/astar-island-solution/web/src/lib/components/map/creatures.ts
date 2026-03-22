/**
 * Living world: GLB characters + animals with walk animations.
 *
 * - Humans (Cube Guy/Woman) walk between settlements (FP mode)
 * - Raiding parties use Goblin, Zombie, Yeti, Demon models
 * - Animals roam biomes: deer/fox/rabbit in forests, cows/sheep on plains,
 *   bears on mountains, birds/chickens everywhere
 *
 * Characters from /static/chars, animals from /static/animals
 * Models sourced from https://poly.pizza/bundle
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkeleton } from 'three/addons/utils/SkeletonUtils.js';
import { mulberry32 } from './prng';
import type { Cluster } from './clusters';
import { TerrainCode } from '$lib/types';
import { findClusters } from './clusters';

export interface CreatureSystem {
	group: THREE.Group;
	update(dt: number, timeOfDay: number): void;
	updateCulling(camera: THREE.PerspectiveCamera): void;
	dispose(): void;
}

// --- Types ---

interface Creature {
	model: THREE.Group;
	mixer: THREE.AnimationMixer | null;
	from: THREE.Vector3;
	to: THREE.Vector3;
	progress: number;
	speed: number;
	/** For war parties: which raid group */
	raidId?: number;
	/** True for biome animals (wander, don't walk to settlements) */
	isAnimal?: boolean;
}

interface RaidGroup {
	creatures: Creature[];
	active: boolean;
	cooldown: number;
}

interface CachedModel {
	scene: THREE.Group;
	animations: THREE.AnimationClip[];
}

// --- Config ---

const HUMAN_FILES = ['chars/Cube Guy Character.glb', 'chars/Cube Woman Character.glb'];
const RAIDER_FILES = ['chars/Goblin.glb', 'chars/Zombie.glb', 'chars/Yeti.glb', 'chars/Demon.glb'];

const HUMAN_SCALE: [number, number] = [0.15, 0.20];
const RAIDER_SCALE: [number, number] = [0.16, 0.22];

// Animals per biome
const BIOME_ANIMALS: Record<number, { files: string[]; count: number; scale: [number, number] }> = {
	// Forest
	4: {
		files: ['animals/Deer.glb', 'animals/Deer-bLT2gJHPPt.glb', 'animals/Fox.glb', 'animals/Rabbit.glb', 'animals/Bird.glb', 'animals/Frog.glb'],
		count: 3,
		scale: [0.10, 0.18]
	},
	// Plains
	11: {
		files: ['animals/Cow.glb', 'animals/Sheep.glb', 'animals/Horse.glb', 'animals/Chicken.glb', 'animals/Chick.glb', 'animals/Dog.glb'],
		count: 4,
		scale: [0.10, 0.18]
	},
	// Mountain
	5: {
		files: ['animals/Bear.glb', 'animals/Bizon.glb', 'animals/Lizard.glb'],
		count: 1,
		scale: [0.12, 0.20]
	},
	// Empty/sandy
	0: {
		files: ['animals/Cat.glb', 'animals/Dog-9bqPCxOyrk.glb', 'animals/Corgi.glb'],
		count: 1,
		scale: [0.08, 0.14]
	},
	// Settlement
	1: {
		files: ['animals/Dog.glb', 'animals/Cat.glb', 'animals/Chicken.glb', 'animals/Duck.glb', 'animals/Corgi.glb', 'animals/Beagle.glb'],
		count: 2,
		scale: [0.08, 0.14]
	},
	// Ruin
	3: {
		files: ['animals/Fox.glb', 'animals/Penguin.glb', 'animals/Bird-h5IzAUdltz.glb'],
		count: 1,
		scale: [0.10, 0.16]
	}
};

// --- Loader ---

const loader = new GLTFLoader();
const cache = new Map<string, CachedModel | null>();

async function loadCreatureModel(file: string): Promise<CachedModel | null> {
	if (cache.has(file)) return cache.get(file)!;
	try {
		const gltf = await loader.loadAsync(`/${file}`);
		const model = gltf.scene;
		model.traverse((child) => {
			if (child instanceof THREE.Mesh) {
				child.castShadow = true;
				child.receiveShadow = true;
			}
		});
		const cached: CachedModel = { scene: model, animations: gltf.animations || [] };
		cache.set(file, cached);
		return cached;
	} catch {
		cache.set(file, null);
		return null;
	}
}

/** Clone a model and optionally set up animation mixer */
function cloneAnimated(cached: CachedModel, scale: number, pos: THREE.Vector3): { model: THREE.Group; mixer: THREE.AnimationMixer | null } {
	const clone = cloneSkeleton(cached.scene) as THREE.Group;
	clone.scale.setScalar(scale);
	clone.position.copy(pos);

	let mixer: THREE.AnimationMixer | null = null;
	if (cached.animations.length > 0) {
		try {
			mixer = new THREE.AnimationMixer(clone);
			const walkClip = cached.animations.find(c =>
				/walk|run|move|locomotion/i.test(c.name)
			) || cached.animations[0];
			const action = mixer.clipAction(walkClip);
			action.play();
		} catch {
			mixer = null;
		}
	}

	return { model: clone, mixer };
}

// --- Main ---

export async function createCreatures(
	scene: THREE.Scene,
	grid: number[][],
	heightFn: (x: number, z: number) => number,
	settlementClusters: Cluster[]
): Promise<CreatureSystem> {
	const rows = grid.length, cols = grid[0].length;
	const ox = -cols / 2, oz = -rows / 2;
	const rng = mulberry32(77777);
	const group = new THREE.Group();

	// Preload all character/animal models
	const allFiles = [...HUMAN_FILES, ...RAIDER_FILES];
	for (const def of Object.values(BIOME_ANIMALS)) {
		for (const f of def.files) allFiles.push(f);
	}
	await Promise.allSettled([...new Set(allFiles)].map(f => loadCreatureModel(f)));

	// Settlement positions
	const settPositions = settlementClusters.map(c =>
		new THREE.Vector3(c.centerX + ox + 0.5, heightFn(c.centerX + ox + 0.5, c.centerY + oz + 0.5) + 0.01, c.centerY + oz + 0.5)
	);

	const creatures: Creature[] = [];
	const raids: RaidGroup[] = [];
	let raidCooldown = 8 + rng() * 15;

	// --- Spawn human wanderers between settlements ---
	if (settPositions.length >= 2) {
		const humanModels: CachedModel[] = [];
		for (const f of HUMAN_FILES) {
			const m = cache.get(f);
			if (m) humanModels.push(m);
		}

		const humanCount = Math.min(20, settPositions.length * 4);
		for (let i = 0; i < humanCount && humanModels.length > 0; i++) {
			const fromIdx = Math.floor(rng() * settPositions.length);
			let toIdx = Math.floor(rng() * settPositions.length);
			if (toIdx === fromIdx) toIdx = (toIdx + 1) % settPositions.length;

			const baseCached = humanModels[Math.floor(rng() * humanModels.length)];
			const scale = HUMAN_SCALE[0] + rng() * (HUMAN_SCALE[1] - HUMAN_SCALE[0]);
			const { model, mixer } = cloneAnimated(baseCached, scale, settPositions[fromIdx].clone());
			model.rotation.y = rng() * Math.PI * 2;
			group.add(model);

			// Vary animation speed slightly per creature
			if (mixer) mixer.timeScale = 0.8 + rng() * 0.4;

			creatures.push({
				model,
				mixer,
				from: settPositions[fromIdx].clone(),
				to: settPositions[toIdx].clone(),
				progress: rng(),
				speed: 0.3 + rng() * 0.4,
			});
		}
	}

	// --- Spawn animals per biome ---
	const clusters = findClusters(grid);

	// Supplement: build clusters for terrain types not in findClusters (0=EMPTY, 11=PLAINS)
	const extraTypes = [0, 11];
	const visited = Array.from({ length: rows }, () => new Array(cols).fill(false));
	for (const cl of clusters) {
		for (const cell of cl.cells) visited[cell.y][cell.x] = true;
	}
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			if (visited[y][x]) continue;
			const terrain = grid[y][x];
			if (!extraTypes.includes(terrain)) { visited[y][x] = true; continue; }
			const cells: { x: number; y: number }[] = [];
			const queue: { x: number; y: number }[] = [{ x, y }];
			visited[y][x] = true;
			while (queue.length > 0) {
				const cell = queue.shift()!;
				cells.push(cell);
				for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
					const nx = cell.x + dx, ny = cell.y + dy;
					if (nx >= 0 && nx < cols && ny >= 0 && ny < rows && !visited[ny][nx] && grid[ny][nx] === terrain) {
						visited[ny][nx] = true;
						queue.push({ x: nx, y: ny });
					}
				}
			}
			if (cells.length >= 2) {
				const centerX = cells.reduce((s, c) => s + c.x, 0) / cells.length;
				const centerY = cells.reduce((s, c) => s + c.y, 0) / cells.length;
				clusters.push({ terrainType: terrain, cells, centerX, centerY, size: cells.length });
			}
		}
	}

	for (const cluster of clusters) {
		const biomeDef = BIOME_ANIMALS[cluster.terrainType];
		if (!biomeDef) continue;

		const animalModels: CachedModel[] = [];
		for (const f of biomeDef.files) {
			const m = cache.get(f);
			if (m) animalModels.push(m);
		}
		if (animalModels.length === 0) continue;

		// Spawn count based on cluster size
		const count = Math.min(biomeDef.count, Math.ceil(cluster.size / 2));
		for (let i = 0; i < count; i++) {
			const cell = cluster.cells[Math.floor(rng() * cluster.cells.length)];
			const px = cell.x + ox + 0.2 + rng() * 0.6;
			const pz = cell.y + oz + 0.2 + rng() * 0.6;
			const py = heightFn(px, pz) + 0.01;

			const baseCached = animalModels[Math.floor(rng() * animalModels.length)];
			const scale = biomeDef.scale[0] + rng() * (biomeDef.scale[1] - biomeDef.scale[0]);
			const { model, mixer } = cloneAnimated(baseCached, scale, new THREE.Vector3(px, py, pz));
			model.rotation.y = rng() * Math.PI * 2;
			group.add(model);

			if (mixer) mixer.timeScale = 0.6 + rng() * 0.4;

			// Animals wander within their cluster
			const cell2 = cluster.cells[Math.floor(rng() * cluster.cells.length)];
			const tx = cell2.x + ox + 0.2 + rng() * 0.6;
			const tz = cell2.y + oz + 0.2 + rng() * 0.6;

			creatures.push({
				model,
				mixer,
				from: new THREE.Vector3(px, py, pz),
				to: new THREE.Vector3(tx, heightFn(tx, tz) + 0.01, tz),
				progress: rng(),
				speed: 0.15 + rng() * 0.25,
				isAnimal: true,
			});
		}
	}

	// --- Raider spawn function ---
	function spawnRaid() {
		if (settPositions.length < 2) return;
		const raiderModels: CachedModel[] = [];
		for (const f of RAIDER_FILES) {
			const m = cache.get(f);
			if (m) raiderModels.push(m);
		}
		if (raiderModels.length === 0) return;

		const fromIdx = Math.floor(Math.random() * settPositions.length);
		let toIdx = Math.floor(Math.random() * settPositions.length);
		if (toIdx === fromIdx) toIdx = (toIdx + 1) % settPositions.length;

		const origin = settPositions[fromIdx];
		const target = settPositions[toIdx];
		// Approach from outside the map edge toward target
		const dx = target.x - origin.x;
		const dz = target.z - origin.z;
		const len = Math.sqrt(dx * dx + dz * dz);
		const spawnPos = origin.clone().addScaledVector(new THREE.Vector3(-dx/len, 0, -dz/len), 4);
		spawnPos.y = heightFn(spawnPos.x, spawnPos.z) + 0.01;

		// Pick a single raider type for this raid (all same species)
		const baseCached = raiderModels[Math.floor(Math.random() * raiderModels.length)];
		const raidCreatures: Creature[] = [];
		const count = 3 + Math.floor(Math.random() * 5);

		for (let i = 0; i < count; i++) {
			const offset = new THREE.Vector3((Math.random() - 0.5) * 0.6, 0, (Math.random() - 0.5) * 0.6);
			const scale = RAIDER_SCALE[0] + Math.random() * (RAIDER_SCALE[1] - RAIDER_SCALE[0]);
			const { model, mixer } = cloneAnimated(baseCached, scale, spawnPos.clone().add(offset));
			model.rotation.y = Math.atan2(dx, dz);
			group.add(model);

			// Raiders move faster
			if (mixer) mixer.timeScale = 1.2 + Math.random() * 0.4;

			const c: Creature = {
				model,
				mixer,
				from: spawnPos.clone().add(offset),
				to: target.clone().add(offset),
				progress: 0,
				speed: 0.6 + Math.random() * 0.3,
				raidId: raids.length
			};
			creatures.push(c);
			raidCreatures.push(c);
		}

		raids.push({ creatures: raidCreatures, active: true, cooldown: 0 });
	}

	scene.add(group);

	// Culling state
	const _frustum = new THREE.Frustum();
	const _projMatrix = new THREE.Matrix4();
	const _sphere = new THREE.Sphere();
	let cullFrame = 0;

	return {
		group,

		update(dt: number, _timeOfDay: number) {
			// Raid spawning
			raidCooldown -= dt;
			if (raidCooldown <= 0 && raids.filter(r => r.active).length < 2) {
				spawnRaid();
				raidCooldown = 20 + Math.random() * 30;
			}

			// Update all creatures
			for (const c of creatures) {
				// Advance animation mixer
				if (c.mixer) c.mixer.update(dt);

				c.progress += c.speed * dt * 0.04;

				if (c.progress >= 1) {
					if (c.raidId !== undefined) {
						// Raider arrived — despawn after cooldown
						const raid = raids[c.raidId];
						if (raid) raid.active = false;
						c.model.visible = false;
						if (c.mixer) c.mixer.stopAllAction();
						continue;
					}

					// Pick new destination
					c.from.copy(c.to);
					if (settPositions.length >= 2 && !c.isAnimal) {
						// Human: go to another settlement
						const idx = Math.floor(Math.random() * settPositions.length);
						c.to.copy(settPositions[idx]);
					} else {
						// Animal: wander nearby
						c.to.set(
							c.from.x + (Math.random() - 0.5) * 2,
							0,
							c.from.z + (Math.random() - 0.5) * 2
						);
						c.to.y = heightFn(c.to.x, c.to.z) + 0.01;
					}
					c.progress = 0;
				}

				// Lerp position
				c.model.position.lerpVectors(c.from, c.to, c.progress);

				// Walking bob for models without embedded animations
				if (!c.mixer) {
					const bobPhase = performance.now() / 1000 * 6 * c.speed;
					c.model.position.y += Math.abs(Math.sin(bobPhase)) * 0.015;
					c.model.rotation.z = Math.sin(bobPhase * 0.5) * 0.03;
				}

				// Face movement direction
				const dx = c.to.x - c.from.x;
				const dz = c.to.z - c.from.z;
				if (dx * dx + dz * dz > 0.01) {
					c.model.rotation.y = Math.atan2(dx, dz);
				}
			}

			// Cleanup finished raids
			for (let i = raids.length - 1; i >= 0; i--) {
				const raid = raids[i];
				if (!raid.active) {
					raid.cooldown += dt;
					if (raid.cooldown > 8) {
						// Remove raider models
						for (const c of raid.creatures) {
							if (c.mixer) c.mixer.stopAllAction();
							group.remove(c.model);
						}
						raids.splice(i, 1);
					}
				}
			}
		},

		updateCulling(camera: THREE.PerspectiveCamera) {
			cullFrame++;
			if (cullFrame % 4 !== 0) return;

			_projMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
			_frustum.setFromProjectionMatrix(_projMatrix);
			const cp = camera.position;
			const DIST_SQ = 60 * 60;

			for (const child of group.children) {
				const dx = child.position.x - cp.x;
				const dz = child.position.z - cp.z;
				if (dx * dx + dz * dz > DIST_SQ) {
					child.visible = false;
					continue;
				}
				_sphere.center.copy(child.position);
				_sphere.radius = 0.3;
				child.visible = _frustum.intersectsSphere(_sphere);
			}
		},

		dispose() {
			// Stop all animations
			for (const c of creatures) {
				if (c.mixer) c.mixer.stopAllAction();
			}
			group.traverse((child) => {
				if (child instanceof THREE.Mesh) {
					child.geometry.dispose();
					if (Array.isArray(child.material)) {
						child.material.forEach(m => m.dispose());
					} else {
						child.material.dispose();
					}
				}
			});
			scene.remove(group);
		}
	};
}
