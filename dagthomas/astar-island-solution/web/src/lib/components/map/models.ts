// GLB model loader — preloads and caches models, returns clones for placement

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
const cache = new Map<string, THREE.Group>();
const pending = new Map<string, Promise<THREE.Group>>();

// Model catalog — maps logical name → file path + default scale
export const MODEL_CATALOG = {
	// Settlements — RTS style (1.5× scale)
	hut: { path: '/models/rts/Hut.glb', scale: 0.52 },
	hut2: { path: '/models/rts/Hut-4MJWbyd6vw.glb', scale: 0.52 },
	house: { path: '/models/rts/House.glb', scale: 0.45 },
	house2: { path: '/models/rts/House-oJJIRwv6Bo.glb', scale: 0.42 },
	townCenter: { path: '/models/rts/Town Center.glb', scale: 0.52 },
	shack: { path: '/models/rts/Shack.glb', scale: 0.45 },
	// Settlements — playfield style (1.5× scale)
	pfHouse: { path: '/models/playfield/House.glb', scale: 0.33 },
	pfHouse2: { path: '/models/playfield/House-k6tP5nFUd2.glb', scale: 0.33 },
	pfHouse3: { path: '/models/playfield/House-nihGGju7DW.glb', scale: 0.33 },
	pfHouse4: { path: '/models/playfield/House-oJJIRwv6Bo.glb', scale: 0.33 },
	pfHouse5: { path: '/models/playfield/House-RSwoYSLblu.glb', scale: 0.33 },
	pfHut: { path: '/models/playfield/Hut.glb', scale: 0.42 },
	pfFarm: { path: '/models/playfield/Farm.glb', scale: 0.38 },
	pfSmallFarm: { path: '/models/playfield/Small Farm.glb', scale: 0.42 },
	pfCrops: { path: '/models/playfield/Crops.glb', scale: 0.45 },
	pfMarket: { path: '/models/playfield/Village Market.glb', scale: 0.42 },
	pfFortress: { path: '/models/playfield/Wooden Fortress.glb', scale: 0.33 },
	// Nature — playfield
	pfMountain: { path: '/models/playfield/Mountain.glb', scale: 0.50 },
	pfMountain2: { path: '/models/playfield/Mountain-XY4ej3Zg3I.glb', scale: 0.50 },
	pfMountainGroup: { path: '/models/playfield/Mountain Group.glb', scale: 0.45 },
	pfMountains: { path: '/models/playfield/Mountains.glb', scale: 0.40 },
	pfPineTrees: { path: '/models/playfield/Pine Trees.glb', scale: 0.35 },
	pfTrees: { path: '/models/playfield/Trees.glb', scale: 0.35 },
	pfTreesCut: { path: '/models/playfield/Trees cut.glb', scale: 0.30 },
	pfRock: { path: '/models/playfield/Rock.glb', scale: 0.35 },
	pfRock2: { path: '/models/playfield/Rock-JmFMh7ztL9.glb', scale: 0.35 },
	pfRock3: { path: '/models/playfield/Rock-RtLRqYjfMs.glb', scale: 0.35 },
	pfRocks: { path: '/models/playfield/Rocks.glb', scale: 0.30 },
	pfGoldRocks: { path: '/models/playfield/Gold rocks.glb', scale: 0.30 },
	pfPort: { path: '/models/playfield/Port.glb', scale: 0.33 },
	// Nature — RTS
	pineTrees: { path: '/models/rts/Pine Trees.glb', scale: 0.40 },
	mountain: { path: '/models/rts/Mountain.glb', scale: 0.70 },
	mountainGroup: { path: '/models/rts/Mountain Group.glb', scale: 0.55 },
	mountains: { path: '/models/rts/Mountains.glb', scale: 0.50 },
	rock: { path: '/models/rts/Rock.glb', scale: 0.45 },
	rocks: { path: '/models/rts/Rocks.glb', scale: 0.40 },
	logs: { path: '/models/rts/Logs.glb', scale: 0.30 },
	// Port/water (1.5× scale)
	dock: { path: '/models/rts/Dock.glb', scale: 0.52 },
	port: { path: '/models/rts/Port.glb', scale: 0.38 },
	// Fortification (1.5× scale)
	fortress: { path: '/models/rts/Fortress.glb', scale: 0.45 },
	watchTower: { path: '/models/rts/Small Watch Tower.glb', scale: 0.45 },
	woodenWall: { path: '/models/rts/Wooden Wall.glb', scale: 0.30 },
	stoneWall: { path: '/models/rts/Stone Wall.glb', scale: 0.30 },
	farm: { path: '/models/rts/Farm.glb', scale: 0.38 },
} as const;

export type ModelName = keyof typeof MODEL_CATALOG;

function loadModel(path: string): Promise<THREE.Group> {
	if (cache.has(path)) return Promise.resolve(cache.get(path)!);
	if (pending.has(path)) return pending.get(path)!;

	const p = new Promise<THREE.Group>((resolve, reject) => {
		loader.load(
			path,
			(gltf) => {
				const model = gltf.scene;
				// Enable shadows on all meshes
				model.traverse((child) => {
					if (child instanceof THREE.Mesh) {
						child.castShadow = true;
						child.receiveShadow = true;
					}
				});
				cache.set(path, model);
				pending.delete(path);
				resolve(model);
			},
			undefined,
			(err) => {
				pending.delete(path);
				reject(err);
			}
		);
	});

	pending.set(path, p);
	return p;
}

/** Preload all models in the catalog. Returns when all are ready. */
export async function preloadModels(): Promise<void> {
	const paths = new Set(Object.values(MODEL_CATALOG).map(m => m.path));
	await Promise.allSettled([...paths].map(p => loadModel(p)));
}

// Cache bounding box bottom Y per model path (computed once)
const bottomYCache = new Map<string, number>();

function getModelBottomY(model: THREE.Group, path: string): number {
	if (bottomYCache.has(path)) return bottomYCache.get(path)!;
	const box = new THREE.Box3().setFromObject(model);
	const bottomY = box.min.y;
	bottomYCache.set(path, bottomY);
	return bottomY;
}

/** Get a clone of a model, positioned and scaled. Lifts model so base sits on ground. */
export function placeModel(
	name: ModelName,
	position: THREE.Vector3,
	rotation = 0,
	scaleOverride?: number
): THREE.Group | null {
	const info = MODEL_CATALOG[name];
	const original = cache.get(info.path);
	if (!original) return null;

	const clone = original.clone();
	const s = scaleOverride ?? info.scale;
	clone.scale.set(s, s, s);
	clone.position.copy(position);
	clone.rotation.y = rotation;

	// Lift model so its bottom sits on the ground, not clipping through
	const bottomY = getModelBottomY(original, info.path);
	clone.position.y -= bottomY * s;

	// Ensure shadows are enabled on cloned meshes
	clone.traverse((child) => {
		if (child instanceof THREE.Mesh) {
			child.castShadow = true;
			child.receiveShadow = true;
		}
	});

	return clone;
}
