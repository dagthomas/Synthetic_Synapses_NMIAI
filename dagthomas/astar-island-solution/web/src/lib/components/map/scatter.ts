/**
 * Environmental model scatter — loads poly.pizza GLB models and distributes
 * them across the terrain for immersive first-person / flythrough viewing.
 *
 * Models sourced from https://poly.pizza/bundle
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mulberry32 } from './prng';

export interface ScatterSystem {
	/** Group containing all scattered objects */
	group: THREE.Group;
	/** Frustum cull objects each frame for performance */
	updateCulling(camera: THREE.PerspectiveCamera): void;
	dispose(): void;
}

// Model definitions per terrain type
interface ScatterDef {
	files: string[];
	density: number;     // instances per cell
	scale: [number, number]; // min, max scale
	yOffset: number;     // height offset above terrain
}

const SCATTER_DEFS: Record<number, ScatterDef> = {
	// Forest (4): dense trees, ferns, mushrooms
	4: {
		files: [
			'Pine.glb', 'Pine-699sFuLCN2.glb', 'Pine-79gmlLnweB.glb', 'Pine-rfnxJv0Rqa.glb',
			'Tree.glb', 'Tree-aVOxaHRPWe.glb', 'Tree-QVOop92WmG.glb',
			'Fern.glb', 'Mushroom.glb', 'Mushroom Laetiporus.glb',
			'Plant.glb', 'Plant Big.glb'
		],
		density: 3,
		scale: [0.08, 0.18],
		yOffset: 0
	},
	// Plains (11): grass, flowers, clover
	11: {
		files: [
			'Grass.glb', 'Grass Wispy.glb', 'Grass Wispy-Msr9zx66VU.glb', 'Tall Grass.glb',
			'Flower Single.glb', 'Flower Group.glb', 'Flower Petal.glb',
			'Clover.glb', 'Bush.glb', 'Bush with Flowers.glb'
		],
		density: 4,
		scale: [0.06, 0.14],
		yOffset: 0
	},
	// Empty/sandy (0): sparse grass, pebbles
	0: {
		files: [
			'Grass Wispy.glb', 'Pebble Round.glb', 'Pebble Square.glb',
			'Pebble Round-icVsN3lmVy.glb', 'Pebble Square-2YtLzwgsWp.glb',
			'Rock Path Round Small.glb'
		],
		density: 1.5,
		scale: [0.04, 0.10],
		yOffset: 0
	},
	// Mountain (5): rocks, dead trees
	5: {
		files: [
			'Rock Medium.glb', 'Rock Medium-JQxF95498B.glb', 'Rock Medium-s1OJ3bBzqc.glb',
			'Dead Tree.glb', 'Dead Tree-CD4edbPSGm.glb', 'Dead Tree-Mcd2zYqyww.glb',
			'Pebble Round-kAMfq1uJUY.glb', 'Pebble Square-6juX57sLHe.glb'
		],
		density: 2,
		scale: [0.06, 0.16],
		yOffset: 0
	},
	// Ruin (3): dead trees, rocks, sparse plants
	3: {
		files: [
			'Dead Tree-MlmK5488ou.glb', 'Dead Tree-n8FhMgMldD.glb',
			'Rock Medium.glb', 'Pebble Round.glb',
			'Twisted Tree.glb', 'Twisted Tree-7PDBpElkQr.glb'
		],
		density: 2,
		scale: [0.05, 0.12],
		yOffset: 0
	},
	// Settlement (1): flowers, small plants, paths
	1: {
		files: [
			'Flower Single-GvfHo0roi3.glb', 'Plant-xH5gNlQxAZ.glb',
			'Rock Path Round Small.glb', 'Rock Path Square Smal.glb',
			'Bush.glb'
		],
		density: 1,
		scale: [0.04, 0.08],
		yOffset: 0
	}
};

// Cache loaded models
const modelCache = new Map<string, THREE.Group | null>();

async function loadModel(file: string): Promise<THREE.Group | null> {
	if (modelCache.has(file)) return modelCache.get(file)!;

	const loader = new GLTFLoader();
	try {
		const gltf = await loader.loadAsync(`/models/${file}`);
		const model = gltf.scene;
		// Enable shadows on all meshes
		model.traverse((child) => {
			if (child instanceof THREE.Mesh) {
				child.castShadow = true;
				child.receiveShadow = true;
			}
		});
		modelCache.set(file, model);
		return model;
	} catch {
		modelCache.set(file, null);
		return null;
	}
}

export async function createScatter(
	grid: number[][],
	heightFn: (x: number, z: number) => number,
	maxInstances = 2000
): Promise<ScatterSystem> {
	const rows = grid.length;
	const cols = grid[0].length;
	const ox = -cols / 2;
	const oz = -rows / 2;
	const rng = mulberry32(54321);

	const group = new THREE.Group();

	// Collect cells per terrain type
	const cellsByType = new Map<number, { x: number; z: number }[]>();
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			const code = grid[y][x];
			if (!SCATTER_DEFS[code]) continue;
			if (!cellsByType.has(code)) cellsByType.set(code, []);
			cellsByType.get(code)!.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
		}
	}

	let totalPlaced = 0;

	// Load and scatter per terrain type
	for (const [code, cells] of cellsByType.entries()) {
		const def = SCATTER_DEFS[code];
		if (!def) continue;

		// Pre-load a random subset of models for this terrain type
		const modelFiles = def.files.slice(0, 6); // limit loaded models for perf
		const loadedModels: THREE.Group[] = [];
		for (const file of modelFiles) {
			const m = await loadModel(file);
			if (m) loadedModels.push(m);
		}
		if (loadedModels.length === 0) continue;

		// Scatter instances
		const instanceCount = Math.floor(cells.length * def.density);
		for (let i = 0; i < instanceCount && totalPlaced < maxInstances; i++) {
			const cell = cells[Math.floor(rng() * cells.length)];
			const px = cell.x + (rng() - 0.5) * 0.85;
			const pz = cell.z + (rng() - 0.5) * 0.85;
			const py = heightFn(px, pz) + def.yOffset;

			const baseModel = loadedModels[Math.floor(rng() * loadedModels.length)];
			const instance = baseModel.clone();

			const scale = def.scale[0] + rng() * (def.scale[1] - def.scale[0]);
			instance.scale.setScalar(scale);
			instance.position.set(px, py, pz);
			instance.rotation.y = rng() * Math.PI * 2;

			group.add(instance);
			totalPlaced++;
		}
	}

	// Frustum culling: hide objects outside camera view + distance cull
	const _frustum = new THREE.Frustum();
	const _projScreenMatrix = new THREE.Matrix4();
	const _sphere = new THREE.Sphere();
	const CULL_DISTANCE_SQ = 25 * 25; // 25 units max render distance
	let cullFrame = 0;

	return {
		group,

		updateCulling(camera: THREE.PerspectiveCamera) {
			// Only cull every 3rd frame for performance
			cullFrame++;
			if (cullFrame % 3 !== 0) return;

			_projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
			_frustum.setFromProjectionMatrix(_projScreenMatrix);

			const camPos = camera.position;

			for (const child of group.children) {
				// Distance culling — hide objects far from camera
				const dx = child.position.x - camPos.x;
				const dz = child.position.z - camPos.z;
				const distSq = dx * dx + dz * dz;

				if (distSq > CULL_DISTANCE_SQ) {
					child.visible = false;
					continue;
				}

				// Frustum culling — hide objects outside camera view
				_sphere.center.copy(child.position);
				_sphere.radius = 0.5; // approximate bounding sphere
				child.visible = _frustum.intersectsSphere(_sphere);
			}
		},

		dispose() {
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
		}
	};
}
