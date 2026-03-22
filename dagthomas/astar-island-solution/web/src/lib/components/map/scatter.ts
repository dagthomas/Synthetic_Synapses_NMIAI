/**
 * Environmental model scatter — densely populates the terrain with vegetation,
 * rocks, and detail models from /static/models (poly.pizza GLBs).
 *
 * Used in both orbit view and FP/flythrough modes.
 * Models sourced from https://poly.pizza/bundle
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mulberry32 } from './prng';

export interface ScatterSystem {
	group: THREE.Group;
	updateCulling(camera: THREE.PerspectiveCamera): void;
	dispose(): void;
}

interface ScatterDef {
	files: string[];
	density: number;
	scale: [number, number];
	yOffset: number;
}

// Terrain code → vegetation/detail definitions
const SCATTER_DEFS: Record<number, ScatterDef> = {
	// Forest (4): dense trees, ferns, mushrooms, undergrowth
	4: {
		files: [
			'Pine.glb', 'Pine-699sFuLCN2.glb', 'Pine-79gmlLnweB.glb', 'Pine-rfnxJv0Rqa.glb', 'Pine-Zt62gceKXZ.glb',
			'Tree.glb', 'Tree-aVOxaHRPWe.glb', 'Tree-QVOop92WmG.glb', 'Tree-qZtx0AHhcy.glb', 'Tree-t9KbsfYdXz.glb',
			'Fern.glb', 'Mushroom.glb', 'Mushroom Laetiporus.glb',
			'Plant.glb', 'Plant Big.glb', 'Plant Big-MbhbP7JrTI.glb', 'Plant-xH5gNlQxAZ.glb',
			'Bush.glb', 'Bush with Flowers.glb', 'Tall Grass.glb',
			'Clover.glb', 'Grass.glb'
		],
		density: 5,
		scale: [0.06, 0.20],
		yOffset: 0
	},
	// Plains (11): flowers, grass, clover, scattered bushes
	11: {
		files: [
			'Grass.glb', 'Grass Wispy.glb', 'Grass Wispy-Msr9zx66VU.glb', 'Tall Grass.glb',
			'Flower Single.glb', 'Flower Single-GvfHo0roi3.glb',
			'Flower Group.glb', 'Flower Group-LqTljN6Wg2.glb',
			'Flower Petal.glb', 'Flower Petal-eVE0j49ux9.glb', 'Flower Petal-LqvxG9OBOU.glb',
			'Flower Petal-niuBUEJdvM.glb', 'Flower Petal-tzG4JcqYWs.glb',
			'Clover.glb', 'Clover-u5SOgBFiut.glb',
			'Bush.glb', 'Bush with Flowers.glb'
		],
		density: 6,
		scale: [0.05, 0.14],
		yOffset: 0
	},
	// Empty/sandy (0): sparse grass, pebbles, occasional bush
	0: {
		files: [
			'Grass Wispy.glb', 'Grass Wispy-Msr9zx66VU.glb',
			'Pebble Round.glb', 'Pebble Round-icVsN3lmVy.glb', 'Pebble Round-kAMfq1uJUY.glb',
			'Pebble Round-KYtJ6JNXh2.glb', 'Pebble Round-nMf8LHOsbM.glb',
			'Pebble Square.glb', 'Pebble Square-2YtLzwgsWp.glb',
			'Rock Path Round Small.glb', 'Rock Path Round Small-GMttpOEFKT.glb'
		],
		density: 2,
		scale: [0.04, 0.10],
		yOffset: 0
	},
	// Mountain (5): rocks, dead/twisted trees, sparse vegetation
	5: {
		files: [
			'Rock Medium.glb', 'Rock Medium-JQxF95498B.glb', 'Rock Medium-s1OJ3bBzqc.glb',
			'Dead Tree.glb', 'Dead Tree-CD4edbPSGm.glb', 'Dead Tree-Mcd2zYqyww.glb',
			'Dead Tree-MlmK5488ou.glb', 'Dead Tree-n8FhMgMldD.glb',
			'Twisted Tree.glb', 'Twisted Tree-7PDBpElkQr.glb', 'Twisted Tree-8oraKn9m0x.glb',
			'Pebble Round-kAMfq1uJUY.glb', 'Pebble Square-6juX57sLHe.glb',
			'Pebble Square-l5XiYQj1oD.glb', 'Pebble Square-Mm4RMgwNO8.glb'
		],
		density: 3,
		scale: [0.05, 0.18],
		yOffset: 0
	},
	// Ruin (3): dead trees, rocks, twisted trees, rubble
	3: {
		files: [
			'Dead Tree-MlmK5488ou.glb', 'Dead Tree-n8FhMgMldD.glb',
			'Rock Medium.glb', 'Rock Medium-s1OJ3bBzqc.glb',
			'Twisted Tree.glb', 'Twisted Tree-7PDBpElkQr.glb',
			'Twisted Tree-9aWlx82xUf.glb', 'Twisted Tree-GVTsMmuzv7.glb',
			'Pebble Round.glb', 'Pebble Square.glb',
			'Fern.glb'
		],
		density: 3,
		scale: [0.05, 0.14],
		yOffset: 0
	},
	// Settlement (1): flowers, plants, stone paths, bushes
	1: {
		files: [
			'Flower Single-GvfHo0roi3.glb', 'Flower Single.glb',
			'Plant-xH5gNlQxAZ.glb', 'Plant.glb',
			'Rock Path Round Small.glb', 'Rock Path Round Small-yHEdadj5I0.glb',
			'Rock Path Square Smal.glb', 'Rock Path Square Smal-cI9XBpVijV.glb',
			'Rock Path Round Thin.glb', 'Rock Path Round Wide.glb',
			'Rock Path Square Thin.glb', 'Rock Path Square Wide.glb',
			'Bush.glb', 'Bush with Flowers.glb'
		],
		density: 3,
		scale: [0.04, 0.10],
		yOffset: 0
	},
	// Port (2): sparse — pebbles, small plants
	2: {
		files: [
			'Pebble Round.glb', 'Pebble Square.glb',
			'Rock Path Round Small.glb', 'Grass Wispy.glb'
		],
		density: 1,
		scale: [0.03, 0.08],
		yOffset: 0
	}
};

const glbCache = new Map<string, THREE.Group | null>();

async function loadGLB(file: string): Promise<THREE.Group | null> {
	if (glbCache.has(file)) return glbCache.get(file)!;
	const loader = new GLTFLoader();
	try {
		const gltf = await loader.loadAsync(`/models/${file}`);
		const model = gltf.scene;
		model.traverse((child) => {
			if (child instanceof THREE.Mesh) {
				child.castShadow = true;
				child.receiveShadow = true;
			}
		});
		glbCache.set(file, model);
		return model;
	} catch {
		glbCache.set(file, null);
		return null;
	}
}

export async function createScatter(
	grid: number[][],
	heightFn: (x: number, z: number) => number,
	maxInstances = 3000
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

	// Load unique files first (dedup across terrain types)
	const allFiles = new Set<string>();
	for (const def of Object.values(SCATTER_DEFS)) {
		for (const f of def.files) allFiles.add(f);
	}
	await Promise.allSettled([...allFiles].map(f => loadGLB(f)));

	// Scatter per terrain type
	for (const [code, cells] of cellsByType.entries()) {
		const def = SCATTER_DEFS[code];
		if (!def) continue;

		const loadedModels: THREE.Group[] = [];
		for (const file of def.files) {
			const m = glbCache.get(file);
			if (m) loadedModels.push(m);
		}
		if (loadedModels.length === 0) continue;

		const instanceCount = Math.floor(cells.length * def.density);
		for (let i = 0; i < instanceCount && totalPlaced < maxInstances; i++) {
			const cell = cells[Math.floor(rng() * cells.length)];
			const px = cell.x + (rng() - 0.5) * 0.9;
			const pz = cell.z + (rng() - 0.5) * 0.9;
			const py = heightFn(px, pz) + def.yOffset;

			const baseModel = loadedModels[Math.floor(rng() * loadedModels.length)];
			const instance = baseModel.clone();

			const scale = def.scale[0] + rng() * (def.scale[1] - def.scale[0]);
			instance.scale.setScalar(scale);
			instance.position.set(px, py, pz);
			instance.rotation.y = rng() * Math.PI * 2;
			// Slight random tilt for organic feel
			instance.rotation.x = (rng() - 0.5) * 0.06;
			instance.rotation.z = (rng() - 0.5) * 0.06;

			group.add(instance);
			totalPlaced++;
		}
	}

	// Frustum + distance culling
	const _frustum = new THREE.Frustum();
	const _projScreenMatrix = new THREE.Matrix4();
	const _sphere = new THREE.Sphere();
	const CULL_DISTANCE_SQ = 60 * 60;
	let cullFrame = 0;

	return {
		group,

		updateCulling(camera: THREE.PerspectiveCamera) {
			cullFrame++;
			if (cullFrame % 3 !== 0) return;

			_projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
			_frustum.setFromProjectionMatrix(_projScreenMatrix);
			const camPos = camera.position;

			for (const child of group.children) {
				const dx = child.position.x - camPos.x;
				const dz = child.position.z - camPos.z;
				if (dx * dx + dz * dz > CULL_DISTANCE_SQ) {
					child.visible = false;
					continue;
				}
				_sphere.center.copy(child.position);
				_sphere.radius = 0.5;
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
