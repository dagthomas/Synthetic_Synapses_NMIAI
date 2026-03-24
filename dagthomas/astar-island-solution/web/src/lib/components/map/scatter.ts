/**
 * Environmental model scatter — densely populates the terrain with vegetation,
 * rocks, and detail models from /static/models (poly.pizza GLBs).
 *
 * Uses InstancedMesh for massive draw-call reduction (~3000 → ~50-100).
 * Models sourced from https://poly.pizza/bundle
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mulberry32 } from './prng';
import { applyWindSway, type WindSystem } from './wind';

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
	// Forest (4): bushes only — tree GLB models placed by addForest()
	4: {
		files: [
			'Bush.glb', 'Bush with Flowers.glb'
		],
		density: 1,
		scale: [0.06, 0.14],
		yOffset: 0
	},
	// Plains (11): only bushes — skip all flowers/grass/clover/petals
	11: {
		files: [
			'Bush.glb', 'Bush with Flowers.glb'
		],
		density: 1,
		scale: [0.06, 0.14],
		yOffset: 0
	},
	// Mountain (5): skip — terrain heightmap handles mountain visuals
	// Ruin (3): dead trees and rocks only
	3: {
		files: [
			'Dead Tree-MlmK5488ou.glb', 'Dead Tree-n8FhMgMldD.glb',
			'Rock Medium.glb', 'Rock Medium-s1OJ3bBzqc.glb',
			'Twisted Tree.glb', 'Twisted Tree-7PDBpElkQr.glb'
		],
		density: 1,
		scale: [0.06, 0.16],
		yOffset: 0
	}
	// Removed: Empty/sandy (0), Settlement (1), Port (2) — too small to see, pure FPS waste
};

const glbCache = new Map<string, THREE.Group | null>();

async function loadGLB(file: string): Promise<THREE.Group | null> {
	if (glbCache.has(file)) return glbCache.get(file)!;
	const loader = new GLTFLoader();
	try {
		const gltf = await loader.loadAsync(`/models/${file}`);
		const model = gltf.scene;
		glbCache.set(file, model);
		return model;
	} catch {
		glbCache.set(file, null);
		return null;
	}
}

/** Extract all child Meshes from a GLB model, keyed by geometry+material UUID */
interface MeshTemplate {
	geometry: THREE.BufferGeometry;
	material: THREE.Material | THREE.Material[];
	localMatrix: THREE.Matrix4; // local transform within the GLB
}

const _inverseRoot = new THREE.Matrix4();

function extractMeshes(model: THREE.Group): MeshTemplate[] {
	const templates: MeshTemplate[] = [];
	// Ensure entire hierarchy has up-to-date matrices
	model.updateMatrixWorld(true);
	// Get inverse of model root to compute child transforms RELATIVE to root
	_inverseRoot.copy(model.matrixWorld).invert();

	model.traverse((child) => {
		if (child instanceof THREE.Mesh) {
			// localMatrix = transform of this mesh relative to the model root
			const relativeMatrix = new THREE.Matrix4().multiplyMatrices(_inverseRoot, child.matrixWorld);
			templates.push({
				geometry: child.geometry,
				material: child.material,
				localMatrix: relativeMatrix
			});
		}
	});
	return templates;
}

/** Build a unique key for a (geometry, material) pair */
function meshKey(geo: THREE.BufferGeometry, mat: THREE.Material | THREE.Material[]): string {
	const matId = Array.isArray(mat) ? mat.map(m => m.uuid).join('+') : mat.uuid;
	return `${geo.uuid}::${matId}`;
}

export async function createScatter(
	grid: number[][],
	heightFn: (x: number, z: number) => number,
	maxInstances = 800,
	windUniforms?: WindSystem['uniforms']
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

	// Extract mesh templates from each loaded GLB
	const modelTemplates = new Map<string, MeshTemplate[]>(); // file → templates
	for (const file of allFiles) {
		const model = glbCache.get(file);
		if (model) {
			modelTemplates.set(file, extractMeshes(model));
		}
	}

	// Collect instance matrices per unique (geometry, material) pair
	// Key: meshKey → { geo, mat, matrices[], parentModelInverse }
	interface InstanceBucket {
		geometry: THREE.BufferGeometry;
		material: THREE.Material | THREE.Material[];
		matrices: THREE.Matrix4[];
	}
	const buckets = new Map<string, InstanceBucket>();

	const _parentMatrix = new THREE.Matrix4();
	const _instanceMatrix = new THREE.Matrix4();
	const _euler = new THREE.Euler();
	const _quat = new THREE.Quaternion();
	const _scale = new THREE.Vector3();
	const _pos = new THREE.Vector3();

	// Scatter per terrain type
	for (const [code, cells] of cellsByType.entries()) {
		const def = SCATTER_DEFS[code];
		if (!def) continue;

		// Get available model files for this terrain
		const availableFiles: string[] = [];
		for (const file of def.files) {
			if (modelTemplates.has(file)) availableFiles.push(file);
		}
		if (availableFiles.length === 0) continue;

		const instanceCount = Math.floor(cells.length * def.density);
		for (let i = 0; i < instanceCount && totalPlaced < maxInstances; i++) {
			const cell = cells[Math.floor(rng() * cells.length)];
			const px = cell.x + (rng() - 0.5) * 0.9;
			const pz = cell.z + (rng() - 0.5) * 0.9;
			const py = heightFn(px, pz) + def.yOffset;

			const file = availableFiles[Math.floor(rng() * availableFiles.length)];
			const templates = modelTemplates.get(file)!;

			const scale = def.scale[0] + rng() * (def.scale[1] - def.scale[0]);
			const rotY = rng() * Math.PI * 2;
			const rotX = (rng() - 0.5) * 0.06;
			const rotZ = (rng() - 0.5) * 0.06;

			// Build the parent (group) transform
			_pos.set(px, py, pz);
			_euler.set(rotX, rotY, rotZ);
			_quat.setFromEuler(_euler);
			_scale.setScalar(scale);
			_parentMatrix.compose(_pos, _quat, _scale);

			// For each sub-mesh in this model, compute the final instance matrix
			for (const tmpl of templates) {
				_instanceMatrix.multiplyMatrices(_parentMatrix, tmpl.localMatrix);

				const key = meshKey(tmpl.geometry, tmpl.material);
				let bucket = buckets.get(key);
				if (!bucket) {
					bucket = { geometry: tmpl.geometry, material: tmpl.material, matrices: [] };
					buckets.set(key, bucket);
				}
				bucket.matrices.push(_instanceMatrix.clone());
			}

			totalPlaced++;
		}
	}

	// Create InstancedMesh for each bucket, with wind sway on vegetation
	for (const bucket of buckets.values()) {
		const count = bucket.matrices.length;
		if (count === 0) continue;

		// Clone material so onBeforeCompile doesn't affect cached originals
		let mat: THREE.Material | THREE.Material[] = bucket.material;
		if (windUniforms && !Array.isArray(mat) && (mat as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
			mat = (mat as THREE.MeshStandardMaterial).clone();
			applyWindSway(mat as THREE.MeshStandardMaterial, windUniforms);
		}

		const instMesh = new THREE.InstancedMesh(bucket.geometry, mat, count);
		instMesh.castShadow = false; // small vegetation — shadow cost >> visual benefit
		instMesh.receiveShadow = true;
		instMesh.frustumCulled = true; // let Three.js handle frustum culling on the bounding sphere

		for (let i = 0; i < count; i++) {
			instMesh.setMatrixAt(i, bucket.matrices[i]);
		}
		instMesh.instanceMatrix.needsUpdate = true;
		instMesh.computeBoundingSphere();

		group.add(instMesh);
	}

	// Frustum culling at InstancedMesh level (not per-instance)
	const _frustum = new THREE.Frustum();
	const _projScreenMatrix = new THREE.Matrix4();
	let cullFrame = 0;

	return {
		group,

		updateCulling(camera: THREE.PerspectiveCamera) {
			cullFrame++;
			if (cullFrame % 3 !== 0) return;

			_projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
			_frustum.setFromProjectionMatrix(_projScreenMatrix);

			for (const child of group.children) {
				if (child instanceof THREE.InstancedMesh && child.boundingSphere) {
					child.visible = _frustum.intersectsSphere(child.boundingSphere);
				}
			}
		},

		dispose() {
			// Only dispose InstancedMeshes we created (geometries/materials are shared from cache)
			for (const child of group.children) {
				if (child instanceof THREE.InstancedMesh) {
					child.dispose();
				}
			}
		}
	};
}
