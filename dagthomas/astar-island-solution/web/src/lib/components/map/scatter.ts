/**
 * Environmental scatter — procedural grass clumps, tiny flowers, and GLB detail models.
 * Grass/flowers use PRE-ALLOCATED InstancedMesh (reused across round transitions).
 * Ruins use GLB models (recreated each time — lightweight).
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mulberry32 } from './prng';
import { applyWindSway, type WindSystem } from './wind';

export interface ScatterSystem {
	group: THREE.Group;
	flowerPositions: { x: number; y: number; z: number }[];
	updatePositions(grid: number[][], heightFn: (x: number, z: number) => number): void;
	updateCulling(camera: THREE.PerspectiveCamera): void;
	dispose(): void;
}

interface ScatterDef {
	files: string[];
	density: number;
	scale: [number, number];
	yOffset: number;
}

const SCATTER_DEFS: Record<number, ScatterDef> = {
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
};

const GRASS_TERRAIN = new Set([4, 11]);

// --- Constants for pre-allocation ---
const GRASS_VARIANTS = 4;
const GRASS_MAX_PER_VARIANT = 200;
const FLOWER_VARIANTS = 6;
const FLOWER_MAX_PER_VARIANT = 40;

// --- Procedural geometry generators ---

function createGrassClumpGeometry(rng: () => number): THREE.BufferGeometry {
	const bladeCount = 3 + Math.floor(rng() * 3);
	const verts: number[] = [];
	const normals: number[] = [];
	const colors: number[] = [];
	for (let b = 0; b < bladeCount; b++) {
		const angle = rng() * Math.PI * 2;
		const dist = rng() * 0.08;
		const bx = Math.cos(angle) * dist;
		const bz = Math.sin(angle) * dist;
		const height = 0.12 + rng() * 0.18;
		const width = 0.015 + rng() * 0.01;
		const lean = (rng() - 0.5) * 0.06;
		const leanZ = (rng() - 0.5) * 0.06;
		const tipX = bx + lean;
		const tipZ = bz + leanZ;
		const perpX = -Math.sin(angle) * width;
		const perpZ = Math.cos(angle) * width;
		const g = 0.35 + rng() * 0.25;
		const r = g * (0.5 + rng() * 0.3);
		const bl = g * (0.2 + rng() * 0.15);
		const tipR = r * 0.8, tipG = g * 1.1, tipB = bl * 0.7;
		verts.push(bx - perpX, 0, bz - perpZ);
		verts.push(bx + perpX, 0, bz + perpZ);
		verts.push(tipX, height, tipZ);
		normals.push(0, 0.3, -0.95, 0, 0.3, -0.95, 0, 0.7, -0.7);
		colors.push(r, g, bl, r, g, bl, tipR, tipG, tipB);
	}
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
	geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
	geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
	return geo;
}

function createFlowerGeometry(rng: () => number): THREE.BufferGeometry {
	const petalCount = 4 + Math.floor(rng() * 2);
	const verts: number[] = [];
	const norms: number[] = [];
	const cols: number[] = [];
	const stemH = 0.08 + rng() * 0.12;
	const sw = 0.005;
	verts.push(-sw, 0, 0, sw, 0, 0, 0, stemH, 0);
	norms.push(0, 0, 1, 0, 0, 1, 0, 0, 1);
	const sg = 0.3 + rng() * 0.15;
	cols.push(0.2, sg, 0.1, 0.2, sg, 0.1, 0.25, sg * 1.1, 0.12);
	const palettes = [[1,0.85,0.2],[1,0.4,0.4],[0.9,0.5,0.9],[1,1,0.95],[0.4,0.6,1],[1,0.6,0.2]];
	const fc = palettes[Math.floor(rng() * palettes.length)];
	const petalLen = 0.025 + rng() * 0.015;
	const petalW = 0.012 + rng() * 0.008;
	for (let p = 0; p < petalCount; p++) {
		const a = (p / petalCount) * Math.PI * 2;
		const cx = Math.cos(a), cz = Math.sin(a);
		const px = -cz * petalW, pz = cx * petalW;
		verts.push(-px, stemH, -pz, px, stemH, pz, cx * petalLen, stemH + 0.003, cz * petalLen);
		norms.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
		cols.push(fc[0], fc[1], fc[2], fc[0], fc[1], fc[2], fc[0] * 0.9, fc[1] * 0.9, fc[2] * 0.9);
	}
	verts.push(-0.005, stemH + 0.002, 0, 0.005, stemH + 0.002, 0, 0, stemH + 0.002, 0.005);
	norms.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
	cols.push(0.9, 0.8, 0.2, 0.9, 0.8, 0.2, 0.9, 0.8, 0.2);
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
	geo.setAttribute('normal', new THREE.Float32BufferAttribute(norms, 3));
	geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
	return geo;
}

// --- GLB loading for ruin scatter ---

const glbCache = new Map<string, THREE.Group | null>();

async function loadGLB(file: string): Promise<THREE.Group | null> {
	if (glbCache.has(file)) return glbCache.get(file)!;
	const loader = new GLTFLoader();
	try {
		const gltf = await loader.loadAsync(`/models/${file}`);
		glbCache.set(file, gltf.scene);
		return gltf.scene;
	} catch {
		glbCache.set(file, null);
		return null;
	}
}

interface MeshTemplate {
	geometry: THREE.BufferGeometry;
	material: THREE.Material | THREE.Material[];
	localMatrix: THREE.Matrix4;
}

const _inverseRoot = new THREE.Matrix4();

function extractMeshes(model: THREE.Group): MeshTemplate[] {
	const templates: MeshTemplate[] = [];
	model.updateMatrixWorld(true);
	_inverseRoot.copy(model.matrixWorld).invert();
	model.traverse((child) => {
		if (child instanceof THREE.Mesh) {
			templates.push({
				geometry: child.geometry,
				material: child.material,
				localMatrix: new THREE.Matrix4().multiplyMatrices(_inverseRoot, child.matrixWorld)
			});
		}
	});
	return templates;
}

function meshKey(geo: THREE.BufferGeometry, mat: THREE.Material | THREE.Material[]): string {
	const matId = Array.isArray(mat) ? mat.map(m => m.uuid).join('+') : mat.uuid;
	return `${geo.uuid}::${matId}`;
}

// --- Reusable math objects ---
const _m = new THREE.Matrix4();
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
const _e = new THREE.Euler();
const _im = new THREE.Matrix4();

/** Fill grass + flower positions into pre-allocated InstancedMesh arrays */
function fillGrassAndFlowers(
	grid: number[][],
	heightFn: (x: number, z: number) => number,
	grassMeshes: THREE.InstancedMesh[],
	flowerMeshes: THREE.InstancedMesh[],
	maxGrassTotal: number,
	maxFlowerTotal: number,
): { x: number; y: number; z: number }[] {
	const rows = grid.length, cols = grid[0].length;
	const ox = -cols / 2, oz = -rows / 2;

	// Collect grass terrain cells
	const grassCells: { x: number; z: number }[] = [];
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			if (GRASS_TERRAIN.has(grid[y][x])) {
				grassCells.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
			}
		}
	}

	// Deterministic RNG from grid hash
	let seed = 0;
	for (let y = 0; y < Math.min(5, rows); y++)
		for (let x = 0; x < Math.min(5, cols); x++)
			seed = (seed * 31 + (grid[y][x] ?? 0)) | 0;
	const rng = mulberry32(seed ^ 54321);

	// Fill grass
	const grassCounts = new Array(GRASS_VARIANTS).fill(0);
	const grassTotal = Math.min(maxGrassTotal, grassCells.length * 2);
	for (let i = 0; i < grassTotal && grassCells.length > 0; i++) {
		const cell = grassCells[Math.floor(rng() * grassCells.length)];
		const px = cell.x + (rng() - 0.5) * 0.9;
		const pz = cell.z + (rng() - 0.5) * 0.9;
		const py = heightFn(px, pz);
		if (py < -0.05) continue;

		const variant = Math.floor(rng() * GRASS_VARIANTS);
		const scale = 0.6 + rng() * 0.8;
		_p.set(px, py, pz);
		_e.set(0, rng() * Math.PI * 2, 0);
		_q.setFromEuler(_e);
		_s.setScalar(scale);
		_m.compose(_p, _q, _s);

		const idx = grassCounts[variant];
		if (idx < GRASS_MAX_PER_VARIANT) {
			grassMeshes[variant].setMatrixAt(idx, _m);
			grassCounts[variant]++;
		}
	}

	for (let v = 0; v < GRASS_VARIANTS; v++) {
		grassMeshes[v].count = grassCounts[v];
		grassMeshes[v].instanceMatrix.needsUpdate = true;
		grassMeshes[v].visible = grassCounts[v] > 0;
		if (grassCounts[v] > 0) grassMeshes[v].computeBoundingSphere();
	}

	// Fill flowers
	const flowerPositions: { x: number; y: number; z: number }[] = [];
	const flowerCounts = new Array(FLOWER_VARIANTS).fill(0);
	const flowerTotal = Math.min(maxFlowerTotal, Math.floor(grassCells.length * 0.4));
	for (let i = 0; i < flowerTotal && grassCells.length > 0; i++) {
		const cell = grassCells[Math.floor(rng() * grassCells.length)];
		const px = cell.x + (rng() - 0.5) * 0.9;
		const pz = cell.z + (rng() - 0.5) * 0.9;
		const py = heightFn(px, pz);
		if (py < -0.05) continue;
		flowerPositions.push({ x: px, y: py, z: pz });

		const variant = Math.floor(rng() * FLOWER_VARIANTS);
		const scale = 0.7 + rng() * 0.6;
		_p.set(px, py, pz);
		_e.set(0, rng() * Math.PI * 2, 0);
		_q.setFromEuler(_e);
		_s.setScalar(scale);
		_m.compose(_p, _q, _s);

		const idx = flowerCounts[variant];
		if (idx < FLOWER_MAX_PER_VARIANT) {
			flowerMeshes[variant].setMatrixAt(idx, _m);
			flowerCounts[variant]++;
		}
	}

	for (let v = 0; v < FLOWER_VARIANTS; v++) {
		flowerMeshes[v].count = flowerCounts[v];
		flowerMeshes[v].instanceMatrix.needsUpdate = true;
		flowerMeshes[v].visible = flowerCounts[v] > 0;
		if (flowerCounts[v] > 0) flowerMeshes[v].computeBoundingSphere();
	}

	return flowerPositions;
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

	// === Pre-allocate grass InstancedMesh (reused across transitions) ===
	const grassGeos: THREE.BufferGeometry[] = [];
	for (let v = 0; v < GRASS_VARIANTS; v++) grassGeos.push(createGrassClumpGeometry(rng));
	const grassMat = new THREE.MeshStandardMaterial({
		vertexColors: true, roughness: 0.9, metalness: 0, side: THREE.DoubleSide, flatShading: true,
	});
	let grassMatFinal = grassMat;
	if (windUniforms) { grassMatFinal = grassMat.clone(); applyWindSway(grassMatFinal, windUniforms); }

	const grassMeshes: THREE.InstancedMesh[] = [];
	for (let v = 0; v < GRASS_VARIANTS; v++) {
		const inst = new THREE.InstancedMesh(grassGeos[v], grassMatFinal, GRASS_MAX_PER_VARIANT);
		inst.castShadow = false;
		inst.receiveShadow = true;
		inst.frustumCulled = true;
		grassMeshes.push(inst);
		group.add(inst);
	}

	// === Pre-allocate flower InstancedMesh (reused across transitions) ===
	const flowerGeos: THREE.BufferGeometry[] = [];
	for (let v = 0; v < FLOWER_VARIANTS; v++) flowerGeos.push(createFlowerGeometry(rng));
	const flowerMat = new THREE.MeshStandardMaterial({
		vertexColors: true, roughness: 0.7, metalness: 0, side: THREE.DoubleSide, flatShading: true,
	});
	let flowerMatFinal = flowerMat;
	if (windUniforms) { flowerMatFinal = flowerMat.clone(); applyWindSway(flowerMatFinal, windUniforms); }

	const flowerMeshes: THREE.InstancedMesh[] = [];
	for (let v = 0; v < FLOWER_VARIANTS; v++) {
		const inst = new THREE.InstancedMesh(flowerGeos[v], flowerMatFinal, FLOWER_MAX_PER_VARIANT);
		inst.castShadow = false;
		inst.receiveShadow = true;
		inst.frustumCulled = true;
		flowerMeshes.push(inst);
		group.add(inst);
	}

	// Initial fill
	const maxGrass = Math.min(600, maxInstances);
	const maxFlower = Math.min(150, Math.floor(maxInstances * 0.2));
	let flowerPositions = fillGrassAndFlowers(grid, heightFn, grassMeshes, flowerMeshes, maxGrass, maxFlower);

	// === GLB detail scatter (ruins only — recreated each time, lightweight) ===
	let totalPlaced = 0;
	const allFiles = new Set<string>();
	for (const def of Object.values(SCATTER_DEFS)) {
		for (const f of def.files) allFiles.add(f);
	}
	await Promise.allSettled([...allFiles].map(f => loadGLB(f)));

	const modelTemplates = new Map<string, MeshTemplate[]>();
	for (const file of allFiles) {
		const model = glbCache.get(file);
		if (model) modelTemplates.set(file, extractMeshes(model));
	}

	const cellsByType = new Map<number, { x: number; z: number }[]>();
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			const code = grid[y][x];
			if (!SCATTER_DEFS[code]) continue;
			if (!cellsByType.has(code)) cellsByType.set(code, []);
			cellsByType.get(code)!.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
		}
	}

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

	for (const [code, cells] of cellsByType.entries()) {
		const def = SCATTER_DEFS[code];
		if (!def) continue;
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
			_pos.set(px, py, pz);
			_euler.set((rng() - 0.5) * 0.06, rng() * Math.PI * 2, (rng() - 0.5) * 0.06);
			_quat.setFromEuler(_euler);
			_scale.setScalar(scale);
			_parentMatrix.compose(_pos, _quat, _scale);
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

	// GLB detail InstancedMesh objects (these ARE recreated — lightweight)
	const glbMeshes: THREE.InstancedMesh[] = [];
	for (const bucket of buckets.values()) {
		const count = bucket.matrices.length;
		if (count === 0) continue;
		let mat: THREE.Material | THREE.Material[] = bucket.material;
		if (windUniforms && !Array.isArray(mat) && (mat as THREE.MeshStandardMaterial).isMeshStandardMaterial) {
			mat = (mat as THREE.MeshStandardMaterial).clone();
			applyWindSway(mat as THREE.MeshStandardMaterial, windUniforms);
		}
		const instMesh = new THREE.InstancedMesh(bucket.geometry, mat, count);
		instMesh.castShadow = false;
		instMesh.receiveShadow = true;
		instMesh.frustumCulled = true;
		for (let i = 0; i < count; i++) instMesh.setMatrixAt(i, bucket.matrices[i]);
		instMesh.instanceMatrix.needsUpdate = true;
		instMesh.computeBoundingSphere();
		group.add(instMesh);
		glbMeshes.push(instMesh);
	}

	const _frustum = new THREE.Frustum();
	const _projScreenMatrix = new THREE.Matrix4();
	let cullFrame = 0;

	return {
		group,
		flowerPositions,

		updatePositions(newGrid: number[][], newHeightFn: (x: number, z: number) => number) {
			// Reuse existing grass + flower InstancedMesh — just rewrite matrices
			flowerPositions = fillGrassAndFlowers(
				newGrid, newHeightFn,
				grassMeshes, flowerMeshes,
				maxGrass, maxFlower
			);
			// Note: GLB detail scatter is NOT updated here (lightweight, stays from initial build)
		},

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
			for (const geo of grassGeos) geo.dispose();
			for (const geo of flowerGeos) geo.dispose();
			grassMat.dispose();
			if (grassMatFinal !== grassMat) grassMatFinal.dispose();
			flowerMat.dispose();
			if (flowerMatFinal !== flowerMat) flowerMatFinal.dispose();
			for (const child of group.children) {
				if (child instanceof THREE.InstancedMesh) child.dispose();
			}
		}
	};
}
