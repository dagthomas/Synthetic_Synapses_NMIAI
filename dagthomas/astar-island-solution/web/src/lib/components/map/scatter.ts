/**
 * Environmental scatter — procedural grass clumps, tiny flowers, and GLB detail models.
 * Grass/flowers are generated geometry (ultra-low poly). Ruins use GLB models.
 * Uses InstancedMesh for massive draw-call reduction.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { mulberry32 } from './prng';
import { applyWindSway, type WindSystem } from './wind';

export interface ScatterSystem {
	group: THREE.Group;
	flowerPositions: { x: number; y: number; z: number }[];
	updateCulling(camera: THREE.PerspectiveCamera): void;
	dispose(): void;
}

interface ScatterDef {
	files: string[];
	density: number;
	scale: [number, number];
	yOffset: number;
}

// Ruin terrain keeps GLB scatter
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

// Terrain types that get procedural grass + flowers
const GRASS_TERRAIN = new Set([4, 11]); // forest, plains

// --- Procedural grass clump geometry (3-5 blades per clump, ~10 tris total) ---
function createGrassClumpGeometry(rng: () => number): THREE.BufferGeometry {
	const bladeCount = 3 + Math.floor(rng() * 3); // 3-5 blades
	const verts: number[] = [];
	const normals: number[] = [];
	const colors: number[] = [];

	for (let b = 0; b < bladeCount; b++) {
		// Each blade: thin triangle (base + tip)
		const angle = rng() * Math.PI * 2;
		const dist = rng() * 0.08;
		const bx = Math.cos(angle) * dist;
		const bz = Math.sin(angle) * dist;
		const height = 0.12 + rng() * 0.18; // 0.12-0.30
		const width = 0.015 + rng() * 0.01;
		const lean = (rng() - 0.5) * 0.06;
		const leanZ = (rng() - 0.5) * 0.06;

		// Blade is a quad (2 triangles) — slight curve via lean
		const tipX = bx + lean;
		const tipZ = bz + leanZ;

		// perpendicular to blade direction for width
		const perpX = -Math.sin(angle) * width;
		const perpZ = Math.cos(angle) * width;

		// Color variation: green with slight yellow/dark tints
		const g = 0.35 + rng() * 0.25;
		const r = g * (0.5 + rng() * 0.3);
		const bl = g * (0.2 + rng() * 0.15);
		const tipR = r * 0.8, tipG = g * 1.1, tipB = bl * 0.7;

		// Triangle 1: left-base, right-base, tip
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

// --- Procedural flower geometry (4-5 petals + center, ~6 tris) ---
function createFlowerGeometry(rng: () => number): { geo: THREE.BufferGeometry; color: THREE.Color } {
	const petalCount = 4 + Math.floor(rng() * 2);
	const verts: number[] = [];
	const norms: number[] = [];
	const cols: number[] = [];

	// Stem (single thin triangle)
	const stemH = 0.08 + rng() * 0.12;
	const sw = 0.005;
	verts.push(-sw, 0, 0, sw, 0, 0, 0, stemH, 0);
	norms.push(0, 0, 1, 0, 0, 1, 0, 0, 1);
	const sg = 0.3 + rng() * 0.15;
	cols.push(0.2, sg, 0.1, 0.2, sg, 0.1, 0.25, sg * 1.1, 0.12);

	// Flower color palette
	const palettes = [
		[1.0, 0.85, 0.2],  // yellow
		[1.0, 0.4, 0.4],   // red/pink
		[0.9, 0.5, 0.9],   // purple
		[1.0, 1.0, 0.95],  // white
		[0.4, 0.6, 1.0],   // blue
		[1.0, 0.6, 0.2],   // orange
	];
	const fc = palettes[Math.floor(rng() * palettes.length)];
	const flowerColor = new THREE.Color(fc[0], fc[1], fc[2]);

	// Petals (triangles radiating from center)
	const petalLen = 0.025 + rng() * 0.015;
	const petalW = 0.012 + rng() * 0.008;
	for (let p = 0; p < petalCount; p++) {
		const a = (p / petalCount) * Math.PI * 2;
		const cx = Math.cos(a), cz = Math.sin(a);
		const px = -cz * petalW, pz = cx * petalW;

		verts.push(-px, stemH, -pz);
		verts.push(px, stemH, pz);
		verts.push(cx * petalLen, stemH + 0.003, cz * petalLen);

		norms.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
		cols.push(fc[0], fc[1], fc[2], fc[0], fc[1], fc[2], fc[0] * 0.9, fc[1] * 0.9, fc[2] * 0.9);
	}

	// Center dot (tiny triangle)
	verts.push(-0.005, stemH + 0.002, 0, 0.005, stemH + 0.002, 0, 0, stemH + 0.002, 0.005);
	norms.push(0, 1, 0, 0, 1, 0, 0, 1, 0);
	cols.push(0.9, 0.8, 0.2, 0.9, 0.8, 0.2, 0.9, 0.8, 0.2);

	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
	geo.setAttribute('normal', new THREE.Float32BufferAttribute(norms, 3));
	geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
	return { geo, color: flowerColor };
}

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

	// Collect cells per terrain type (GLB scatter + grass/flower terrain)
	const cellsByType = new Map<number, { x: number; z: number }[]>();
	const grassCells: { x: number; z: number }[] = [];
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			const code = grid[y][x];
			if (GRASS_TERRAIN.has(code)) {
				grassCells.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
			}
			if (!SCATTER_DEFS[code]) continue;
			if (!cellsByType.has(code)) cellsByType.set(code, []);
			cellsByType.get(code)!.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
		}
	}

	let totalPlaced = 0;

	// === Procedural grass clumps ===
	const GRASS_CLUMPS = Math.min(600, grassCells.length * 2);
	const GRASS_VARIANTS = 4; // different clump shapes
	const grassGeos: THREE.BufferGeometry[] = [];
	for (let v = 0; v < GRASS_VARIANTS; v++) {
		grassGeos.push(createGrassClumpGeometry(rng));
	}
	const grassMat = new THREE.MeshStandardMaterial({
		vertexColors: true,
		roughness: 0.9,
		metalness: 0.0,
		side: THREE.DoubleSide,
		flatShading: true,
	});
	let grassMatWithWind = grassMat as THREE.MeshStandardMaterial;
	if (windUniforms) {
		grassMatWithWind = grassMat.clone();
		applyWindSway(grassMatWithWind, windUniforms);
	}

	// Create one InstancedMesh per grass variant
	const grassInstancesPerVariant = Math.ceil(GRASS_CLUMPS / GRASS_VARIANTS);
	const grassInstMeshes: THREE.InstancedMesh[] = [];
	const grassCounts: number[] = new Array(GRASS_VARIANTS).fill(0);

	for (let v = 0; v < GRASS_VARIANTS; v++) {
		const inst = new THREE.InstancedMesh(grassGeos[v], grassMatWithWind, grassInstancesPerVariant);
		inst.castShadow = false;
		inst.receiveShadow = true;
		inst.frustumCulled = true;
		grassInstMeshes.push(inst);
	}

	const _gm = new THREE.Matrix4();
	const _gp = new THREE.Vector3();
	const _gq = new THREE.Quaternion();
	const _gs = new THREE.Vector3();
	const _ge = new THREE.Euler();

	for (let i = 0; i < GRASS_CLUMPS && grassCells.length > 0; i++) {
		const cell = grassCells[Math.floor(rng() * grassCells.length)];
		const px = cell.x + (rng() - 0.5) * 0.9;
		const pz = cell.z + (rng() - 0.5) * 0.9;
		const py = heightFn(px, pz);
		if (py < -0.05) continue; // skip underwater

		const variant = Math.floor(rng() * GRASS_VARIANTS);
		const scale = 0.6 + rng() * 0.8;
		_gp.set(px, py, pz);
		_ge.set(0, rng() * Math.PI * 2, 0);
		_gq.setFromEuler(_ge);
		_gs.setScalar(scale);
		_gm.compose(_gp, _gq, _gs);

		const idx = grassCounts[variant];
		if (idx < grassInstancesPerVariant) {
			grassInstMeshes[variant].setMatrixAt(idx, _gm);
			grassCounts[variant]++;
		}
	}

	for (let v = 0; v < GRASS_VARIANTS; v++) {
		grassInstMeshes[v].count = grassCounts[v];
		grassInstMeshes[v].instanceMatrix.needsUpdate = true;
		if (grassCounts[v] > 0) {
			grassInstMeshes[v].computeBoundingSphere();
			group.add(grassInstMeshes[v]);
		}
	}

	// === Procedural flowers (sparse, among grass) ===
	const FLOWER_COUNT = Math.min(150, Math.floor(grassCells.length * 0.4));
	const FLOWER_VARIANTS = 6;
	const flowerGeos: THREE.BufferGeometry[] = [];
	for (let v = 0; v < FLOWER_VARIANTS; v++) {
		flowerGeos.push(createFlowerGeometry(rng).geo);
	}
	const flowerMat = new THREE.MeshStandardMaterial({
		vertexColors: true,
		roughness: 0.7,
		metalness: 0.0,
		side: THREE.DoubleSide,
		flatShading: true,
	});
	let flowerMatWithWind = flowerMat as THREE.MeshStandardMaterial;
	if (windUniforms) {
		flowerMatWithWind = flowerMat.clone();
		applyWindSway(flowerMatWithWind, windUniforms);
	}

	const flowerInstancesPerVariant = Math.ceil(FLOWER_COUNT / FLOWER_VARIANTS);
	const flowerInstMeshes: THREE.InstancedMesh[] = [];
	const flowerCounts: number[] = new Array(FLOWER_VARIANTS).fill(0);

	for (let v = 0; v < FLOWER_VARIANTS; v++) {
		const inst = new THREE.InstancedMesh(flowerGeos[v], flowerMatWithWind, flowerInstancesPerVariant);
		inst.castShadow = false;
		inst.receiveShadow = true;
		inst.frustumCulled = true;
		flowerInstMeshes.push(inst);
	}

	const flowerPositions: { x: number; y: number; z: number }[] = [];
	for (let i = 0; i < FLOWER_COUNT && grassCells.length > 0; i++) {
		const cell = grassCells[Math.floor(rng() * grassCells.length)];
		const px = cell.x + (rng() - 0.5) * 0.9;
		const pz = cell.z + (rng() - 0.5) * 0.9;
		const py = heightFn(px, pz);
		if (py < -0.05) continue;
		flowerPositions.push({ x: px, y: py, z: pz });

		const variant = Math.floor(rng() * FLOWER_VARIANTS);
		const scale = 0.7 + rng() * 0.6;
		_gp.set(px, py, pz);
		_ge.set(0, rng() * Math.PI * 2, 0);
		_gq.setFromEuler(_ge);
		_gs.setScalar(scale);
		_gm.compose(_gp, _gq, _gs);

		const idx = flowerCounts[variant];
		if (idx < flowerInstancesPerVariant) {
			flowerInstMeshes[variant].setMatrixAt(idx, _gm);
			flowerCounts[variant]++;
		}
	}

	for (let v = 0; v < FLOWER_VARIANTS; v++) {
		flowerInstMeshes[v].count = flowerCounts[v];
		flowerInstMeshes[v].instanceMatrix.needsUpdate = true;
		if (flowerCounts[v] > 0) {
			flowerInstMeshes[v].computeBoundingSphere();
			group.add(flowerInstMeshes[v]);
		}
	}

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
		flowerPositions,

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
			for (const child of group.children) {
				if (child instanceof THREE.InstancedMesh) {
					child.dispose();
				}
			}
			// Dispose procedural geometries and materials
			for (const geo of grassGeos) geo.dispose();
			for (const geo of flowerGeos) geo.dispose();
			grassMat.dispose();
			if (grassMatWithWind !== grassMat) grassMatWithWind.dispose();
			flowerMat.dispose();
			if (flowerMatWithWind !== flowerMat) flowerMatWithWind.dispose();
		}
	};
}
