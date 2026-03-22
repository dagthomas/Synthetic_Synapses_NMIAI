/**
 * Epic procedural terrain — dramatic heightmap with zone-blended textures.
 * Mountains rise 1.5–4 units, ocean sinks below water level, zones blend smoothly.
 * Snow caps on peaks, exposed rock on slopes, wet sand at shorelines.
 */
import * as THREE from 'three';

export interface TerrainSystem {
	mesh: THREE.Mesh;
	getHeightAt(worldX: number, worldZ: number): number;
	dispose(): void;
}

// ═══════════════════════════════════════
//  Constants
// ═══════════════════════════════════════

export const WATER_LEVEL = -0.12;
const SNOW_START = 1.6;
const SNOW_FULL = 2.8;

/** Base heights for non-mountain terrain (mountain height added via overlay) */
const BASE_HEIGHTS: Record<number, number> = {
	0:  0.12,   // Sandy
	1:  0.16,   // Settlement
	2:  0.04,   // Port (just above water)
	3:  0.16,   // Ruin
	4:  0.20,   // Forest
	5:  0.36,   // Mountain base (peaks come from overlay)
	10: -0.55,  // Ocean (deep)
	11: 0.14,   // Plains
};

const BIOME_COLORS: Record<number, [number, number, number]> = {
	0:  [0.82, 0.74, 0.50],  // warm sand
	1:  [0.70, 0.60, 0.38],  // worn settlement earth
	2:  [0.45, 0.55, 0.52],  // port wet stone
	3:  [0.56, 0.52, 0.44],  // mossy ruin stone
	4:  [0.20, 0.44, 0.14],  // deep forest floor
	5:  [0.50, 0.48, 0.44],  // granite
	10: [0.10, 0.25, 0.42],  // ocean floor
	11: [0.36, 0.62, 0.18],  // lush plains
};

const SNOW_COL: [number, number, number] = [0.94, 0.94, 0.97];
const ROCK_COL: [number, number, number] = [0.46, 0.44, 0.40];
const WET_SAND_COL: [number, number, number] = [0.58, 0.52, 0.38];
const DEFAULT_COL: [number, number, number] = [0.65, 0.58, 0.42];

/** FBM noise amplitude per terrain type */
const NOISE_AMP: Record<number, number> = {
	0:  0.018,  // Sandy: gentle dunes
	1:  0.012,  // Settlement: smooth
	2:  0.010,  // Port: smooth
	3:  0.030,  // Ruin: rubble
	4:  0.035,  // Forest: root bumps
	5:  0.12,   // Mountain: rough rock
	10: 0.008,  // Ocean: gentle ripples
	11: 0.020,  // Plains: rolling
};

// ═══════════════════════════════════════
//  Noise functions
// ═══════════════════════════════════════

function hash(x: number, y: number): number {
	const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
	return n - Math.floor(n);
}

function smoothNoise(x: number, z: number): number {
	const ix = Math.floor(x), iz = Math.floor(z);
	const fx = x - ix, fz = z - iz;
	const sx = fx * fx * (3 - 2 * fx);
	const sz = fz * fz * (3 - 2 * fz);
	return hash(ix, iz) * (1 - sx) * (1 - sz)
		+ hash(ix + 1, iz) * sx * (1 - sz)
		+ hash(ix, iz + 1) * (1 - sx) * sz
		+ hash(ix + 1, iz + 1) * sx * sz;
}

function fbm(x: number, z: number, octaves: number, lacunarity: number, gain: number): number {
	let value = 0, amp = 1, freq = 1, maxAmp = 0;
	for (let i = 0; i < octaves; i++) {
		value += smoothNoise(x * freq, z * freq) * amp;
		maxAmp += amp;
		amp *= gain;
		freq *= lacunarity;
	}
	return value / maxAmp;
}

/** Ridge noise for sharp mountain peaks */
function ridgeNoise(x: number, z: number, octaves: number): number {
	let value = 0, amp = 1, freq = 1, maxAmp = 0;
	for (let i = 0; i < octaves; i++) {
		let n = smoothNoise(x * freq, z * freq);
		n = 1.0 - Math.abs(n * 2 - 1);
		n *= n;
		value += n * amp;
		maxAmp += amp;
		amp *= 0.5;
		freq *= 2.1;
	}
	return value / maxAmp;
}

/** Domain-warped FBM for organic shapes */
function warpedFbm(x: number, z: number, octaves: number): number {
	const wx = fbm(x + 5.2, z + 1.3, 3, 2.0, 0.5) * 1.5;
	const wz = fbm(x + 1.7, z + 9.2, 3, 2.0, 0.5) * 1.5;
	return fbm(x + wx, z + wz, octaves, 2.0, 0.5);
}

// ═══════════════════════════════════════
//  Height computation
// ═══════════════════════════════════════

/** Precompute mountain density field at grid resolution */
function buildMountainField(grid: number[][]): Float32Array {
	const rows = grid.length, cols = grid[0].length;
	const field = new Float32Array(rows * cols);
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			let count = 0;
			for (let dy = -3; dy <= 3; dy++) {
				for (let dx = -3; dx <= 3; dx++) {
					const cx = x + dx, cy = y + dy;
					if (cx >= 0 && cx < cols && cy >= 0 && cy < rows && grid[cy][cx] === 5) {
						count++;
					}
				}
			}
			field[y * cols + x] = count;
		}
	}
	return field;
}

/** Sample mountain density with bilinear interpolation */
function sampleMtnField(field: Float32Array, cols: number, rows: number, gx: number, gz: number): number {
	const cx = Math.max(0, Math.min(cols - 1.001, gx));
	const cz = Math.max(0, Math.min(rows - 1.001, gz));
	const ix = Math.floor(cx), iz = Math.floor(cz);
	const fx = cx - ix, fz = cz - iz;
	const ix1 = Math.min(ix + 1, cols - 1), iz1 = Math.min(iz + 1, rows - 1);
	return field[iz * cols + ix] * (1 - fx) * (1 - fz)
		+ field[iz * cols + ix1] * fx * (1 - fz)
		+ field[iz1 * cols + ix] * (1 - fx) * fz
		+ field[iz1 * cols + ix1] * fx * fz;
}

/** Gaussian-blended base height (excludes mountain overlay) */
function blendBaseHeight(grid: number[][], gx: number, gz: number): number {
	const rows = grid.length, cols = grid[0].length;
	const sigma2 = 1.8; // 2 * sigma^2 where sigma ≈ 0.95
	let total = 0, weight = 0;
	for (let dy = -2; dy <= 2; dy++) {
		for (let dx = -2; dx <= 2; dx++) {
			const cx = Math.round(gx) + dx;
			const cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			const d2 = (gx - cx) * (gx - cx) + (gz - cy) * (gz - cy);
			const w = Math.exp(-d2 / sigma2);
			const code = grid[cy][cx];
			// For mountains, use the base foothill height (peak comes from overlay)
			total += (BASE_HEIGHTS[code] ?? 0.06) * w;
			weight += w;
		}
	}
	return weight > 0 ? total / weight : 0.06;
}

/** Full height at any grid coordinate */
function computeHeight(
	grid: number[][], mtnField: Float32Array,
	cols: number, rows: number,
	gx: number, gz: number
): number {
	// Base terrain blend
	let h = blendBaseHeight(grid, gx, gz);

	// Mountain overlay: additive peaked height
	const tix = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
	const tiy = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
	const code = grid[tiy]?.[tix] ?? 0;

	// Check nearby mountain cells for peak overlay
	let mtnPeak = 0;
	for (let dy = -3; dy <= 3; dy++) {
		for (let dx = -3; dx <= 3; dx++) {
			const cx = Math.round(gx) + dx;
			const cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			if (grid[cy][cx] !== 5) continue;
			const dist = Math.sqrt((gx - cx) * (gx - cx) + (gz - cy) * (gz - cy));
			const radius = 2.8;
			if (dist >= radius) continue;
			const t = dist / radius;
			const profile = (1 - t * t) * (1 - t * t); // quartic falloff
			mtnPeak = Math.max(mtnPeak, profile);
		}
	}
	if (mtnPeak > 0) {
		// Scale peak height by nearby mountain density
		const density = sampleMtnField(mtnField, cols, rows, gx, gz);
		const peakScale = density >= 12 ? 4.0 : density >= 6 ? 3.2 : density >= 3 ? 2.2 : 1.5;
		h += mtnPeak * peakScale;
	}

	// Terrain-type noise
	const amp = NOISE_AMP[code] ?? 0.015;
	h += (warpedFbm(gx * 0.8, gz * 0.8, 5) - 0.5) * 2 * amp;

	// Mountain ridge noise for craggy peaks
	if (mtnPeak > 0.2) {
		h += ridgeNoise(gx * 0.5, gz * 0.5, 4) * mtnPeak * 0.6;
	}

	return h;
}

// ═══════════════════════════════════════
//  Color computation
// ═══════════════════════════════════════

/** Gaussian-blended biome color */
function blendColor(grid: number[][], gx: number, gz: number): [number, number, number] {
	const rows = grid.length, cols = grid[0].length;
	const sigma2 = 1.4;
	let r = 0, g = 0, b = 0, w = 0;
	for (let dy = -2; dy <= 2; dy++) {
		for (let dx = -2; dx <= 2; dx++) {
			const cx = Math.round(gx) + dx;
			const cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			const d2 = (gx - cx) * (gx - cx) + (gz - cy) * (gz - cy);
			const wt = Math.exp(-d2 / sigma2);
			const c = BIOME_COLORS[grid[cy][cx]] ?? DEFAULT_COL;
			r += c[0] * wt;
			g += c[1] * wt;
			b += c[2] * wt;
			w += wt;
		}
	}
	return w > 0 ? [r / w, g / w, b / w] : [...DEFAULT_COL];
}

// ═══════════════════════════════════════
//  Normal map (micro-detail per terrain type)
// ═══════════════════════════════════════

function createNormalMap(grid: number[][], cols: number, rows: number): THREE.CanvasTexture {
	const res = 512;
	const canvas = document.createElement('canvas');
	canvas.width = res;
	canvas.height = res;
	const ctx = canvas.getContext('2d')!;
	const imageData = ctx.createImageData(res, res);
	const data = imageData.data;

	for (let py = 0; py < res; py++) {
		for (let px = 0; px < res; px++) {
			const gx = (px / res) * cols;
			const gy = (py / res) * rows;
			const ix = Math.floor(Math.min(cols - 1, gx));
			const iy = Math.floor(Math.min(rows - 1, gy));
			const type = grid[iy][ix];
			const idx = (py * res + px) * 4;

			let nx = 0, ny = 0;
			switch (type) {
				case 5: // Mountain: craggy rock
					nx = Math.sin(px * 0.5 + py * 0.15) * 0.35 + Math.sin(px * 1.6 + py * 0.8) * 0.25
						+ Math.sin(px * 3.5 + py * 1.9) * 0.15 + Math.sin(px * 7.1) * 0.06;
					ny = Math.cos(py * 0.45 + px * 0.25) * 0.35 + Math.cos(py * 1.4 + px * 0.6) * 0.25
						+ Math.cos(py * 3.2 + px * 1.5) * 0.15 + Math.cos(py * 6.3) * 0.06;
					break;
				case 4: // Forest: clumpy undergrowth
					nx = Math.sin(px * 0.35 + py * 0.12) * 0.28 + Math.sin(px * 0.9 + py * 0.7) * 0.14
						+ Math.sin(px * 2.3 + py * 1.1) * 0.08;
					ny = Math.cos(py * 0.3 + px * 0.15) * 0.26 + Math.cos(py * 0.85 + px * 0.6) * 0.12
						+ Math.cos(py * 2.1 + px * 1.4) * 0.08;
					break;
				case 11: // Plains: wind-swept grass
					nx = Math.sin(px * 0.18 + py * 0.06) * 0.12 + Math.sin(px * 0.5) * 0.07
						+ Math.sin(px * 1.8 + py * 0.4) * 0.04;
					ny = Math.cos(py * 0.15 + px * 0.05) * 0.12 + Math.cos(py * 0.45) * 0.07
						+ Math.cos(py * 1.6 + px * 0.3) * 0.04;
					break;
				case 10: // Ocean: rolling ripples
					nx = Math.sin(px * 0.10 + py * 0.03) * 0.12 + Math.sin(px * 0.3 + py * 0.08) * 0.06;
					ny = Math.cos(py * 0.09 + px * 0.025) * 0.12 + Math.cos(py * 0.25 + px * 0.07) * 0.06;
					break;
				case 1: // Settlement: cobblestone
					nx = Math.sin(px * 1.2) * Math.cos(py * 1.5) * 0.20 + Math.sin(px * 3.5 + py * 2.1) * 0.08;
					ny = Math.cos(px * 1.4) * Math.sin(py * 1.3) * 0.20 + Math.cos(py * 3.3 + px * 1.8) * 0.08;
					break;
				case 3: // Ruin: cracked stone
					nx = Math.sin(px * 0.6 + py * 1.8) * 0.22 + Math.sin(px * 2.5 + py * 0.5) * 0.12;
					ny = Math.cos(py * 0.7 + px * 1.6) * 0.22 + Math.cos(py * 2.3 + px * 0.6) * 0.12;
					break;
				default: // Sandy: fine grain
					nx = Math.sin(px * 0.7 + py * 0.4) * 0.07 + Math.sin(px * 2.5 + py * 1.5) * 0.04;
					ny = Math.cos(py * 0.6 + px * 0.35) * 0.07 + Math.cos(py * 2.2 + px * 1.3) * 0.04;
					break;
			}

			data[idx]     = Math.round(128 + Math.max(-1, Math.min(1, nx)) * 127);
			data[idx + 1] = Math.round(128 + Math.max(-1, Math.min(1, ny)) * 127);
			data[idx + 2] = 255;
			data[idx + 3] = 255;
		}
	}

	ctx.putImageData(imageData, 0, 0);
	const tex = new THREE.CanvasTexture(canvas);
	tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
	return tex;
}

// ═══════════════════════════════════════
//  Main terrain builder
// ═══════════════════════════════════════

export function createTerrain(grid: number[][]): TerrainSystem {
	const rows = grid.length;
	const cols = grid[0].length;

	// Precompute mountain density
	const mtnField = buildMountainField(grid);

	// 4 subdivisions per cell (160x160 = 25K verts — smooth enough, much cheaper than 12)
	const subdivs = 4;
	const segX = cols * subdivs;
	const segZ = rows * subdivs;

	const geometry = new THREE.PlaneGeometry(cols, rows, segX, segZ);
	geometry.rotateX(-Math.PI / 2);

	const positions = geometry.attributes.position;
	const vertexCount = positions.count;
	const colors = new Float32Array(vertexCount * 3);

	// Pass 1: height displacement
	for (let i = 0; i < vertexCount; i++) {
		const wx = positions.getX(i);
		const wz = positions.getZ(i);
		const gx = wx + cols / 2 - 0.5;
		const gz = wz + rows / 2 - 0.5;
		positions.setY(i, computeHeight(grid, mtnField, cols, rows, gx, gz));
	}
	positions.needsUpdate = true;
	geometry.computeVertexNormals();

	// Pass 2: vertex colors (needs normals for slope detection)
	const normals = geometry.attributes.normal;
	for (let i = 0; i < vertexCount; i++) {
		const wx = positions.getX(i);
		const wz = positions.getZ(i);
		const h = positions.getY(i);
		const gx = wx + cols / 2 - 0.5;
		const gz = wz + rows / 2 - 0.5;

		// Blended biome color
		let [r, g, b] = blendColor(grid, gx, gz);

		// Slope-based exposed rock
		const ny = normals.getY(i);
		const slope = 1 - Math.abs(ny);
		if (slope > 0.25) {
			const t = Math.min(1, (slope - 0.25) / 0.35);
			r = r * (1 - t) + ROCK_COL[0] * t;
			g = g * (1 - t) + ROCK_COL[1] * t;
			b = b * (1 - t) + ROCK_COL[2] * t;
		}

		// Height-based snow caps
		if (h > SNOW_START) {
			const snowT = Math.min(1, (h - SNOW_START) / (SNOW_FULL - SNOW_START));
			const snowF = snowT * Math.max(0, ny * ny); // less snow on steep
			r = r * (1 - snowF) + SNOW_COL[0] * snowF;
			g = g * (1 - snowF) + SNOW_COL[1] * snowF;
			b = b * (1 - snowF) + SNOW_COL[2] * snowF;
		}

		// Wet sand near waterline
		if (h > -0.15 && h < 0.12) {
			const wetT = 1.0 - Math.max(0, Math.min(1, (h + 0.15) / 0.27));
			r = r * (1 - wetT * 0.6) + WET_SAND_COL[0] * wetT * 0.6;
			g = g * (1 - wetT * 0.6) + WET_SAND_COL[1] * wetT * 0.6;
			b = b * (1 - wetT * 0.6) + WET_SAND_COL[2] * wetT * 0.6;
		}

		// Noise-driven color variation for natural look
		const cn = (warpedFbm(gx * 2.5, gz * 2.5, 3) - 0.5) * 0.07;
		colors[i * 3]     = Math.max(0, Math.min(1, r + cn));
		colors[i * 3 + 1] = Math.max(0, Math.min(1, g + cn * 0.7));
		colors[i * 3 + 2] = Math.max(0, Math.min(1, b + cn * 0.5));
	}

	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

	// Normal map for micro-detail texture
	const normalMap = createNormalMap(grid, cols, rows);

	const material = new THREE.MeshStandardMaterial({
		vertexColors: true,
		roughness: 0.82,
		metalness: 0.02,
		normalMap,
		normalScale: new THREE.Vector2(1.0, 1.0),
		flatShading: false,
		envMapIntensity: 0.5,
	});

	const mesh = new THREE.Mesh(geometry, material);
	mesh.receiveShadow = true;
	mesh.castShadow = true;

	/** Height lookup — must match vertex computation exactly */
	function getHeightAt(worldX: number, worldZ: number): number {
		const gx = worldX + cols / 2 - 0.5;
		const gz = worldZ + rows / 2 - 0.5;
		return computeHeight(grid, mtnField, cols, rows, gx, gz);
	}

	return {
		mesh,
		getHeightAt,
		dispose() {
			geometry.dispose();
			material.dispose();
			normalMap.dispose();
		}
	};
}
