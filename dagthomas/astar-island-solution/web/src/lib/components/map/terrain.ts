/**
 * Low-poly procedural terrain — flat-shaded heightmap with seasonal vertex colors.
 * Subdivs=2 per cell, no normal map, reduced noise for fast rebuilds.
 * Mountains rise 1.5–4 units, ocean sinks below water level.
 */
import * as THREE from 'three';

export type Season = 'spring' | 'summer' | 'autumn' | 'winter';

export interface TerrainSystem {
	mesh: THREE.Mesh;
	getHeightAt(worldX: number, worldZ: number): number;
	dispose(): void;
}

// ═══════════════════════════════════════
//  Constants
// ═══════════════════════════════════════

export const WATER_LEVEL = -0.12;

const SNOW_THRESHOLDS: Record<Season, { start: number; full: number }> = {
	spring: { start: 2.0, full: 3.2 },
	summer: { start: 2.4, full: 3.5 },
	autumn: { start: 1.8, full: 2.8 },
	winter: { start: 0.8, full: 1.6 },
};

const BASE_HEIGHTS: Record<number, number> = {
	0:  0.12,   // Sandy
	1:  0.16,   // Settlement
	2:  0.04,   // Port
	3:  0.16,   // Ruin
	4:  0.20,   // Forest
	5:  0.36,   // Mountain base
	10: -0.55,  // Ocean
	11: 0.14,   // Plains
};

const SEASONAL_BIOME_COLORS: Record<Season, Record<number, [number, number, number]>> = {
	summer: {
		0:  [0.82, 0.74, 0.50], 1:  [0.70, 0.60, 0.38], 2:  [0.45, 0.55, 0.52],
		3:  [0.56, 0.52, 0.44], 4:  [0.20, 0.44, 0.14], 5:  [0.50, 0.48, 0.44],
		10: [0.10, 0.25, 0.42], 11: [0.36, 0.62, 0.18],
	},
	spring: {
		0:  [0.82, 0.74, 0.50], 1:  [0.68, 0.60, 0.40], 2:  [0.45, 0.55, 0.52],
		3:  [0.52, 0.52, 0.42], 4:  [0.28, 0.52, 0.18], 5:  [0.50, 0.48, 0.44],
		10: [0.10, 0.26, 0.44], 11: [0.42, 0.65, 0.22],
	},
	autumn: {
		0:  [0.80, 0.72, 0.48], 1:  [0.68, 0.56, 0.36], 2:  [0.44, 0.52, 0.50],
		3:  [0.58, 0.50, 0.40], 4:  [0.55, 0.35, 0.12], 5:  [0.50, 0.48, 0.44],
		10: [0.10, 0.22, 0.38], 11: [0.62, 0.52, 0.20],
	},
	winter: {
		0:  [0.78, 0.74, 0.58], 1:  [0.64, 0.58, 0.46], 2:  [0.48, 0.54, 0.54],
		3:  [0.54, 0.52, 0.48], 4:  [0.30, 0.28, 0.24], 5:  [0.58, 0.56, 0.54],
		10: [0.08, 0.20, 0.38], 11: [0.45, 0.40, 0.32],
	},
};

const SNOW_COL: [number, number, number] = [0.94, 0.94, 0.97];
const ROCK_COL: [number, number, number] = [0.46, 0.44, 0.40];
const WET_SAND_COL: [number, number, number] = [0.58, 0.52, 0.38];
const DEFAULT_COL: [number, number, number] = [0.65, 0.58, 0.42];

const NOISE_AMP: Record<number, number> = {
	0: 0.018, 1: 0.012, 2: 0.010, 3: 0.030,
	4: 0.035, 5: 0.12, 10: 0.008, 11: 0.020,
};

const SUBDIVS = 2; // low-poly: 2 subdivisions per grid cell

function getBiomeColors(season: Season): Record<number, [number, number, number]> {
	return SEASONAL_BIOME_COLORS[season];
}

// ═══════════════════════════════════════
//  Noise (reduced octaves for speed)
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
		maxAmp += amp; amp *= gain; freq *= lacunarity;
	}
	return value / maxAmp;
}

function ridgeNoise(x: number, z: number): number {
	let value = 0, amp = 1, freq = 1, maxAmp = 0;
	for (let i = 0; i < 2; i++) {
		let n = smoothNoise(x * freq, z * freq);
		n = 1.0 - Math.abs(n * 2 - 1); n *= n;
		value += n * amp; maxAmp += amp; amp *= 0.5; freq *= 2.1;
	}
	return value / maxAmp;
}

/** Simple FBM — no domain warping for speed */
function simpleFbm(x: number, z: number): number {
	return fbm(x, z, 2, 2.0, 0.5);
}

// ═══════════════════════════════════════
//  Height computation (3x3 kernels)
// ═══════════════════════════════════════

function buildMountainField(grid: number[][]): Float32Array {
	const rows = grid.length, cols = grid[0].length;
	const field = new Float32Array(rows * cols);
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			let count = 0;
			for (let dy = -3; dy <= 3; dy++) {
				for (let dx = -3; dx <= 3; dx++) {
					const cx = x + dx, cy = y + dy;
					if (cx >= 0 && cx < cols && cy >= 0 && cy < rows && grid[cy][cx] === 5) count++;
				}
			}
			field[y * cols + x] = count;
		}
	}
	return field;
}

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

/** 3x3 Gaussian-blended base height */
function blendBaseHeight(grid: number[][], gx: number, gz: number): number {
	const rows = grid.length, cols = grid[0].length;
	const sigma2 = 1.8;
	let total = 0, weight = 0;
	for (let dy = -1; dy <= 1; dy++) {
		for (let dx = -1; dx <= 1; dx++) {
			const cx = Math.round(gx) + dx, cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			const d2 = (gx - cx) * (gx - cx) + (gz - cy) * (gz - cy);
			const w = Math.exp(-d2 / sigma2);
			total += (BASE_HEIGHTS[grid[cy][cx]] ?? 0.06) * w;
			weight += w;
		}
	}
	return weight > 0 ? total / weight : 0.06;
}

function computeHeight(
	grid: number[][], mtnField: Float32Array,
	cols: number, rows: number, gx: number, gz: number
): number {
	let h = blendBaseHeight(grid, gx, gz);
	const tix = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
	const tiy = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
	const code = grid[tiy]?.[tix] ?? 0;

	let mtnPeak = 0;
	for (let dy = -3; dy <= 3; dy++) {
		for (let dx = -3; dx <= 3; dx++) {
			const cx = Math.round(gx) + dx, cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			if (grid[cy][cx] !== 5) continue;
			const dist = Math.sqrt((gx - cx) * (gx - cx) + (gz - cy) * (gz - cy));
			const radius = 2.8;
			if (dist >= radius) continue;
			const t = dist / radius;
			mtnPeak = Math.max(mtnPeak, (1 - t * t) * (1 - t * t));
		}
	}
	if (mtnPeak > 0) {
		const density = sampleMtnField(mtnField, cols, rows, gx, gz);
		const peakScale = density >= 12 ? 4.0 : density >= 6 ? 3.2 : density >= 3 ? 2.2 : 1.5;
		h += mtnPeak * peakScale;
	}

	const amp = NOISE_AMP[code] ?? 0.015;
	h += (simpleFbm(gx * 0.8, gz * 0.8) - 0.5) * 2 * amp;
	if (mtnPeak > 0.2) h += ridgeNoise(gx * 0.5, gz * 0.5) * mtnPeak * 0.6;
	return h;
}

// ═══════════════════════════════════════
//  Color (3x3 Gaussian, flat per-face)
// ═══════════════════════════════════════

function blendColor(grid: number[][], gx: number, gz: number, biomeColors: Record<number, [number, number, number]>): [number, number, number] {
	const rows = grid.length, cols = grid[0].length;
	const sigma2 = 1.4;
	let r = 0, g = 0, b = 0, w = 0;
	for (let dy = -1; dy <= 1; dy++) {
		for (let dx = -1; dx <= 1; dx++) {
			const cx = Math.round(gx) + dx, cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			const d2 = (gx - cx) * (gx - cx) + (gz - cy) * (gz - cy);
			const wt = Math.exp(-d2 / sigma2);
			const c = biomeColors[grid[cy][cx]] ?? DEFAULT_COL;
			r += c[0] * wt; g += c[1] * wt; b += c[2] * wt; w += wt;
		}
	}
	return w > 0 ? [r / w, g / w, b / w] : [...DEFAULT_COL];
}

// ═══════════════════════════════════════
//  Exports
// ═══════════════════════════════════════

export function computeTerrainMorphData(grid: number[][], season: Season = 'summer', roads: RoadPath[] = []): { heights: Float32Array; colors: Float32Array; vertexCount: number } {
	const rows = grid.length, cols = grid[0].length;
	const mtnField = buildMountainField(grid);
	const biomeColors = getBiomeColors(season);
	const snow = SNOW_THRESHOLDS[season];
	const segX = cols * SUBDIVS, segZ = rows * SUBDIVS;
	const vertexCount = (segX + 1) * (segZ + 1);
	const heights = new Float32Array(vertexCount);
	const colors = new Float32Array(vertexCount * 3);
	const roadBlendArr = roads.length > 0 ? new Float32Array(vertexCount) : null;

	for (let iz = 0; iz <= segZ; iz++) {
		for (let ix = 0; ix <= segX; ix++) {
			const i = iz * (segX + 1) + ix;
			const wx = (ix / segX - 0.5) * cols, wz = (iz / segZ - 0.5) * rows;
			const gx = wx + cols / 2 - 0.5, gz = wz + rows / 2 - 0.5;
			let h = computeHeight(grid, mtnField, cols, rows, gx, gz);

			if (roadBlendArr && roads.length > 0) {
				const rd = roadDistance(wx, wz, roads);
				if (rd.roadIdx >= 0) {
					const hw = roads[rd.roadIdx].width / 2;
					if (rd.dist < hw) {
						const blend = 1 - rd.dist / hw;
						const smooth = blend * blend * (3 - 2 * blend);
						roadBlendArr[i] = smooth;
						h -= smooth * 0.06;
					}
				}
			}
			heights[i] = h;

			let [r, g, b] = blendColor(grid, gx, gz, biomeColors);
			if (h > snow.start) {
				const t = Math.min(1, (h - snow.start) / (snow.full - snow.start));
				r = r * (1 - t) + SNOW_COL[0] * t; g = g * (1 - t) + SNOW_COL[1] * t; b = b * (1 - t) + SNOW_COL[2] * t;
			}
			if (h > -0.15 && h < 0.12) {
				const t = 1.0 - Math.max(0, Math.min(1, (h + 0.15) / 0.27));
				r = r * (1 - t * 0.6) + WET_SAND_COL[0] * t * 0.6;
				g = g * (1 - t * 0.6) + WET_SAND_COL[1] * t * 0.6;
				b = b * (1 - t * 0.6) + WET_SAND_COL[2] * t * 0.6;
			}
			if (roadBlendArr) {
				const rb = roadBlendArr[i];
				if (rb > 0.01) {
					const cellX = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
					const cellZ = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
					const rc = ROAD_COLS[grid[cellZ]?.[cellX] ?? 0] ?? DEFAULT_ROAD_COL;
					const strength = Math.min(1, rb * 1.5);
					r = r * (1 - strength) + rc[0] * strength;
					g = g * (1 - strength) + rc[1] * strength;
					b = b * (1 - strength) + rc[2] * strength;
				}
			}
			const cn = (simpleFbm(gx * 2.5, gz * 2.5) - 0.5) * 0.05;
			colors[i * 3] = Math.max(0, Math.min(1, r + cn));
			colors[i * 3 + 1] = Math.max(0, Math.min(1, g + cn * 0.7));
			colors[i * 3 + 2] = Math.max(0, Math.min(1, b + cn * 0.5));
		}
	}
	return { heights, colors, vertexCount };
}

export type RoadPath = { points: [number, number][]; width: number };

function roadDistance(wx: number, wz: number, roads: RoadPath[]): { dist: number; roadIdx: number } {
	let bestDist = Infinity, bestIdx = -1;
	for (let ri = 0; ri < roads.length; ri++) {
		const pts = roads[ri].points;
		for (let j = 0; j < pts.length - 1; j++) {
			const [ax, az] = pts[j], [bx, bz] = pts[j + 1];
			const dx = bx - ax, dz = bz - az, len2 = dx * dx + dz * dz;
			if (len2 < 0.0001) continue;
			const t = Math.max(0, Math.min(1, ((wx - ax) * dx + (wz - az) * dz) / len2));
			const d = Math.sqrt((wx - ax - dx * t) ** 2 + (wz - az - dz * t) ** 2);
			if (d < bestDist) { bestDist = d; bestIdx = ri; }
		}
	}
	return { dist: bestDist, roadIdx: bestIdx };
}

const ROAD_COLS: Record<number, [number, number, number]> = {
	0: [0.60, 0.50, 0.30], 1: [0.38, 0.34, 0.30], 2: [0.45, 0.38, 0.28],
	3: [0.40, 0.36, 0.32], 4: [0.35, 0.28, 0.15], 5: [0.42, 0.40, 0.38],
	10: [0.60, 0.50, 0.30], 11: [0.42, 0.32, 0.18],
};
const DEFAULT_ROAD_COL: [number, number, number] = [0.40, 0.32, 0.20];

export interface CreateTerrainOptions {
	roads?: RoadPath[];
	season?: Season;
	skipNormalMap?: boolean; // kept for API compat, ignored (no normal map)
}

export function createTerrain(grid: number[][], options: CreateTerrainOptions = {}): TerrainSystem {
	const { roads = [], season = 'summer' } = options;
	const rows = grid.length, cols = grid[0].length;
	const biomeColors = getBiomeColors(season);
	const snow = SNOW_THRESHOLDS[season];
	const mtnField = buildMountainField(grid);
	const segX = cols * SUBDIVS, segZ = rows * SUBDIVS;

	const geometry = new THREE.PlaneGeometry(cols, rows, segX, segZ);
	geometry.rotateX(-Math.PI / 2);

	const positions = geometry.attributes.position;
	const vertexCount = positions.count;
	const colors = new Float32Array(vertexCount * 3);

	// Pass 1: heights + road indentation
	const roadBlend = new Float32Array(vertexCount);
	for (let i = 0; i < vertexCount; i++) {
		const wx = positions.getX(i), wz = positions.getZ(i);
		const gx = wx + cols / 2 - 0.5, gz = wz + rows / 2 - 0.5;
		let h = computeHeight(grid, mtnField, cols, rows, gx, gz);

		if (roads.length > 0) {
			const rd = roadDistance(wx, wz, roads);
			if (rd.roadIdx >= 0) {
				const hw = roads[rd.roadIdx].width / 2;
				if (rd.dist < hw) {
					const blend = 1 - rd.dist / hw;
					const smooth = blend * blend * (3 - 2 * blend);
					roadBlend[i] = smooth;
					h -= smooth * 0.06;
				}
			}
		}
		positions.setY(i, h);
	}
	positions.needsUpdate = true;
	geometry.computeVertexNormals();

	// Pass 2: vertex colors
	const normals = geometry.attributes.normal;
	for (let i = 0; i < vertexCount; i++) {
		const wx = positions.getX(i), wz = positions.getZ(i), h = positions.getY(i);
		const gx = wx + cols / 2 - 0.5, gz = wz + rows / 2 - 0.5;

		let [r, g, b] = blendColor(grid, gx, gz, biomeColors);

		// Slope rock
		const ny = normals.getY(i);
		const slope = 1 - Math.abs(ny);
		if (slope > 0.25) {
			const t = Math.min(1, (slope - 0.25) / 0.35);
			r = r * (1 - t) + ROCK_COL[0] * t; g = g * (1 - t) + ROCK_COL[1] * t; b = b * (1 - t) + ROCK_COL[2] * t;
		}

		// Snow
		if (h > snow.start) {
			const t = Math.min(1, (h - snow.start) / (snow.full - snow.start)) * Math.max(0, ny * ny);
			r = r * (1 - t) + SNOW_COL[0] * t; g = g * (1 - t) + SNOW_COL[1] * t; b = b * (1 - t) + SNOW_COL[2] * t;
		}

		// Wet sand
		if (h > -0.15 && h < 0.12) {
			const t = (1.0 - Math.max(0, Math.min(1, (h + 0.15) / 0.27))) * 0.6;
			r = r * (1 - t) + WET_SAND_COL[0] * t; g = g * (1 - t) + WET_SAND_COL[1] * t; b = b * (1 - t) + WET_SAND_COL[2] * t;
		}

		// Road
		const rb = roadBlend[i];
		if (rb > 0.01) {
			const cellX = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
			const cellZ = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
			const rc = ROAD_COLS[grid[cellZ]?.[cellX] ?? 0] ?? DEFAULT_ROAD_COL;
			const strength = Math.min(1, rb * 1.5);
			r = r * (1 - strength) + rc[0] * strength;
			g = g * (1 - strength) + rc[1] * strength;
			b = b * (1 - strength) + rc[2] * strength;
		}

		// Slight noise variation
		const cn = (simpleFbm(gx * 2.5, gz * 2.5) - 0.5) * 0.05;
		colors[i * 3]     = Math.max(0, Math.min(1, r + cn));
		colors[i * 3 + 1] = Math.max(0, Math.min(1, g + cn * 0.7));
		colors[i * 3 + 2] = Math.max(0, Math.min(1, b + cn * 0.5));
	}

	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

	const material = new THREE.MeshStandardMaterial({
		vertexColors: true,
		roughness: 0.85,
		metalness: 0.02,
		flatShading: true,
		envMapIntensity: 0.4,
	});

	const mesh = new THREE.Mesh(geometry, material);
	mesh.receiveShadow = true;
	mesh.castShadow = true;

	function getHeightAt(worldX: number, worldZ: number): number {
		const gx = worldX + cols / 2 - 0.5, gz = worldZ + rows / 2 - 0.5;
		return computeHeight(grid, mtnField, cols, rows, gx, gz);
	}

	return {
		mesh, getHeightAt,
		dispose() { geometry.dispose(); material.dispose(); }
	};
}

// No-op for API compat — normal map removed
export function applyNormalMap(_terrain: TerrainSystem, _grid: number[][]): void {}

/** Update getHeightAt on existing TerrainSystem to use new grid data (no mesh rebuild) */
export function updateTerrainHeightFn(terrain: TerrainSystem, grid: number[][]): void {
	const rows = grid.length, cols = grid[0].length;
	const mtnField = buildMountainField(grid);
	(terrain as any).getHeightAt = (worldX: number, worldZ: number): number => {
		const gx = worldX + cols / 2 - 0.5, gz = worldZ + rows / 2 - 0.5;
		return computeHeight(grid, mtnField, cols, rows, gx, gz);
	};
}

export function getCurrentSeason(): Season {
	const month = new Date().getMonth();
	if (month >= 2 && month <= 4) return 'spring';
	if (month >= 5 && month <= 7) return 'summer';
	if (month >= 8 && month <= 10) return 'autumn';
	return 'winter';
}

export function getSeasonalTreeTint(season: Season): THREE.Color {
	switch (season) {
		case 'spring': return new THREE.Color(0.85, 1.0, 0.75);
		case 'summer': return new THREE.Color(1.0, 1.0, 1.0);
		case 'autumn': return new THREE.Color(1.0, 0.6, 0.25);
		case 'winter': return new THREE.Color(0.65, 0.65, 0.65);
	}
}
