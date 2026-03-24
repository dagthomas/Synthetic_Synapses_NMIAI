/**
 * Web Worker for computing terrain morph data off the main thread.
 * Contains all pure math functions (no THREE.js dependency).
 */

const WATER_LEVEL = -0.12;
const SNOW_START = 1.6;
const SNOW_FULL = 2.8;

const BASE_HEIGHTS: Record<number, number> = {
	0: 0.12, 1: 0.16, 2: 0.04, 3: 0.16, 4: 0.20, 5: 0.36, 10: -0.55, 11: 0.14,
};

const BIOME_COLORS: Record<number, [number, number, number]> = {
	0: [0.82, 0.74, 0.50], 1: [0.70, 0.60, 0.38], 2: [0.45, 0.55, 0.52],
	3: [0.56, 0.52, 0.44], 4: [0.20, 0.44, 0.14], 5: [0.50, 0.48, 0.44],
	10: [0.10, 0.25, 0.42], 11: [0.36, 0.62, 0.18],
};

const SNOW_COL: [number, number, number] = [0.94, 0.94, 0.97];
const WET_SAND_COL: [number, number, number] = [0.58, 0.52, 0.38];
const DEFAULT_COL: [number, number, number] = [0.65, 0.58, 0.42];

const NOISE_AMP: Record<number, number> = {
	0: 0.018, 1: 0.012, 2: 0.010, 3: 0.030, 4: 0.035, 5: 0.12, 10: 0.008, 11: 0.020,
};

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

function warpedFbm(x: number, z: number, octaves: number): number {
	const wx = fbm(x + 5.2, z + 1.3, 3, 2.0, 0.5) * 1.5;
	const wz = fbm(x + 1.7, z + 9.2, 3, 2.0, 0.5) * 1.5;
	return fbm(x + wx, z + wz, octaves, 2.0, 0.5);
}

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

function blendBaseHeight(grid: number[][], gx: number, gz: number): number {
	const rows = grid.length, cols = grid[0].length;
	const sigma2 = 1.8;
	let total = 0, weight = 0;
	for (let dy = -2; dy <= 2; dy++) {
		for (let dx = -2; dx <= 2; dx++) {
			const cx = Math.round(gx) + dx;
			const cy = Math.round(gz) + dy;
			if (cx < 0 || cx >= cols || cy < 0 || cy >= rows) continue;
			const d2 = (gx - cx) * (gx - cx) + (gz - cy) * (gz - cy);
			const w = Math.exp(-d2 / sigma2);
			total += (BASE_HEIGHTS[grid[cy][cx]] ?? 0.06) * w;
			weight += w;
		}
	}
	return weight > 0 ? total / weight : 0.06;
}

function computeHeight(grid: number[][], mtnField: Float32Array, cols: number, rows: number, gx: number, gz: number): number {
	let h = blendBaseHeight(grid, gx, gz);
	const tix = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
	const tiy = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
	const code = grid[tiy]?.[tix] ?? 0;

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
			mtnPeak = Math.max(mtnPeak, (1 - t * t) * (1 - t * t));
		}
	}
	if (mtnPeak > 0) {
		const density = sampleMtnField(mtnField, cols, rows, gx, gz);
		const peakScale = density >= 12 ? 4.0 : density >= 6 ? 3.2 : density >= 3 ? 2.2 : 1.5;
		h += mtnPeak * peakScale;
	}

	const amp = NOISE_AMP[code] ?? 0.015;
	h += (warpedFbm(gx * 0.8, gz * 0.8, 5) - 0.5) * 2 * amp;
	if (mtnPeak > 0.2) h += ridgeNoise(gx * 0.5, gz * 0.5, 4) * mtnPeak * 0.6;
	return h;
}

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
			r += c[0] * wt; g += c[1] * wt; b += c[2] * wt;
			w += wt;
		}
	}
	return w > 0 ? [r / w, g / w, b / w] : [...DEFAULT_COL];
}

type RoadPath = { points: [number, number][]; width: number };

const ROAD_COLS: Record<number, [number, number, number]> = {
	0: [0.60, 0.50, 0.30], 1: [0.38, 0.34, 0.30], 2: [0.45, 0.38, 0.28],
	3: [0.40, 0.36, 0.32], 4: [0.35, 0.28, 0.15], 5: [0.42, 0.40, 0.38],
	10: [0.60, 0.50, 0.30], 11: [0.42, 0.32, 0.18],
};
const DEFAULT_ROAD_COL: [number, number, number] = [0.40, 0.32, 0.20];

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

// --- Worker message handler ---
self.onmessage = (e: MessageEvent<{ grid: number[][]; season?: string; roads?: RoadPath[] }>) => {
	const { grid, roads = [] } = e.data;
	const rows = grid.length;
	const cols = grid[0].length;
	const mtnField = buildMountainField(grid);
	const subdivs = 4;
	const segX = cols * subdivs;
	const segZ = rows * subdivs;
	const vertexCount = (segX + 1) * (segZ + 1);
	const heights = new Float32Array(vertexCount);
	const colors = new Float32Array(vertexCount * 3);
	const roadBlendArr = roads.length > 0 ? new Float32Array(vertexCount) : null;

	for (let iz = 0; iz <= segZ; iz++) {
		for (let ix = 0; ix <= segX; ix++) {
			const i = iz * (segX + 1) + ix;
			const wx = (ix / segX - 0.5) * cols;
			const wz = (iz / segZ - 0.5) * rows;
			const gx = wx + cols / 2 - 0.5;
			const gz = wz + rows / 2 - 0.5;
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

			let [r, g, b] = blendColor(grid, gx, gz);
			if (h > SNOW_START) {
				const snowT = Math.min(1, (h - SNOW_START) / (SNOW_FULL - SNOW_START));
				r = r * (1 - snowT) + SNOW_COL[0] * snowT;
				g = g * (1 - snowT) + SNOW_COL[1] * snowT;
				b = b * (1 - snowT) + SNOW_COL[2] * snowT;
			}
			if (h > -0.15 && h < 0.12) {
				const wetT = 1.0 - Math.max(0, Math.min(1, (h + 0.15) / 0.27));
				r = r * (1 - wetT * 0.6) + WET_SAND_COL[0] * wetT * 0.6;
				g = g * (1 - wetT * 0.6) + WET_SAND_COL[1] * wetT * 0.6;
				b = b * (1 - wetT * 0.6) + WET_SAND_COL[2] * wetT * 0.6;
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
			const cn = (warpedFbm(gx * 2.5, gz * 2.5, 3) - 0.5) * 0.07;
			colors[i * 3]     = Math.max(0, Math.min(1, r + cn));
			colors[i * 3 + 1] = Math.max(0, Math.min(1, g + cn * 0.7));
			colors[i * 3 + 2] = Math.max(0, Math.min(1, b + cn * 0.5));
		}
	}

	// Transfer ownership of buffers (zero-copy)
	self.postMessage(
		{ heights, colors, vertexCount },
		{ transfer: [heights.buffer, colors.buffer] } as any
	);
};
