/**
 * Tessellated heightmap terrain for first-person mode.
 * Multi-octave FBM noise, bilinear blending, high-res procedural normal maps.
 */
import * as THREE from 'three';
import { mulberry32 } from './prng';

export interface TerrainSystem {
	mesh: THREE.Mesh;
	getHeightAt(worldX: number, worldZ: number): number;
	dispose(): void;
}

// Heights with more dramatic variation for epic ground-level feel
const HEIGHTS: Record<number, number> = {
	0: 0.05,   // Sandy
	1: 0.15,   // Settlement: raised platform feel
	2: -0.08,  // Port: below waterline
	3: 0.12,   // Ruin: uneven
	4: 0.14,   // Forest: root mounds
	5: 0.40,   // Mountain base: dramatically raised
	10: -0.15, // Ocean: deep depression
	11: 0.08   // Plains
};

// Richer colors for ground-level viewing
const COLORS: Record<number, [number, number, number]> = {
	0:  [0.78, 0.71, 0.48],  // warm sand
	1:  [0.72, 0.62, 0.38],  // worn earth
	2:  [0.22, 0.55, 0.72],  // port blue-gray
	3:  [0.58, 0.54, 0.48],  // mossy stone
	4:  [0.28, 0.50, 0.18],  // deep forest green
	5:  [0.50, 0.48, 0.44],  // granite gray
	10: [0.18, 0.48, 0.66],  // ocean deep
	11: [0.42, 0.68, 0.24]   // lush plains
};

const DEFAULT_COLOR: [number, number, number] = [0.78, 0.71, 0.48];

function getHeight(code: number): number { return HEIGHTS[code] ?? 0.05; }
function getColor(code: number): [number, number, number] { return COLORS[code] ?? DEFAULT_COLOR; }

function sampleBilinear(grid: number[][], gx: number, gy: number, fn: (c: number) => number): number {
	const rows = grid.length, cols = grid[0].length;
	const cx = Math.max(0, Math.min(cols - 1.001, gx));
	const cy = Math.max(0, Math.min(rows - 1.001, gy));
	const ix = Math.floor(cx), iy = Math.floor(cy);
	const fx = cx - ix, fy = cy - iy;
	const ix1 = Math.min(ix + 1, cols - 1), iy1 = Math.min(iy + 1, rows - 1);
	return fn(grid[iy][ix]) * (1-fx)*(1-fy) + fn(grid[iy][ix1]) * fx*(1-fy) +
		fn(grid[iy1][ix]) * (1-fx)*fy + fn(grid[iy1][ix1]) * fx*fy;
}

function sampleColorBilinear(grid: number[][], gx: number, gy: number): [number, number, number] {
	const rows = grid.length, cols = grid[0].length;
	const cx = Math.max(0, Math.min(cols - 1.001, gx));
	const cy = Math.max(0, Math.min(rows - 1.001, gy));
	const ix = Math.floor(cx), iy = Math.floor(cy);
	const fx = cx - ix, fy = cy - iy;
	const ix1 = Math.min(ix + 1, cols - 1), iy1 = Math.min(iy + 1, rows - 1);
	const c00 = getColor(grid[iy][ix]), c10 = getColor(grid[iy][ix1]);
	const c01 = getColor(grid[iy1][ix]), c11 = getColor(grid[iy1][ix1]);
	return [
		c00[0]*(1-fx)*(1-fy) + c10[0]*fx*(1-fy) + c01[0]*(1-fx)*fy + c11[0]*fx*fy,
		c00[1]*(1-fx)*(1-fy) + c10[1]*fx*(1-fy) + c01[1]*(1-fx)*fy + c11[1]*fx*fy,
		c00[2]*(1-fx)*(1-fy) + c10[2]*fx*(1-fy) + c01[2]*(1-fx)*fy + c11[2]*fx*fy
	];
}

/** Multi-octave FBM noise for natural terrain */
function fbmNoise(x: number, z: number, octaves: number, amplitude: number, lacunarity: number): number {
	let value = 0;
	let amp = amplitude;
	let freq = 1.0;
	for (let i = 0; i < octaves; i++) {
		value += Math.sin(x * freq * 7.3 + z * freq * 13.1 + i * 31.7) * amp;
		value += Math.cos(x * freq * 11.9 + z * freq * 5.7 + i * 17.3) * amp * 0.7;
		amp *= 0.5;
		freq *= lacunarity;
	}
	return value;
}

/** Per-terrain-type noise amplitude — mountains get chunky, plains get smooth */
function terrainNoiseScale(code: number): number {
	switch (code) {
		case 5: return 0.06;  // Mountain: rough
		case 4: return 0.025; // Forest: root bumps
		case 3: return 0.03;  // Ruin: rubble
		case 10: return 0.008; // Ocean: gentle waves
		case 11: return 0.012; // Plains: subtle
		default: return 0.015;
	}
}

/** High-resolution procedural normal map with multi-frequency detail */
function createNormalMap(grid: number[][], cols: number, rows: number): THREE.CanvasTexture {
	const res = 1024;
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
			const terrainType = grid[iy][ix];
			const idx = (py * res + px) * 4;

			let nx = 0, ny = 0;

			// Multi-frequency detail per terrain type
			switch (terrainType) {
				case 4: // Forest: clumpy grass + leaf litter
					nx = Math.sin(px*0.35 + py*0.12)*0.30 + Math.sin(px*0.9 + py*0.7)*0.15 + Math.sin(px*2.3 + py*1.1)*0.08;
					ny = Math.cos(py*0.3 + px*0.15)*0.28 + Math.cos(py*0.85 + px*0.6)*0.12 + Math.cos(py*2.1 + px*1.4)*0.08;
					break;
				case 5: // Mountain: aggressive rocky detail
					nx = (Math.sin(px*0.5 + py*0.15) + Math.sin(px*1.6 + py*0.8))*0.35 + Math.sin(px*3.5 + py*1.9)*0.18 + Math.sin(px*7.1)*0.06;
					ny = (Math.cos(py*0.45 + px*0.25) + Math.cos(py*1.4 + px*0.6))*0.35 + Math.cos(py*3.2 + px*1.5)*0.18 + Math.cos(py*6.3)*0.06;
					break;
				case 10: // Ocean: rolling wave ripples
					nx = Math.sin(px*0.10 + py*0.03)*0.15 + Math.sin(px*0.3 + py*0.08)*0.08;
					ny = Math.cos(py*0.09 + px*0.025)*0.15 + Math.cos(py*0.25 + px*0.07)*0.08;
					break;
				case 11: // Plains: wind-swept grass waves
					nx = Math.sin(px*0.18 + py*0.06)*0.14 + Math.sin(px*0.5)*0.08 + Math.sin(px*1.8 + py*0.4)*0.04;
					ny = Math.cos(py*0.15 + px*0.05)*0.14 + Math.cos(py*0.45)*0.08 + Math.cos(py*1.6 + px*0.3)*0.04;
					break;
				case 1: // Settlement: cobblestone + wear
					nx = Math.sin(px*1.2)*Math.cos(py*1.5)*0.22 + Math.sin(px*3.5 + py*2.1)*0.08;
					ny = Math.cos(px*1.4)*Math.sin(py*1.3)*0.22 + Math.cos(py*3.3 + px*1.8)*0.08;
					break;
				case 3: // Ruin: cracked stone + rubble
					nx = Math.sin(px*0.6 + py*1.8)*0.25 + Math.sin(px*2.5 + py*0.5)*0.12;
					ny = Math.cos(py*0.7 + px*1.6)*0.25 + Math.cos(py*2.3 + px*0.6)*0.12;
					break;
				default: // Sandy: fine grain texture
					nx = Math.sin(px*0.7 + py*0.4)*0.08 + Math.sin(px*2.5 + py*1.5)*0.04;
					ny = Math.cos(py*0.6 + px*0.35)*0.08 + Math.cos(py*2.2 + px*1.3)*0.04;
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

export function createTerrain(grid: number[][]): TerrainSystem {
	const rows = grid.length;
	const cols = grid[0].length;

	// 8 subdivisions per cell — smooth enough for ground-level
	const subdivs = 8;
	const segX = cols * subdivs;
	const segZ = rows * subdivs;

	const geometry = new THREE.PlaneGeometry(cols, rows, segX, segZ);
	geometry.rotateX(-Math.PI / 2);

	const positions = geometry.attributes.position;
	const vertexCount = positions.count;
	const colors = new Float32Array(vertexCount * 3);

	for (let i = 0; i < vertexCount; i++) {
		const wx = positions.getX(i);
		const wz = positions.getZ(i);
		const gx = wx + cols / 2 - 0.5;
		const gz = wz + rows / 2 - 0.5;

		// Get terrain type for noise scaling
		const tix = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
		const tiy = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
		const terrainCode = grid[tiy]?.[tix] ?? 0;

		const h = sampleBilinear(grid, gx, gz, getHeight);
		const noiseAmp = terrainNoiseScale(terrainCode);
		const noise = fbmNoise(gx, gz, 4, noiseAmp, 2.1);
		positions.setY(i, h + noise);

		const [r, g, b] = sampleColorBilinear(grid, gx, gz);
		// Color variation: per-terrain noise + subtle warm/cool shift
		const cn = fbmNoise(gx, gz, 2, 0.035, 2.5);
		colors[i * 3]     = Math.max(0, Math.min(1, r + cn + 0.02));
		colors[i * 3 + 1] = Math.max(0, Math.min(1, g + cn * 0.6));
		colors[i * 3 + 2] = Math.max(0, Math.min(1, b + cn * 0.4 - 0.01));
	}

	positions.needsUpdate = true;
	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
	geometry.computeVertexNormals();

	const normalMap = createNormalMap(grid, cols, rows);

	const material = new THREE.MeshStandardMaterial({
		vertexColors: true,
		roughness: 0.85,
		metalness: 0.01,
		normalMap,
		normalScale: new THREE.Vector2(1.0, 1.0),
		flatShading: false,
		envMapIntensity: 0.4
	});

	const mesh = new THREE.Mesh(geometry, material);
	mesh.receiveShadow = true;
	mesh.castShadow = false;

	function getTerrainHeight(worldX: number, worldZ: number): number {
		const gx = worldX + cols / 2 - 0.5;
		const gz = worldZ + rows / 2 - 0.5;
		const tix = Math.floor(Math.max(0, Math.min(cols - 1, gx + 0.5)));
		const tiy = Math.floor(Math.max(0, Math.min(rows - 1, gz + 0.5)));
		const code = grid[tiy]?.[tix] ?? 0;
		const h = sampleBilinear(grid, gx, gz, getHeight);
		return h + fbmNoise(gx, gz, 4, terrainNoiseScale(code), 2.1);
	}

	return {
		mesh,
		getHeightAt: getTerrainHeight,
		dispose() {
			geometry.dispose();
			material.dispose();
			normalMap.dispose();
		}
	};
}
