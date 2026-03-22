/**
 * Tessellated heightmap terrain for first-person mode.
 * Replaces blocky BoxGeometry grid with a smooth, bilinear-interpolated
 * PlaneGeometry with vertex colors and procedural normal maps.
 */
import * as THREE from 'three';
import { mulberry32 } from './prng';

export interface TerrainSystem {
	mesh: THREE.Mesh;
	/** Get interpolated ground height at world position */
	getHeightAt(worldX: number, worldZ: number): number;
	dispose(): void;
}

// Smooth terrain heights (world-space Y) per terrain code
const HEIGHTS: Record<number, number> = {
	0: 0.06,   // Empty: sandy
	1: 0.14,   // Settlement: slightly raised
	2: -0.06,  // Port: at water level
	3: 0.10,   // Ruin
	4: 0.12,   // Forest: slightly raised
	5: 0.30,   // Mountain base: elevated
	10: -0.12, // Ocean: depressed
	11: 0.08   // Plains
};

// Terrain colors
const COLORS: Record<number, [number, number, number]> = {
	0:  [0.788, 0.722, 0.510], // sandy tan
	1:  [0.769, 0.659, 0.384], // warm tan
	2:  [0.290, 0.620, 0.788], // port blue
	3:  [0.604, 0.565, 0.502], // ruin gray-brown
	4:  [0.353, 0.541, 0.227], // forest green
	5:  [0.541, 0.541, 0.510], // mountain gray
	10: [0.227, 0.533, 0.722], // ocean blue
	11: [0.478, 0.714, 0.282]  // plains green
};

const DEFAULT_COLOR: [number, number, number] = [0.788, 0.722, 0.510];

function getHeight(code: number): number {
	return HEIGHTS[code] ?? 0.06;
}

function getColor(code: number): [number, number, number] {
	return COLORS[code] ?? DEFAULT_COLOR;
}

/** Bilinear interpolation on grid */
function sampleBilinear(
	grid: number[][],
	gx: number,
	gy: number,
	fn: (code: number) => number
): number {
	const rows = grid.length;
	const cols = grid[0].length;

	const cx = Math.max(0, Math.min(cols - 1.001, gx));
	const cy = Math.max(0, Math.min(rows - 1.001, gy));

	const ix = Math.floor(cx);
	const iy = Math.floor(cy);
	const fx = cx - ix;
	const fy = cy - iy;

	const ix1 = Math.min(ix + 1, cols - 1);
	const iy1 = Math.min(iy + 1, rows - 1);

	const v00 = fn(grid[iy][ix]);
	const v10 = fn(grid[iy][ix1]);
	const v01 = fn(grid[iy1][ix]);
	const v11 = fn(grid[iy1][ix1]);

	return (
		v00 * (1 - fx) * (1 - fy) +
		v10 * fx * (1 - fy) +
		v01 * (1 - fx) * fy +
		v11 * fx * fy
	);
}

/** Bilinear color interpolation */
function sampleColorBilinear(
	grid: number[][],
	gx: number,
	gy: number
): [number, number, number] {
	const rows = grid.length;
	const cols = grid[0].length;

	const cx = Math.max(0, Math.min(cols - 1.001, gx));
	const cy = Math.max(0, Math.min(rows - 1.001, gy));

	const ix = Math.floor(cx);
	const iy = Math.floor(cy);
	const fx = cx - ix;
	const fy = cy - iy;

	const ix1 = Math.min(ix + 1, cols - 1);
	const iy1 = Math.min(iy + 1, rows - 1);

	const c00 = getColor(grid[iy][ix]);
	const c10 = getColor(grid[iy][ix1]);
	const c01 = getColor(grid[iy1][ix]);
	const c11 = getColor(grid[iy1][ix1]);

	const r =
		c00[0] * (1 - fx) * (1 - fy) +
		c10[0] * fx * (1 - fy) +
		c01[0] * (1 - fx) * fy +
		c11[0] * fx * fy;
	const g =
		c00[1] * (1 - fx) * (1 - fy) +
		c10[1] * fx * (1 - fy) +
		c01[1] * (1 - fx) * fy +
		c11[1] * fx * fy;
	const b =
		c00[2] * (1 - fx) * (1 - fy) +
		c10[2] * fx * (1 - fy) +
		c01[2] * (1 - fx) * fy +
		c11[2] * fx * fy;

	return [r, g, b];
}

/** Procedural noise for natural terrain variation */
function terrainNoise(x: number, z: number, scale: number, amplitude: number): number {
	return (
		Math.sin(x * scale * 13.7 + z * scale * 7.3) * amplitude +
		Math.sin(x * scale * 5.1 + z * scale * 11.9) * amplitude * 0.6 +
		Math.sin(x * scale * 23.1 + z * scale * 17.7) * amplitude * 0.3
	);
}

/** Generate a procedural normal map based on terrain types */
function createNormalMap(grid: number[][], cols: number, rows: number): THREE.CanvasTexture {
	const res = Math.min(1024, Math.max(512, cols * 16));
	const canvas = document.createElement('canvas');
	canvas.width = res;
	canvas.height = res;
	const ctx = canvas.getContext('2d')!;

	// Start with flat normal (128, 128, 255)
	const imageData = ctx.createImageData(res, res);
	const data = imageData.data;
	const rng = mulberry32(42);

	for (let py = 0; py < res; py++) {
		for (let px = 0; px < res; px++) {
			const gx = (px / res) * cols;
			const gy = (py / res) * rows;
			const ix = Math.floor(Math.min(cols - 1, gx));
			const iy = Math.floor(Math.min(rows - 1, gy));
			const terrainType = grid[iy][ix];

			const idx = (py * res + px) * 4;

			// Base perturbation per terrain type
			let nx = 0;
			let ny = 0;
			const hf = px * 0.08 + py * 0.06; // high-freq coordinate

			switch (terrainType) {
				case 4: // Forest: clumpy grass bumps
					nx =
						Math.sin(px * 0.35 + py * 0.12) * 0.25 +
						Math.sin(px * 0.9 + py * 0.7) * 0.12;
					ny =
						Math.cos(py * 0.3 + px * 0.15) * 0.22 +
						Math.cos(py * 0.85 + px * 0.6) * 0.1;
					break;
				case 5: // Mountain: sharp rocky cracks
					nx =
						(Math.sin(px * 0.6 + py * 0.2) + Math.sin(px * 1.8 + py * 0.9)) * 0.3;
					ny =
						(Math.cos(py * 0.5 + px * 0.3) + Math.cos(py * 1.5 + px * 0.7)) * 0.3;
					// Add crack-like detail
					nx += Math.sin(px * 3.1 + py * 1.7) * 0.15;
					ny += Math.cos(py * 2.9 + px * 1.3) * 0.15;
					break;
				case 10: // Ocean: gentle wave ripples
					nx = Math.sin(px * 0.12 + py * 0.04) * 0.12;
					ny = Math.cos(py * 0.1 + px * 0.03) * 0.12;
					break;
				case 11: // Plains: gentle grass waves
					nx = Math.sin(hf * 2.5) * 0.1 + Math.sin(px * 0.4) * 0.06;
					ny = Math.cos(hf * 2.2 + 0.5) * 0.1 + Math.cos(py * 0.35) * 0.06;
					break;
				case 1: // Settlement: cobblestone-like bumps
					nx = Math.sin(px * 1.5) * Math.cos(py * 1.8) * 0.18;
					ny = Math.cos(px * 1.7) * Math.sin(py * 1.4) * 0.18;
					break;
				case 3: // Ruin: cracked stone
					nx = Math.sin(px * 0.7 + py * 2.1) * 0.2;
					ny = Math.cos(py * 0.8 + px * 1.9) * 0.2;
					break;
				default: // Sandy: fine grain
					nx = Math.sin(px * 0.9 + py * 0.5) * 0.06;
					ny = Math.cos(py * 0.8 + px * 0.4) * 0.06;
					break;
			}

			data[idx] = Math.round(128 + nx * 127);
			data[idx + 1] = Math.round(128 + ny * 127);
			data[idx + 2] = 255;
			data[idx + 3] = 255;
		}
	}

	ctx.putImageData(imageData, 0, 0);
	const texture = new THREE.CanvasTexture(canvas);
	texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
	return texture;
}

/** Build a smooth, tessellated terrain mesh from the grid */
export function createTerrain(grid: number[][]): TerrainSystem {
	const rows = grid.length;
	const cols = grid[0].length;

	// 6 subdivisions per cell for smooth rolling terrain
	const subdivs = 6;
	const segX = cols * subdivs;
	const segZ = rows * subdivs;

	const geometry = new THREE.PlaneGeometry(cols, rows, segX, segZ);
	geometry.rotateX(-Math.PI / 2);

	const positions = geometry.attributes.position;
	const vertexCount = positions.count;
	const colors = new Float32Array(vertexCount * 3);

	// The plane is centered at origin, spanning -cols/2..cols/2, -rows/2..rows/2
	// which matches the blocky terrain's coordinate system

	for (let i = 0; i < vertexCount; i++) {
		const wx = positions.getX(i);
		const wz = positions.getZ(i);

		// Grid coordinates (cell-centered)
		const gx = wx + cols / 2 - 0.5;
		const gz = wz + rows / 2 - 0.5;

		// Bilinear height
		const h = sampleBilinear(grid, gx, gz, getHeight);
		// Add natural noise
		const noise = terrainNoise(gx, gz, 1.0, 0.015);
		positions.setY(i, h + noise);

		// Bilinear vertex color
		const [r, g, b] = sampleColorBilinear(grid, gx, gz);
		// Slight color noise for organic feel
		const cn = terrainNoise(gx, gz, 3.0, 0.03);
		colors[i * 3] = Math.max(0, Math.min(1, r + cn));
		colors[i * 3 + 1] = Math.max(0, Math.min(1, g + cn * 0.7));
		colors[i * 3 + 2] = Math.max(0, Math.min(1, b + cn * 0.5));
	}

	positions.needsUpdate = true;
	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
	geometry.computeVertexNormals();

	const normalMap = createNormalMap(grid, cols, rows);

	const material = new THREE.MeshStandardMaterial({
		vertexColors: true,
		roughness: 0.82,
		metalness: 0.02,
		normalMap: normalMap,
		normalScale: new THREE.Vector2(0.7, 0.7),
		flatShading: false
	});

	const mesh = new THREE.Mesh(geometry, material);
	mesh.receiveShadow = true;
	mesh.castShadow = false;

	return {
		mesh,
		getHeightAt(worldX: number, worldZ: number): number {
			const gx = worldX + cols / 2 - 0.5;
			const gz = worldZ + rows / 2 - 0.5;
			const h = sampleBilinear(grid, gx, gz, getHeight);
			const noise = terrainNoise(gx, gz, 1.0, 0.015);
			return h + noise;
		},
		dispose() {
			geometry.dispose();
			material.dispose();
			normalMap.dispose();
		}
	};
}
