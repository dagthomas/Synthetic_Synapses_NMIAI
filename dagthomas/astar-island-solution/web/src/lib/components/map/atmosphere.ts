/**
 * Atmospheric enhancements for first-person mode:
 * - Player-following fill light (warm/cool based on time)
 * - Ground-level grass particle billboards
 * - Mist sprites near water/ocean cells
 * - Dust motes in the air
 */
import * as THREE from 'three';
import { mulberry32 } from './prng';

export interface AtmosphereSystem {
	playerLight: THREE.PointLight;
	grassPoints: THREE.Points | null;
	mistPoints: THREE.Points | null;
	dustPoints: THREE.Points | null;
	update(camera: THREE.PerspectiveCamera, dt: number, timeOfDay: number): void;
	dispose(): void;
}

function createGrassParticles(
	grid: number[][], ox: number, oz: number,
	heightFn: (x: number, z: number) => number
): THREE.Points {
	const rng = mulberry32(7777);
	const rows = grid.length, cols = grid[0].length;

	const grassCells: { x: number; z: number }[] = [];
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			const code = grid[y][x];
			if (code === 4 || code === 11 || code === 0) {
				grassCells.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
			}
		}
	}

	const count = Math.min(grassCells.length * 8, 5000);
	const positions = new Float32Array(count * 3);
	const colors = new Float32Array(count * 3);

	for (let i = 0; i < count; i++) {
		const cell = grassCells[Math.floor(rng() * grassCells.length)];
		const px = cell.x + (rng() - 0.5) * 0.95;
		const pz = cell.z + (rng() - 0.5) * 0.95;
		const py = heightFn(px, pz) + 0.01 + rng() * 0.05;

		positions[i * 3] = px;
		positions[i * 3 + 1] = py;
		positions[i * 3 + 2] = pz;

		// Varied greens
		colors[i * 3]     = 0.15 + rng() * 0.15;
		colors[i * 3 + 1] = 0.30 + rng() * 0.30;
		colors[i * 3 + 2] = 0.05 + rng() * 0.10;
	}

	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

	const mat = new THREE.PointsMaterial({
		vertexColors: true, size: 0.055, sizeAttenuation: true,
		transparent: true, opacity: 0.75, depthWrite: false
	});

	return new THREE.Points(geo, mat);
}

/** Low-lying mist near water cells */
function createMistParticles(
	grid: number[][], ox: number, oz: number,
	heightFn: (x: number, z: number) => number
): THREE.Points {
	const rng = mulberry32(3333);
	const rows = grid.length, cols = grid[0].length;

	const waterCells: { x: number; z: number }[] = [];
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			if (grid[y][x] === 10 || grid[y][x] === 2) {
				waterCells.push({ x: x + ox + 0.5, z: y + oz + 0.5 });
			}
		}
	}

	const count = Math.min(waterCells.length * 4, 1500);
	const positions = new Float32Array(count * 3);
	// Store original positions for animation
	const origPositions = new Float32Array(count * 3);

	for (let i = 0; i < count; i++) {
		const cell = waterCells[Math.floor(rng() * waterCells.length)];
		const px = cell.x + (rng() - 0.5) * 1.5;
		const pz = cell.z + (rng() - 0.5) * 1.5;
		const py = heightFn(px, pz) + 0.05 + rng() * 0.15;

		positions[i * 3] = origPositions[i * 3] = px;
		positions[i * 3 + 1] = origPositions[i * 3 + 1] = py;
		positions[i * 3 + 2] = origPositions[i * 3 + 2] = pz;
	}

	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	// Store originals as custom attribute for drift reference
	(geo as any)._origPositions = origPositions;

	const mat = new THREE.PointsMaterial({
		color: 0xccddee, size: 0.18, sizeAttenuation: true,
		transparent: true, opacity: 0.15, depthWrite: false,
		blending: THREE.AdditiveBlending
	});

	return new THREE.Points(geo, mat);
}

/** Floating dust motes in the air */
function createDustParticles(): THREE.Points {
	const rng = mulberry32(9999);
	const count = 800;
	const positions = new Float32Array(count * 3);

	for (let i = 0; i < count; i++) {
		positions[i * 3]     = (rng() - 0.5) * 30;
		positions[i * 3 + 1] = rng() * 2.0 + 0.1;
		positions[i * 3 + 2] = (rng() - 0.5) * 30;
	}

	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	const mat = new THREE.PointsMaterial({
		color: 0xffeedd, size: 0.012, sizeAttenuation: true,
		transparent: true, opacity: 0.3, depthWrite: false,
		blending: THREE.AdditiveBlending
	});

	return new THREE.Points(geo, mat);
}

export function createAtmosphere(
	scene: THREE.Scene,
	grid: number[][],
	heightFn: (x: number, z: number) => number
): AtmosphereSystem {
	const rows = grid.length, cols = grid[0].length;
	const ox = -cols / 2, oz = -rows / 2;

	// Player fill light
	const playerLight = new THREE.PointLight(0xffeedd, 0.5, 10, 1.2);
	playerLight.castShadow = false;
	scene.add(playerLight);

	const grassPoints = createGrassParticles(grid, ox, oz, heightFn);
	scene.add(grassPoints);

	const mistPoints = createMistParticles(grid, ox, oz, heightFn);
	scene.add(mistPoints);

	const dustPoints = createDustParticles();
	scene.add(dustPoints);

	const _camDir = new THREE.Vector3();

	return {
		playerLight, grassPoints, mistPoints, dustPoints,

		update(camera: THREE.PerspectiveCamera, dt: number, timeOfDay: number) {
			// Player fill light follows camera
			camera.getWorldDirection(_camDir);
			playerLight.position.copy(camera.position);
			playerLight.position.y += 0.2;
			playerLight.position.addScaledVector(_camDir, -0.4);

			const isNight = timeOfDay < 6 || timeOfDay > 18;
			const isDawnDusk = (timeOfDay >= 5 && timeOfDay <= 7) || (timeOfDay >= 17 && timeOfDay <= 19);

			if (isNight) {
				playerLight.intensity = 1.0;
				playerLight.color.setHex(0x8899cc);
				playerLight.distance = 12;
			} else if (isDawnDusk) {
				playerLight.intensity = 0.6;
				playerLight.color.setHex(0xffbb66);
				playerLight.distance = 10;
			} else {
				playerLight.intensity = 0.35;
				playerLight.color.setHex(0xffeedd);
				playerLight.distance = 8;
			}

			const time = performance.now() / 1000;

			// Mist drift animation
			if (mistPoints) {
				const pos = mistPoints.geometry.attributes.position;
				const orig = (mistPoints.geometry as any)._origPositions as Float32Array;
				for (let i = 0; i < pos.count; i++) {
					const ox2 = orig[i * 3], oz2 = orig[i * 3 + 2], oy = orig[i * 3 + 1];
					pos.setX(i, ox2 + Math.sin(time * 0.3 + ox2 * 2) * 0.15);
					pos.setZ(i, oz2 + Math.cos(time * 0.25 + oz2 * 1.5) * 0.12);
					pos.setY(i, oy + Math.sin(time * 0.5 + i * 0.1) * 0.03);
				}
				pos.needsUpdate = true;

				// Mist thicker at dawn/dusk, thin at noon
				const mistMat = mistPoints.material as THREE.PointsMaterial;
				if (isDawnDusk) mistMat.opacity = 0.30;
				else if (isNight) mistMat.opacity = 0.20;
				else mistMat.opacity = 0.10;
			}

			// Dust motes: follow camera loosely, gentle float
			if (dustPoints) {
				const pos = dustPoints.geometry.attributes.position;
				for (let i = 0; i < pos.count; i++) {
					let y = pos.getY(i);
					y += Math.sin(time * 0.8 + i * 0.7) * 0.0008;
					// Keep dust near camera
					let x = pos.getX(i), z = pos.getZ(i);
					const dx = x - camera.position.x, dz = z - camera.position.z;
					if (dx * dx + dz * dz > 225) { // 15^2
						x = camera.position.x + (Math.random() - 0.5) * 20;
						z = camera.position.z + (Math.random() - 0.5) * 20;
						y = camera.position.y - 0.3 + Math.random() * 1.5;
					}
					pos.setXYZ(i, x, y, z);
				}
				pos.needsUpdate = true;

				// Dust visible in sunbeams (day), less at night
				const dustMat = dustPoints.material as THREE.PointsMaterial;
				if (isDawnDusk) dustMat.opacity = 0.5;
				else if (isNight) dustMat.opacity = 0.15;
				else dustMat.opacity = 0.35;
			}
		},

		dispose() {
			scene.remove(playerLight); playerLight.dispose();
			for (const pts of [grassPoints, mistPoints, dustPoints]) {
				if (pts) {
					scene.remove(pts);
					pts.geometry.dispose();
					(pts.material as THREE.Material).dispose();
				}
			}
		}
	};
}
