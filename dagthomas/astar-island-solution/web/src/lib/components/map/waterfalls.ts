// Edge waterfall system — water flowing off the floating island slab into the void

import * as THREE from 'three';
import { mulberry32 } from './prng';

const PARTICLES_PER_FALL = 80;
const SLAB_TOP = -0.2;
const SLAB_BOTTOM = -3.5;
const VOID_BOTTOM = -8;
const GRAVITY = 4.0;

export interface WaterfallSystem {
	points: THREE.Points;
	mist: THREE.Points | null;
	update(dt: number): void;
	dispose(): void;
}

interface Particle {
	x: number;
	y: number;
	z: number;
	vy: number;
	vx: number;
	vz: number;
	baseX: number;
	baseZ: number;
	edgeDirX: number;
	edgeDirZ: number;
}

export interface WaterfallSource {
	x: number;
	z: number;
	dirX: number; // outward direction from island center
	dirZ: number;
}

/**
 * Find waterfall source positions around the slab perimeter.
 * Picks spots where ocean meets land at the edge, water cascades outward.
 */
export function findWaterfallSources(
	grid: number[][],
	offsetX: number,
	offsetZ: number
): WaterfallSource[] {
	const rows = grid.length;
	const cols = grid[0]?.length ?? 0;
	const OCEAN = 10;
	const candidates: WaterfallSource[] = [];
	const cx = cols / 2;
	const cz = rows / 2;

	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			// Only ocean cells on the perimeter or adjacent to perimeter
			if (grid[y][x] !== OCEAN) continue;

			const onPerimeter = x <= 1 || x >= cols - 2 || y <= 1 || y >= rows - 2;
			if (!onPerimeter) continue;

			// Direction outward from center
			const dx = x - cx;
			const dz = y - cz;
			const len = Math.sqrt(dx * dx + dz * dz) || 1;

			candidates.push({
				x: x + offsetX + 0.5,
				z: y + offsetZ + 0.5,
				dirX: dx / len,
				dirZ: dz / len
			});
		}
	}

	// Also add some non-ocean edge cells for "earth runoff" waterfalls
	for (let y = 0; y < rows; y++) {
		for (let x = 0; x < cols; x++) {
			if (grid[y][x] === OCEAN) continue;
			const onEdge = x === 0 || x === cols - 1 || y === 0 || y === rows - 1;
			if (!onEdge) continue;

			// Only some edge cells (every ~5th cell)
			if ((x + y) % 5 !== 0) continue;

			const dx = x - cx;
			const dz = y - cz;
			const len = Math.sqrt(dx * dx + dz * dz) || 1;

			candidates.push({
				x: x + offsetX + 0.5,
				z: y + offsetZ + 0.5,
				dirX: dx / len,
				dirZ: dz / len
			});
		}
	}

	// Sample up to 12 waterfall sources
	if (candidates.length <= 12) return candidates;
	const rng = mulberry32(42);
	// Shuffle and take first 12
	for (let i = candidates.length - 1; i > 0; i--) {
		const j = Math.floor(rng() * (i + 1));
		[candidates[i], candidates[j]] = [candidates[j], candidates[i]];
	}
	return candidates.slice(0, 12);
}

export function createWaterfallSystem(
	scene: THREE.Scene,
	sources: WaterfallSource[]
): WaterfallSystem | null {
	if (sources.length === 0) return null;

	const totalParticles = sources.length * PARTICLES_PER_FALL;
	const geometry = new THREE.BufferGeometry();
	const positions = new Float32Array(totalParticles * 3);
	const alphas = new Float32Array(totalParticles);

	const particles: Particle[] = [];
	const rng = mulberry32(7777);

	for (const src of sources) {
		for (let i = 0; i < PARTICLES_PER_FALL; i++) {
			const progress = rng(); // random start along fall path
			const p: Particle = {
				baseX: src.x,
				baseZ: src.z,
				edgeDirX: src.dirX,
				edgeDirZ: src.dirZ,
				x: src.x + src.dirX * rng() * 1.5 + (rng() - 0.5) * 0.3,
				z: src.z + src.dirZ * rng() * 1.5 + (rng() - 0.5) * 0.3,
				y: SLAB_TOP - progress * (SLAB_TOP - VOID_BOTTOM),
				vy: -(0.5 + rng() * 2),
				vx: src.dirX * (0.2 + rng() * 0.5),
				vz: src.dirZ * (0.2 + rng() * 0.5)
			};
			particles.push(p);
		}
	}

	for (let i = 0; i < particles.length; i++) {
		positions[i * 3] = particles[i].x;
		positions[i * 3 + 1] = particles[i].y;
		positions[i * 3 + 2] = particles[i].z;
		alphas[i] = 1.0;
	}

	geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

	const material = new THREE.PointsMaterial({
		color: 0x88ccff,
		size: 0.1,
		transparent: true,
		opacity: 0.7,
		depthWrite: false,
		sizeAttenuation: true
	});

	const points = new THREE.Points(geometry, material);
	scene.add(points);

	// Mist particles at the bottom where water disappears
	const mistCount = sources.length * 15;
	const mistGeo = new THREE.BufferGeometry();
	const mistPos = new Float32Array(mistCount * 3);
	const mistParticles: { x: number; y: number; z: number; life: number; maxLife: number }[] = [];

	for (let i = 0; i < sources.length; i++) {
		for (let j = 0; j < 15; j++) {
			const p = {
				x: sources[i].x + sources[i].dirX * 1.5 + (rng() - 0.5) * 2,
				y: SLAB_BOTTOM + rng() * 1.5 - 0.5,
				z: sources[i].z + sources[i].dirZ * 1.5 + (rng() - 0.5) * 2,
				life: rng() * 3,
				maxLife: 2 + rng() * 2
			};
			mistParticles.push(p);
			const idx = i * 15 + j;
			mistPos[idx * 3] = p.x;
			mistPos[idx * 3 + 1] = p.y;
			mistPos[idx * 3 + 2] = p.z;
		}
	}
	mistGeo.setAttribute('position', new THREE.BufferAttribute(mistPos, 3));

	const mistMat = new THREE.PointsMaterial({
		color: 0xaaddff,
		size: 0.3,
		transparent: true,
		opacity: 0.25,
		depthWrite: false,
		sizeAttenuation: true
	});
	const mist = new THREE.Points(mistGeo, mistMat);
	scene.add(mist);

	return {
		points,
		mist,
		update(dt: number) {
			const pos = geometry.attributes.position as THREE.BufferAttribute;

			for (let i = 0; i < particles.length; i++) {
				const p = particles[i];
				p.vy -= GRAVITY * dt;
				p.y += p.vy * dt;
				p.x += p.vx * dt;
				p.z += p.vz * dt;

				// Reset when deep into the void
				if (p.y < VOID_BOTTOM) {
					p.y = SLAB_TOP + Math.random() * 0.3;
					p.vy = -(0.3 + Math.random() * 0.5);
					p.x = p.baseX + (Math.random() - 0.5) * 0.3;
					p.z = p.baseZ + (Math.random() - 0.5) * 0.3;
					p.vx = p.edgeDirX * (0.2 + Math.random() * 0.5);
					p.vz = p.edgeDirZ * (0.2 + Math.random() * 0.5);
				}

				pos.setXYZ(i, p.x, p.y, p.z);
			}
			pos.needsUpdate = true;

			// Animate mist
			if (mist) {
				const mPos = mistGeo.attributes.position as THREE.BufferAttribute;
				for (let i = 0; i < mistParticles.length; i++) {
					const mp = mistParticles[i];
					mp.life += dt;
					mp.y += dt * 0.3; // slowly rises
					if (mp.life > mp.maxLife) {
						mp.life = 0;
						mp.y = SLAB_BOTTOM + Math.random() * 0.5 - 0.5;
					}
					mPos.setXYZ(i, mp.x + Math.sin(mp.life * 2) * 0.2, mp.y, mp.z);
				}
				mPos.needsUpdate = true;
			}
		},
		dispose() {
			geometry.dispose();
			material.dispose();
			scene.remove(points);
			if (mist) {
				mistGeo.dispose();
				mistMat.dispose();
				scene.remove(mist);
			}
		}
	};
}
