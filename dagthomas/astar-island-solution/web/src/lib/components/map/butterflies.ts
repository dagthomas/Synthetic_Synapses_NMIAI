// Butterfly clusters — 2-3 butterflies fluttering around flower positions
// Procedural geometry (two triangular wings), no GLB needed

import * as THREE from 'three';
import { mulberry32 } from './prng';

const CLUSTER_COUNT = 5;
const BUTTERFLIES_PER_CLUSTER = [2, 3, 2, 3, 2];
const WING_SCALE = 0.06;
const FLUTTER_SPEED_MIN = 8;
const FLUTTER_SPEED_MAX = 14;
const ORBIT_RADIUS = 0.3;
const ORBIT_SPEED_MIN = 0.4;
const ORBIT_SPEED_MAX = 1.2;
const HOVER_HEIGHT = 0.15;
const HOVER_BOB = 0.08;

// Butterfly color palettes
const PALETTES: [number, number, number][] = [
	[1.0, 0.6, 0.1],   // orange monarch
	[0.3, 0.5, 1.0],   // blue morpho
	[1.0, 1.0, 0.3],   // yellow sulphur
	[0.9, 0.3, 0.6],   // pink
	[1.0, 0.95, 0.9],  // white cabbage
	[0.6, 0.2, 0.8],   // purple emperor
];

interface ButterflyState {
	wingL: THREE.Mesh;
	wingR: THREE.Mesh;
	centerX: number;
	centerY: number;
	centerZ: number;
	orbitAngle: number;
	orbitSpeed: number;
	orbitRadius: number;
	flutterPhase: number;
	flutterSpeed: number;
	bobPhase: number;
	facing: number;
}

export interface ButterflySystem {
	group: THREE.Group;
	update(dt: number): void;
	dispose(): void;
}

function createWingGeometry(side: number): THREE.BufferGeometry {
	// Two-triangle wing: inner triangle + outer triangle
	const s = side;
	const verts = new Float32Array([
		// Inner wing
		0, 0, 0,
		s * 0.04, 0.01, -0.015,
		s * 0.02, -0.005, 0.02,
		// Outer wing
		s * 0.04, 0.01, -0.015,
		s * 0.06, 0.005, 0.005,
		s * 0.02, -0.005, 0.02,
	]);
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
	geo.computeVertexNormals();
	return geo;
}

export function createButterflySystem(
	flowerPositions: { x: number; y: number; z: number }[],
	seed = 8888
): ButterflySystem {
	const group = new THREE.Group();
	const rng = mulberry32(seed);
	const butterflies: ButterflyState[] = [];

	if (flowerPositions.length === 0) {
		return { group, update() {}, dispose() {} };
	}

	// Pick 5 spread-out flower positions
	const shuffled = [...flowerPositions].sort(() => rng() - 0.5);
	const chosen: { x: number; y: number; z: number }[] = [];
	for (const pos of shuffled) {
		if (chosen.length >= CLUSTER_COUNT) break;
		// Ensure clusters are spread apart (min 2 units)
		const tooClose = chosen.some(c =>
			(c.x - pos.x) ** 2 + (c.z - pos.z) ** 2 < 4
		);
		if (!tooClose) chosen.push(pos);
	}
	// Fill remaining with random picks if not enough spread positions
	while (chosen.length < CLUSTER_COUNT && chosen.length < flowerPositions.length) {
		chosen.push(shuffled[chosen.length % shuffled.length]);
	}

	const wingLGeo = createWingGeometry(1);
	const wingRGeo = createWingGeometry(-1);

	for (let c = 0; c < chosen.length; c++) {
		const center = chosen[c];
		const count = BUTTERFLIES_PER_CLUSTER[c % BUTTERFLIES_PER_CLUSTER.length];

		for (let i = 0; i < count; i++) {
			const palette = PALETTES[Math.floor(rng() * PALETTES.length)];
			const color = new THREE.Color(palette[0], palette[1], palette[2]);

			const mat = new THREE.MeshBasicMaterial({
				color,
				side: THREE.DoubleSide,
				transparent: true,
				opacity: 0.85,
			});

			const wingL = new THREE.Mesh(wingLGeo, mat);
			const wingR = new THREE.Mesh(wingRGeo, mat.clone());
			(wingR.material as THREE.MeshBasicMaterial).color.copy(color).multiplyScalar(0.85);

			const scale = WING_SCALE * (0.8 + rng() * 0.4);
			wingL.scale.setScalar(scale);
			wingR.scale.setScalar(scale);

			group.add(wingL, wingR);

			butterflies.push({
				wingL,
				wingR,
				centerX: center.x + (rng() - 0.5) * 0.3,
				centerY: center.y + HOVER_HEIGHT,
				centerZ: center.z + (rng() - 0.5) * 0.3,
				orbitAngle: rng() * Math.PI * 2,
				orbitSpeed: ORBIT_SPEED_MIN + rng() * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN),
				orbitRadius: ORBIT_RADIUS * (0.6 + rng() * 0.8),
				flutterPhase: rng() * Math.PI * 2,
				flutterSpeed: FLUTTER_SPEED_MIN + rng() * (FLUTTER_SPEED_MAX - FLUTTER_SPEED_MIN),
				bobPhase: rng() * Math.PI * 2,
				facing: rng() * Math.PI * 2,
			});
		}
	}

	return {
		group,

		update(dt: number) {
			for (const b of butterflies) {
				// Orbit around flower
				b.orbitAngle += b.orbitSpeed * dt;
				const px = b.centerX + Math.cos(b.orbitAngle) * b.orbitRadius;
				const pz = b.centerZ + Math.sin(b.orbitAngle) * b.orbitRadius;

				// Vertical bob
				b.bobPhase += dt * 2.5;
				const py = b.centerY + Math.sin(b.bobPhase) * HOVER_BOB;

				// Face orbit direction (tangent)
				b.facing = b.orbitAngle + Math.PI * 0.5;

				// Wing flutter
				b.flutterPhase += dt * b.flutterSpeed;
				const flapAngle = Math.sin(b.flutterPhase) * 0.7; // ±40 degrees

				// Position both wings at same spot
				b.wingL.position.set(px, py, pz);
				b.wingR.position.set(px, py, pz);

				// Face direction + wing flap
				b.wingL.rotation.set(0, b.facing, flapAngle);
				b.wingR.rotation.set(0, b.facing, -flapAngle);
			}
		},

		dispose() {
			wingLGeo.dispose();
			wingRGeo.dispose();
			for (const b of butterflies) {
				(b.wingL.material as THREE.Material).dispose();
				(b.wingR.material as THREE.Material).dispose();
			}
		},
	};
}
