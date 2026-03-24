// Boids flocking system — separation, alignment, cohesion
// Low-poly 3D birds with wing flap animation via InstancedMesh

import * as THREE from 'three';
import { mulberry32 } from './prng';

const BIRD_COUNT = 50;
const VISUAL_RANGE = 4.0;
const PROTECTED_RANGE = 1.2;
const CENTERING = 0.003;
const AVOID = 0.05;
const MATCHING = 0.04;
const MAX_SPEED = 4.0;
const MIN_SPEED = 1.5;
const TURN_FACTOR = 0.3;
const BOUNDARY_RADIUS = 18;
const MIN_HEIGHT = 2.0;
const MAX_HEIGHT = 8.0;
const BIRD_SCALE = 2.0;

interface BirdState {
	px: number; py: number; pz: number;
	vx: number; vy: number; vz: number;
	flapOffset: number;
	flapSpeed: number;
}

export interface WildlifeSystem {
	update(dt: number, timeOfDay: number): void;
	dispose(): void;
	group: THREE.Group;
}

/** Create a simple V-shaped bird mesh (body + wings as one geometry) */
function createBirdGeometry(): THREE.BufferGeometry {
	const v = new Float32Array([
		// Body — elongated diamond
		 0.00,  0.00,  0.12,   // nose
		-0.03,  0.01, -0.06,   // left back
		 0.03,  0.01, -0.06,   // right back
		 0.00,  0.00,  0.12,   // nose bottom
		 0.03, -0.01, -0.06,   // right back bottom
		-0.03, -0.01, -0.06,   // left back bottom
		// Tail
		 0.00,  0.02, -0.06,
		-0.02,  0.00, -0.12,
		 0.02,  0.00, -0.12,
	]);
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(v, 3));
	geo.computeVertexNormals();
	return geo;
}

function createWingGeometry(side: number): THREE.BufferGeometry {
	// Flat angular wing — wider, more visible
	const tipX = side * 0.22;
	const midX = side * 0.12;
	const v = new Float32Array([
		// Main wing triangle
		0.00,  0.005,  0.04,    // root front
		tipX,  0.02,  -0.01,    // tip
		0.00,  0.005, -0.04,    // root back
		// Secondary feather
		midX,  0.01,  -0.02,
		tipX,  0.02,  -0.01,
		0.00,  0.005, -0.04,
	]);
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(v, 3));
	geo.computeVertexNormals();
	return geo;
}

export function createWildlifeSystem(
	scene: THREE.Scene,
	heightFn?: (x: number, z: number) => number
): WildlifeSystem {
	const rng = mulberry32(7777);
	const group = new THREE.Group();

	// Initialize birds in circular orbits
	const birds: BirdState[] = [];
	for (let i = 0; i < BIRD_COUNT; i++) {
		const angle = rng() * Math.PI * 2;
		const radius = 4 + rng() * (BOUNDARY_RADIUS - 6);
		const speed = MIN_SPEED + rng() * (MAX_SPEED - MIN_SPEED) * 0.5;
		const dir = angle + Math.PI * 0.5;
		birds.push({
			px: Math.cos(angle) * radius,
			py: MIN_HEIGHT + rng() * (MAX_HEIGHT - MIN_HEIGHT),
			pz: Math.sin(angle) * radius,
			vx: Math.cos(dir) * speed,
			vy: (rng() - 0.5) * 0.3,
			vz: Math.sin(dir) * speed,
			flapOffset: rng() * Math.PI * 2,
			flapSpeed: 5 + rng() * 4,
		});
	}

	// Geometry + materials
	const bodyGeo = createBirdGeometry();
	const leftWingGeo = createWingGeometry(-1);
	const rightWingGeo = createWingGeometry(1);

	const bodyMat = new THREE.MeshBasicMaterial({
		color: 0x222222, side: THREE.DoubleSide,
	});
	const wingMat = new THREE.MeshBasicMaterial({
		color: 0x111111, side: THREE.DoubleSide,
	});

	const bodyInst = new THREE.InstancedMesh(bodyGeo, bodyMat, BIRD_COUNT);
	const leftWingInst = new THREE.InstancedMesh(leftWingGeo, wingMat, BIRD_COUNT);
	const rightWingInst = new THREE.InstancedMesh(rightWingGeo, wingMat, BIRD_COUNT);
	bodyInst.castShadow = true;

	// Initialize all instance matrices to identity so they render on first frame
	const identityMat = new THREE.Matrix4();
	for (let i = 0; i < BIRD_COUNT; i++) {
		bodyInst.setMatrixAt(i, identityMat);
		leftWingInst.setMatrixAt(i, identityMat);
		rightWingInst.setMatrixAt(i, identityMat);
	}
	bodyInst.instanceMatrix.needsUpdate = true;
	leftWingInst.instanceMatrix.needsUpdate = true;
	rightWingInst.instanceMatrix.needsUpdate = true;

	group.add(bodyInst, leftWingInst, rightWingInst);
	scene.add(group);

	// Reusable math objects
	const _dummy = new THREE.Object3D();
	const _target = new THREE.Vector3();
	const _wingDummy = new THREE.Object3D();

	let elapsed = 0;

	function updateBoids(dt: number) {
		const dt2 = Math.min(dt, 0.05);

		for (let i = 0; i < BIRD_COUNT; i++) {
			const b = birds[i];
			let closeDx = 0, closeDy = 0, closeDz = 0;
			let avgVx = 0, avgVy = 0, avgVz = 0;
			let avgPx = 0, avgPy = 0, avgPz = 0;
			let neighbors = 0;

			for (let j = 0; j < BIRD_COUNT; j++) {
				if (i === j) continue;
				const o = birds[j];
				const dx = b.px - o.px, dy = b.py - o.py, dz = b.pz - o.pz;
				const dist2 = dx * dx + dy * dy + dz * dz;

				if (dist2 < PROTECTED_RANGE * PROTECTED_RANGE) {
					closeDx += dx; closeDy += dy; closeDz += dz;
				} else if (dist2 < VISUAL_RANGE * VISUAL_RANGE) {
					avgVx += o.vx; avgVy += o.vy; avgVz += o.vz;
					avgPx += o.px; avgPy += o.py; avgPz += o.pz;
					neighbors++;
				}
			}

			b.vx += closeDx * AVOID;
			b.vy += closeDy * AVOID;
			b.vz += closeDz * AVOID;

			if (neighbors > 0) {
				b.vx += (avgVx / neighbors - b.vx) * MATCHING;
				b.vy += (avgVy / neighbors - b.vy) * MATCHING;
				b.vz += (avgVz / neighbors - b.vz) * MATCHING;
				b.vx += (avgPx / neighbors - b.px) * CENTERING;
				b.vy += (avgPy / neighbors - b.py) * CENTERING;
				b.vz += (avgPz / neighbors - b.pz) * CENTERING;
			}

			// Boundary
			const distXZ = Math.sqrt(b.px * b.px + b.pz * b.pz);
			if (distXZ > BOUNDARY_RADIUS) {
				b.vx -= (b.px / distXZ) * TURN_FACTOR;
				b.vz -= (b.pz / distXZ) * TURN_FACTOR;
			}
			if (b.py < MIN_HEIGHT) b.vy += TURN_FACTOR;
			if (b.py > MAX_HEIGHT) b.vy -= TURN_FACTOR;

			// Terrain avoidance — push up and steer away from mountains
			if (heightFn) {
				const groundH = heightFn(b.px, b.pz);
				const clearance = b.py - groundH;
				const MIN_CLEARANCE = 1.5;
				if (clearance < MIN_CLEARANCE) {
					// Strong upward push proportional to how close we are
					const urgency = 1 - clearance / MIN_CLEARANCE;
					b.vy += urgency * TURN_FACTOR * 4;
					// Also check ahead and steer away
					const aheadX = b.px + b.vx * 0.5;
					const aheadZ = b.pz + b.vz * 0.5;
					const aheadH = heightFn(aheadX, aheadZ);
					if (aheadH > groundH) {
						// Terrain rising ahead — steer away from it
						b.vx -= (aheadX - b.px) * TURN_FACTOR * 2;
						b.vz -= (aheadZ - b.pz) * TURN_FACTOR * 2;
					}
				}
			}

			// Speed clamp
			const speed = Math.sqrt(b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
			if (speed > MAX_SPEED) {
				const s = MAX_SPEED / speed;
				b.vx *= s; b.vy *= s; b.vz *= s;
			} else if (speed < MIN_SPEED && speed > 0.01) {
				const s = MIN_SPEED / speed;
				b.vx *= s; b.vy *= s; b.vz *= s;
			}

			b.px += b.vx * dt2;
			b.py += b.vy * dt2;
			b.pz += b.vz * dt2;
		}
	}

	function updateMeshes(time: number) {
		for (let i = 0; i < BIRD_COUNT; i++) {
			const b = birds[i];

			// Body — look along velocity
			_dummy.position.set(b.px, b.py, b.pz);
			_target.set(b.px + b.vx, b.py + b.vy, b.pz + b.vz);
			_dummy.lookAt(_target);
			_dummy.scale.setScalar(BIRD_SCALE);
			_dummy.updateMatrix();
			bodyInst.setMatrixAt(i, _dummy.matrix);

			// Wing flap
			const flap = Math.sin(time * b.flapSpeed + b.flapOffset) * 0.7;

			// Left wing — same position/orientation as body, plus Z rotation for flap
			_wingDummy.position.copy(_dummy.position);
			_wingDummy.rotation.copy(_dummy.rotation);
			_wingDummy.rotateZ(flap);
			_wingDummy.scale.setScalar(BIRD_SCALE);
			_wingDummy.updateMatrix();
			leftWingInst.setMatrixAt(i, _wingDummy.matrix);

			// Right wing — opposite flap
			_wingDummy.rotation.copy(_dummy.rotation);
			_wingDummy.rotateZ(-flap);
			_wingDummy.updateMatrix();
			rightWingInst.setMatrixAt(i, _wingDummy.matrix);
		}

		bodyInst.instanceMatrix.needsUpdate = true;
		leftWingInst.instanceMatrix.needsUpdate = true;
		rightWingInst.instanceMatrix.needsUpdate = true;
	}

	// Initialize meshes at spawn positions
	updateMeshes(0);

	return {
		group,

		update(dt: number, timeOfDay: number) {
			elapsed += dt;
			const isNight = timeOfDay < 5 || timeOfDay > 20;
			group.visible = !isNight;

			if (!isNight) {
				updateBoids(dt);
				updateMeshes(elapsed);
			}
		},

		dispose() {
			bodyGeo.dispose();
			leftWingGeo.dispose();
			rightWingGeo.dispose();
			bodyMat.dispose();
			wingMat.dispose();
			scene.remove(group);
		}
	};
}
