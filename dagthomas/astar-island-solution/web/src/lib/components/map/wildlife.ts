// Living world system: birds, wandering people, settlement activity

import * as THREE from 'three';
import { mulberry32 } from './prng';
import type { Cluster } from './clusters';

// --- Birds (flocking sprites circling the island) ---

const BIRD_COUNT = 20;
const FLOCK_COUNT = 4; // 4 flocks of ~5 birds each

interface Bird {
	flockId: number;
	offset: THREE.Vector3; // offset from flock center
	wingPhase: number;
}

interface Flock {
	center: THREE.Vector3;
	angle: number;
	radius: number;
	height: number;
	speed: number;
	verticalOsc: number;
}

// --- Wandering people (tiny dots moving between settlements) ---

interface Wanderer {
	position: THREE.Vector3;
	target: THREE.Vector3;
	speed: number;
	fromIdx: number;
	toIdx: number;
	progress: number;
}

// --- War parties (red dots moving toward a settlement, with smoke) ---

interface WarParty {
	warriors: THREE.Vector3[];
	target: THREE.Vector3;
	origin: THREE.Vector3;
	progress: number;
	speed: number;
	active: boolean;
	cooldown: number;
}

export interface WildlifeSystem {
	update(dt: number, timeOfDay: number): void;
	dispose(): void;
}

function createBirdTexture(): THREE.Texture {
	const canvas = document.createElement('canvas');
	canvas.width = 32;
	canvas.height = 16;
	const ctx = canvas.getContext('2d')!;
	ctx.clearRect(0, 0, 32, 16);
	// Simple bird shape: two angled lines (wings)
	ctx.strokeStyle = '#222';
	ctx.lineWidth = 2;
	ctx.beginPath();
	ctx.moveTo(2, 10);
	ctx.quadraticCurveTo(8, 3, 16, 8);
	ctx.quadraticCurveTo(24, 3, 30, 10);
	ctx.stroke();
	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	return tex;
}

export function createWildlifeSystem(
	scene: THREE.Scene,
	settlementClusters: Cluster[],
	offsetX: number,
	offsetZ: number
): WildlifeSystem {
	const rng = mulberry32(9999);

	// === Birds ===
	const birdTex = createBirdTexture();
	const birdSprites: THREE.Sprite[] = [];
	const birds: Bird[] = [];
	const flocks: Flock[] = [];

	for (let f = 0; f < FLOCK_COUNT; f++) {
		flocks.push({
			center: new THREE.Vector3(),
			angle: rng() * Math.PI * 2,
			radius: 8 + rng() * 15,
			height: 5 + rng() * 10,
			speed: 0.3 + rng() * 0.4,
			verticalOsc: rng() * Math.PI * 2
		});
	}

	for (let i = 0; i < BIRD_COUNT; i++) {
		const flockId = i % FLOCK_COUNT;
		const mat = new THREE.SpriteMaterial({
			map: birdTex,
			transparent: true,
			depthWrite: false,
			fog: false
		});
		const sprite = new THREE.Sprite(mat);
		sprite.scale.set(0.6, 0.3, 1);
		sprite.renderOrder = 95;
		scene.add(sprite);
		birdSprites.push(sprite);
		birds.push({
			flockId,
			offset: new THREE.Vector3(
				(rng() - 0.5) * 2,
				(rng() - 0.5) * 1,
				(rng() - 0.5) * 2
			),
			wingPhase: rng() * Math.PI * 2
		});
	}

	// === Wandering people (dots between settlements) ===
	const settPositions = settlementClusters.map(c => new THREE.Vector3(
		c.centerX + offsetX + 0.5,
		0.35,
		c.centerY + offsetZ + 0.5
	));

	const wanderers: Wanderer[] = [];
	const wandererPoints: THREE.Points | null = settPositions.length >= 2 ? (() => {
		const count = Math.min(15, settPositions.length * 3);
		for (let i = 0; i < count; i++) {
			const fromIdx = Math.floor(rng() * settPositions.length);
			let toIdx = Math.floor(rng() * settPositions.length);
			if (toIdx === fromIdx) toIdx = (toIdx + 1) % settPositions.length;
			wanderers.push({
				position: settPositions[fromIdx].clone(),
				target: settPositions[toIdx].clone(),
				speed: 0.5 + rng() * 1.0,
				fromIdx,
				toIdx,
				progress: rng()
			});
		}
		const geo = new THREE.BufferGeometry();
		const pos = new Float32Array(count * 3);
		geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		const mat = new THREE.PointsMaterial({
			color: 0xddbb77,
			size: 0.15,
			sizeAttenuation: true,
			transparent: true,
			opacity: 0.9
		});
		const points = new THREE.Points(geo, mat);
		scene.add(points);
		return points;
	})() : null;

	// === War parties ===
	const warParties: WarParty[] = [];
	let warCooldown = 5 + rng() * 10; // seconds until first war
	const warPoints: THREE.Points | null = settPositions.length >= 2 ? (() => {
		const maxWarriors = 30;
		const geo = new THREE.BufferGeometry();
		const pos = new Float32Array(maxWarriors * 3);
		geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
		const mat = new THREE.PointsMaterial({
			color: 0xff3333,
			size: 0.18,
			sizeAttenuation: true,
			transparent: true,
			opacity: 0.9
		});
		const points = new THREE.Points(geo, mat);
		scene.add(points);
		return points;
	})() : null;

	// Smoke sprites for battles
	const smokeSprites: THREE.Sprite[] = [];
	const smokeMat = new THREE.SpriteMaterial({
		color: 0x444444,
		transparent: true,
		opacity: 0,
		depthWrite: false
	});

	function spawnWarParty() {
		if (settPositions.length < 2) return;
		const fromIdx = Math.floor(Math.random() * settPositions.length);
		let toIdx = Math.floor(Math.random() * settPositions.length);
		if (toIdx === fromIdx) toIdx = (toIdx + 1) % settPositions.length;

		const count = 4 + Math.floor(Math.random() * 6);
		const warriors: THREE.Vector3[] = [];
		for (let i = 0; i < count; i++) {
			warriors.push(settPositions[fromIdx].clone().add(
				new THREE.Vector3((Math.random() - 0.5) * 0.5, 0, (Math.random() - 0.5) * 0.5)
			));
		}

		warParties.push({
			warriors,
			target: settPositions[toIdx].clone(),
			origin: settPositions[fromIdx].clone(),
			progress: 0,
			speed: 0.8 + Math.random() * 0.5,
			active: true,
			cooldown: 0
		});
	}

	return {
		update(dt: number, timeOfDay: number) {
			const now = performance.now() / 1000;

			// Birds rest at night (reduce to near ground, slow movement)
			const isNight = timeOfDay < 5 || timeOfDay > 19.5;
			const birdActivity = isNight ? 0.1 : 1.0;

			// Update flocks
			for (const flock of flocks) {
				flock.angle += flock.speed * dt * birdActivity;
				flock.verticalOsc += dt * 0.5;
				flock.center.set(
					Math.cos(flock.angle) * flock.radius,
					flock.height * birdActivity + Math.sin(flock.verticalOsc) * 1.5,
					Math.sin(flock.angle) * flock.radius
				);
			}

			// Update bird positions
			for (let i = 0; i < birds.length; i++) {
				const bird = birds[i];
				const flock = flocks[bird.flockId];
				bird.wingPhase += dt * 8;

				// Bird position = flock center + offset (oscillating)
				const bx = flock.center.x + bird.offset.x + Math.sin(now * 0.5 + i) * 0.3;
				const by = flock.center.y + bird.offset.y + Math.sin(bird.wingPhase * 0.3) * 0.2;
				const bz = flock.center.z + bird.offset.z + Math.cos(now * 0.5 + i) * 0.3;
				birdSprites[i].position.set(bx, by, bz);

				// Wing flap scale animation
				const wingFlap = 0.2 + Math.abs(Math.sin(bird.wingPhase)) * 0.15;
				birdSprites[i].scale.set(0.6, wingFlap, 1);

				// Fade at night
				(birdSprites[i].material as THREE.SpriteMaterial).opacity = isNight ? 0.2 : 0.8;
			}

			// Update wanderers
			if (wandererPoints && wanderers.length > 0) {
				const pos = wandererPoints.geometry.attributes.position as THREE.BufferAttribute;
				for (let i = 0; i < wanderers.length; i++) {
					const w = wanderers[i];
					w.progress += w.speed * dt * 0.05;
					if (w.progress >= 1) {
						// Arrived — pick new destination
						w.fromIdx = w.toIdx;
						w.toIdx = Math.floor(Math.random() * settPositions.length);
						if (w.toIdx === w.fromIdx) w.toIdx = (w.toIdx + 1) % settPositions.length;
						w.target = settPositions[w.toIdx].clone();
						w.progress = 0;
					}

					const from = settPositions[w.fromIdx];
					const to = w.target;
					w.position.lerpVectors(from, to, w.progress);
					// Slight bobble
					w.position.y = 0.35 + Math.sin(now * 3 + i) * 0.02;
					pos.setXYZ(i, w.position.x, w.position.y, w.position.z);
				}
				pos.needsUpdate = true;
			}

			// War parties
			warCooldown -= dt;
			if (warCooldown <= 0 && warParties.filter(w => w.active).length < 2) {
				spawnWarParty();
				warCooldown = 15 + Math.random() * 20;
			}

			if (warPoints) {
				const pos = warPoints.geometry.attributes.position as THREE.BufferAttribute;
				let wIdx = 0;

				for (const wp of warParties) {
					if (!wp.active) continue;
					wp.progress += wp.speed * dt * 0.03;

					if (wp.progress >= 1) {
						// Battle reached! Create smoke
						wp.active = false;
						// Spawn smoke sprite at target
						const smoke = new THREE.Sprite(smokeMat.clone());
						smoke.position.copy(wp.target);
						smoke.position.y += 0.5;
						smoke.scale.set(0.1, 0.1, 1);
						scene.add(smoke);
						smokeSprites.push(smoke);
						continue;
					}

					for (const warrior of wp.warriors) {
						warrior.lerpVectors(wp.origin, wp.target, wp.progress);
						warrior.x += (Math.random() - 0.5) * 0.1;
						warrior.z += (Math.random() - 0.5) * 0.1;
						warrior.y = 0.35;
						if (wIdx < 30) {
							pos.setXYZ(wIdx, warrior.x, warrior.y, warrior.z);
							wIdx++;
						}
					}
				}

				// Clear remaining slots
				for (let i = wIdx; i < 30; i++) {
					pos.setXYZ(i, 0, -100, 0); // off-screen
				}
				pos.needsUpdate = true;
			}

			// Animate smoke (grows then fades)
			for (let i = smokeSprites.length - 1; i >= 0; i--) {
				const smoke = smokeSprites[i];
				const mat = smoke.material as THREE.SpriteMaterial;
				const scale = smoke.scale.x;
				if (scale < 2) {
					smoke.scale.set(scale + dt * 0.8, scale + dt * 0.8, 1);
					smoke.position.y += dt * 0.3;
					mat.opacity = Math.min(0.5, mat.opacity + dt * 0.3);
				} else {
					mat.opacity -= dt * 0.15;
					smoke.position.y += dt * 0.1;
					if (mat.opacity <= 0) {
						scene.remove(smoke);
						mat.dispose();
						smokeSprites.splice(i, 1);
					}
				}
			}

			// Cleanup dead war parties
			for (let i = warParties.length - 1; i >= 0; i--) {
				if (!warParties[i].active) {
					warParties[i].cooldown += dt;
					if (warParties[i].cooldown > 5) warParties.splice(i, 1);
				}
			}
		},

		dispose() {
			birdTex.dispose();
			for (const s of birdSprites) {
				(s.material as THREE.SpriteMaterial).dispose();
				scene.remove(s);
			}
			if (wandererPoints) {
				wandererPoints.geometry.dispose();
				(wandererPoints.material as THREE.PointsMaterial).dispose();
				scene.remove(wandererPoints);
			}
			if (warPoints) {
				warPoints.geometry.dispose();
				(warPoints.material as THREE.PointsMaterial).dispose();
				scene.remove(warPoints);
			}
			for (const smoke of smokeSprites) {
				(smoke.material as THREE.SpriteMaterial).dispose();
				scene.remove(smoke);
			}
			smokeSprites.length = 0;
		}
	};
}
