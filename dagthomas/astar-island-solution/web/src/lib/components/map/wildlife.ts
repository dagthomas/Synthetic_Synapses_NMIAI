// Living world system: bird flocking sprites

import * as THREE from 'three';
import { mulberry32 } from './prng';

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
	scene: THREE.Scene
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
			height: 1.5 + rng() * 3,
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
		},

		dispose() {
			birdTex.dispose();
			for (const s of birdSprites) {
				(s.material as THREE.SpriteMaterial).dispose();
				scene.remove(s);
			}
		}
	};
}
