// Cloud billboard sprite system — 10 drifting sprites with procedural texture

import * as THREE from 'three';

const CLOUD_COUNT = 10;
const BOUNDS_X = 30; // respawn bounds
const _tint = new THREE.Color();

export interface CloudSystem {
	sprites: THREE.Sprite[];
	speeds: number[];
	update(dt: number, skyColor: THREE.Color): void;
	dispose(): void;
}

function createCloudTexture(): THREE.Texture {
	const canvas = document.createElement('canvas');
	canvas.width = 128;
	canvas.height = 64;
	const ctx = canvas.getContext('2d')!;

	// Overlapping white circles with gaussian alpha
	ctx.clearRect(0, 0, 128, 64);
	const circles = [
		{ x: 40, y: 32, r: 22 },
		{ x: 64, y: 28, r: 26 },
		{ x: 88, y: 32, r: 20 },
		{ x: 52, y: 36, r: 18 },
		{ x: 76, y: 34, r: 18 },
		{ x: 58, y: 24, r: 16 },
		{ x: 72, y: 26, r: 14 }
	];

	for (const c of circles) {
		const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, c.r);
		grad.addColorStop(0, 'rgba(255,255,255,0.6)');
		grad.addColorStop(0.5, 'rgba(255,255,255,0.3)');
		grad.addColorStop(1, 'rgba(255,255,255,0)');
		ctx.fillStyle = grad;
		ctx.fillRect(c.x - c.r, c.y - c.r, c.r * 2, c.r * 2);
	}

	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	return tex;
}

export function createCloudSystem(scene: THREE.Scene): CloudSystem {
	const texture = createCloudTexture();
	const sprites: THREE.Sprite[] = [];
	const speeds: number[] = [];

	for (let i = 0; i < CLOUD_COUNT; i++) {
		const mat = new THREE.SpriteMaterial({
			map: texture,
			transparent: true,
			opacity: 0.5,
			depthWrite: false,
			fog: false
		});
		const sprite = new THREE.Sprite(mat);

		const scale = 10 + Math.random() * 16;
		sprite.scale.set(scale, scale * 0.35, 1);
		sprite.position.set(
			(Math.random() - 0.5) * BOUNDS_X * 2,
			3 + Math.random() * 5,
			(Math.random() - 0.5) * BOUNDS_X
		);
		sprite.renderOrder = 100;

		scene.add(sprite);
		sprites.push(sprite);
		speeds.push(0.5 + Math.random() * 1.5);
	}

	return {
		sprites,
		speeds,
		update(dt: number, skyColor: THREE.Color) {
			for (let i = 0; i < sprites.length; i++) {
				sprites[i].position.x += speeds[i] * dt;
				// Respawn at opposite edge
				if (sprites[i].position.x > BOUNDS_X) {
					sprites[i].position.x = -BOUNDS_X;
					sprites[i].position.z = (Math.random() - 0.5) * BOUNDS_X;
				}
				// Tint clouds with sky color (warm at dusk, gray at night)
				const mat = sprites[i].material as THREE.SpriteMaterial;
				_tint.set(0xffffff).lerp(skyColor, 0.25);
				mat.color.copy(_tint);
			}
		},
		dispose() {
			texture.dispose();
			for (const s of sprites) {
				(s.material as THREE.SpriteMaterial).dispose();
				scene.remove(s);
			}
		}
	};
}
