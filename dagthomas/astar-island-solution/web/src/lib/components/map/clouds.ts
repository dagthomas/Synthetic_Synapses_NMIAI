// Cloud billboard sprite system — 10 drifting sprites with procedural texture

import * as THREE from 'three';

const CLOUD_COUNT = 10;
const BOUNDS = 30; // respawn bounds
const _tint = new THREE.Color();

export interface CloudSystem {
	sprites: THREE.Sprite[];
	speeds: number[];
	update(dt: number, skyColor: THREE.Color, windDir?: THREE.Vector2, windStrength?: number, stormIntensity?: number): void;
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
			(Math.random() - 0.5) * BOUNDS * 2,
			3 + Math.random() * 5,
			(Math.random() - 0.5) * BOUNDS
		);
		sprite.renderOrder = 100;

		scene.add(sprite);
		sprites.push(sprite);
		speeds.push(0.5 + Math.random() * 1.5);
	}

	return {
		sprites,
		speeds,
		update(dt: number, skyColor: THREE.Color, windDir?: THREE.Vector2, windStrength?: number, stormIntensity?: number) {
			const wdx = windDir?.x ?? 1;
			const wdz = windDir?.y ?? 0;
			const ws = windStrength ?? 0.4;
			const si = stormIntensity ?? 0;

			// Storm cloud color: white → gray → dark gray
			const stormGray = Math.max(0.25, 1.0 - si * 0.7);
			const _stormColor = new THREE.Color(stormGray, stormGray, stormGray);

			for (let i = 0; i < sprites.length; i++) {
				const speed = speeds[i] * (0.5 + ws * 0.8);
				sprites[i].position.x += wdx * speed * dt;
				sprites[i].position.z += wdz * speed * dt;

				// Lower clouds during storms
				if (si > 0.1) {
					const targetY = 2 + Math.random() * 2;
					sprites[i].position.y += (targetY - sprites[i].position.y) * dt * 0.1 * si;
				}

				// Respawn at upwind edge when cloud drifts out of bounds
				const px = sprites[i].position.x;
				const pz = sprites[i].position.z;
				if (px > BOUNDS || px < -BOUNDS || pz > BOUNDS || pz < -BOUNDS) {
					sprites[i].position.x = -wdx * BOUNDS + (Math.random() - 0.5) * BOUNDS * 0.5;
					sprites[i].position.z = -wdz * BOUNDS + (Math.random() - 0.5) * BOUNDS * 0.5;
					sprites[i].position.y = si > 0.3 ? 2 + Math.random() * 2 : 3 + Math.random() * 5;
				}

				// Tint: blend sky color with storm darkness
				const mat = sprites[i].material as THREE.SpriteMaterial;
				_tint.set(0xffffff).lerp(skyColor, 0.25);
				_tint.lerp(_stormColor, si);
				mat.color.copy(_tint);
				mat.opacity = 0.5 + si * 0.25; // denser during storms
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
