// Cloud billboard sprite system — drifting sprites with procedural texture
// Mountain collision: clouds deflect upward over peaks
// Peak clouds: small wispy clouds orbit mountain summits

import * as THREE from 'three';

const CLOUD_COUNT = 10;
const PEAK_CLOUD_COUNT = 3; // per mountain peak (max peaks capped)
const MAX_PEAK_CLOUDS = 4; // max mountains that get peak clouds
const BOUNDS = 30; // respawn bounds
const CLOUD_BASE_HEIGHT = 3.0;
const CLOUD_CLEARANCE = 1.0; // min height above terrain
const _tint = new THREE.Color();

export type HeightFn = (x: number, z: number) => number;

export interface MountainPeak {
	x: number;
	z: number;
	height: number;
}

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

function createPeakCloudTexture(): THREE.Texture {
	const canvas = document.createElement('canvas');
	canvas.width = 64;
	canvas.height = 32;
	const ctx = canvas.getContext('2d')!;
	ctx.clearRect(0, 0, 64, 32);
	const circles = [
		{ x: 20, y: 16, r: 10 },
		{ x: 32, y: 14, r: 12 },
		{ x: 44, y: 16, r: 10 },
	];
	for (const c of circles) {
		const grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, c.r);
		grad.addColorStop(0, 'rgba(255,255,255,0.4)');
		grad.addColorStop(0.6, 'rgba(255,255,255,0.15)');
		grad.addColorStop(1, 'rgba(255,255,255,0)');
		ctx.fillStyle = grad;
		ctx.fillRect(c.x - c.r, c.y - c.r, c.r * 2, c.r * 2);
	}
	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	return tex;
}

export function createCloudSystem(scene: THREE.Scene, heightFn?: HeightFn, peaks?: MountainPeak[]): CloudSystem {
	const texture = createCloudTexture();
	const peakTexture = createPeakCloudTexture();
	const sprites: THREE.Sprite[] = [];
	const speeds: number[] = [];
	const isPeakCloud: boolean[] = [];

	// Drifting clouds
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
			CLOUD_BASE_HEIGHT + Math.random() * 5,
			(Math.random() - 0.5) * BOUNDS
		);
		sprite.renderOrder = 100;

		scene.add(sprite);
		sprites.push(sprite);
		speeds.push(0.5 + Math.random() * 1.5);
		isPeakCloud.push(false);
	}

	// Peak clouds — small wisps orbiting mountain summits
	const usedPeaks = (peaks || [])
		.filter(p => p.height > 1.0)
		.sort((a, b) => b.height - a.height)
		.slice(0, MAX_PEAK_CLOUDS);

	const peakCloudAngles: number[] = [];
	for (const peak of usedPeaks) {
		for (let j = 0; j < PEAK_CLOUD_COUNT; j++) {
			const mat = new THREE.SpriteMaterial({
				map: peakTexture,
				transparent: true,
				opacity: 0.35,
				depthWrite: false,
				fog: false,
			});
			const sprite = new THREE.Sprite(mat);
			const scale = 2 + Math.random() * 3;
			sprite.scale.set(scale, scale * 0.4, 1);

			const angle = (j / PEAK_CLOUD_COUNT) * Math.PI * 2 + Math.random() * 0.5;
			const radius = 0.8 + Math.random() * 1.2;
			sprite.position.set(
				peak.x + Math.cos(angle) * radius,
				peak.height - 0.3 + Math.random() * 0.8,
				peak.z + Math.sin(angle) * radius,
			);
			sprite.renderOrder = 100;

			scene.add(sprite);
			sprites.push(sprite);
			speeds.push(0.3 + Math.random() * 0.4);
			isPeakCloud.push(true);
			peakCloudAngles.push(angle);
		}
	}

	// Track peak cloud orbit state
	const peakCloudState = usedPeaks.flatMap((peak, pi) =>
		Array.from({ length: PEAK_CLOUD_COUNT }, (_, j) => ({
			peak,
			angle: peakCloudAngles[pi * PEAK_CLOUD_COUNT + j],
			radius: 0.8 + Math.random() * 1.2,
			yOffset: -0.3 + Math.random() * 0.8,
			orbitSpeed: 0.15 + Math.random() * 0.2,
		}))
	);

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

			// Update drifting clouds
			for (let i = 0; i < CLOUD_COUNT; i++) {
				const speed = speeds[i] * (0.5 + ws * 0.8);
				sprites[i].position.x += wdx * speed * dt;
				sprites[i].position.z += wdz * speed * dt;

				// Mountain collision: push cloud upward when over high terrain
				if (heightFn) {
					const terrainH = heightFn(sprites[i].position.x, sprites[i].position.z);
					const minY = terrainH + CLOUD_CLEARANCE;
					if (sprites[i].position.y < minY) {
						// Deflect upward smoothly
						sprites[i].position.y += (minY - sprites[i].position.y + 0.5) * dt * 3;
					} else if (si <= 0.1) {
						// Gently settle back to base height when past mountain
						const baseY = CLOUD_BASE_HEIGHT + Math.random() * 5;
						if (sprites[i].position.y > baseY + 1) {
							sprites[i].position.y += (baseY - sprites[i].position.y) * dt * 0.3;
						}
					}
				}

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
					sprites[i].position.y = si > 0.3 ? 2 + Math.random() * 2 : CLOUD_BASE_HEIGHT + Math.random() * 5;
				}

				// Tint: blend sky color with storm darkness
				const mat = sprites[i].material as THREE.SpriteMaterial;
				_tint.set(0xffffff).lerp(skyColor, 0.25);
				_tint.lerp(_stormColor, si);
				mat.color.copy(_tint);
				mat.opacity = 0.5 + si * 0.25; // denser during storms
			}

			// Update peak clouds — slow orbit around summits
			for (let j = 0; j < peakCloudState.length; j++) {
				const idx = CLOUD_COUNT + j;
				if (idx >= sprites.length) break;
				const pc = peakCloudState[j];
				pc.angle += pc.orbitSpeed * dt;

				sprites[idx].position.x = pc.peak.x + Math.cos(pc.angle) * pc.radius;
				sprites[idx].position.z = pc.peak.z + Math.sin(pc.angle) * pc.radius;
				sprites[idx].position.y = pc.peak.height + pc.yOffset + Math.sin(pc.angle * 0.7) * 0.15;

				// Tint peak clouds same as main clouds
				const mat = sprites[idx].material as THREE.SpriteMaterial;
				_tint.set(0xffffff).lerp(skyColor, 0.15);
				mat.color.copy(_tint);
			}
		},
		dispose() {
			texture.dispose();
			peakTexture.dispose();
			for (const s of sprites) {
				(s.material as THREE.SpriteMaterial).dispose();
				scene.remove(s);
			}
		}
	};
}
