// Localized weather: rain near mountains, lightning flashes, fog around lakes

import * as THREE from 'three';

export interface WeatherSystem {
	update(dt: number, timeOfDay: number, stormIntensity?: number): void;
	dispose(): void;
}

interface RainZone {
	center: THREE.Vector3;
	radius: number;
	points: THREE.Points;
	velocities: Float32Array;
}

interface LightningState {
	nextStrike: number;
	flashLight: THREE.PointLight | null;
	flashTimer: number;
	flashPhase: number;
	boltLine: THREE.Line | null;
}

interface FogSprite {
	sprite: THREE.Sprite;
	baseX: number;
	baseY: number;
	baseZ: number;
	driftRadius: number;
	phase: number;
}

export function createWeatherSystem(
	scene: THREE.Scene,
	mountainPositions: { x: number; z: number; radius: number }[],
	lakePositions: { x: number; z: number }[],
	mapRadius = 12
): WeatherSystem {
	const rainZones: RainZone[] = [];
	const fogSprites: FogSprite[] = [];
	const lightning: LightningState = {
		nextStrike: 5 + Math.random() * 5,
		flashLight: null,
		flashTimer: 0,
		flashPhase: 0,
		boltLine: null
	};

	// --- Global rain covering entire map ---
	{
		const particleCount = 600;
		const spread = mapRadius;
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(particleCount * 3);
		const velocities = new Float32Array(particleCount);

		for (let i = 0; i < particleCount; i++) {
			positions[i * 3] = (Math.random() - 0.5) * spread * 2;
			positions[i * 3 + 1] = Math.random() * 12 + 2;
			positions[i * 3 + 2] = (Math.random() - 0.5) * spread * 2;
			velocities[i] = 6 + Math.random() * 4;
		}

		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

		const mat = new THREE.PointsMaterial({
			color: 0xaabbcc,
			size: 0.04,
			transparent: true,
			opacity: 0.25,
			depthWrite: false,
			blending: THREE.NormalBlending
		});

		const points = new THREE.Points(geo, mat);
		points.visible = false;
		scene.add(points);

		rainZones.push({
			center: new THREE.Vector3(0, 0, 0),
			radius: spread,
			points,
			velocities
		});
	}

	// --- Extra rain near mountains ---
	for (const mt of mountainPositions) {
		const particleCount = Math.round(60 + mt.radius * 15);
		const spread = Math.max(2.5, mt.radius);
		const geo = new THREE.BufferGeometry();
		const positions = new Float32Array(particleCount * 3);
		const velocities = new Float32Array(particleCount);

		for (let i = 0; i < particleCount; i++) {
			positions[i * 3] = mt.x + (Math.random() - 0.5) * spread * 2;
			positions[i * 3 + 1] = Math.random() * 10 + 2;
			positions[i * 3 + 2] = mt.z + (Math.random() - 0.5) * spread * 2;
			velocities[i] = 6 + Math.random() * 4;
		}

		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

		const mat = new THREE.PointsMaterial({
			color: 0xaabbcc,
			size: 0.03,
			transparent: true,
			opacity: 0.25,
			depthWrite: false,
			blending: THREE.NormalBlending
		});

		const points = new THREE.Points(geo, mat);
		points.visible = false;
		scene.add(points);

		rainZones.push({
			center: new THREE.Vector3(mt.x, 0, mt.z),
			radius: spread,
			points,
			velocities
		});
	}

	// --- Fog around lakes ---
	const fogTex = createFogTexture();
	for (const lake of lakePositions) {
		const count = 3 + Math.floor(Math.random() * 3);
		for (let i = 0; i < count; i++) {
			const mat = new THREE.SpriteMaterial({
				map: fogTex,
				transparent: true,
				opacity: 0.18,
				depthWrite: false,
				fog: true
			});
			const sprite = new THREE.Sprite(mat);
			const scale = 2 + Math.random() * 2;
			sprite.scale.set(scale, scale * 0.4, 1);
			const baseX = lake.x + (Math.random() - 0.5) * 2;
			const baseY = 0.3 + Math.random() * 0.5;
			const baseZ = lake.z + (Math.random() - 0.5) * 2;
			sprite.position.set(baseX, baseY, baseZ);
			sprite.renderOrder = 50;
			scene.add(sprite);

			fogSprites.push({
				sprite,
				baseX,
				baseY,
				baseZ,
				driftRadius: 0.5 + Math.random() * 1.0,
				phase: Math.random() * Math.PI * 2
			});
		}
	}

	// --- Lightning setup ---
	if (mountainPositions.length > 0) {
		lightning.flashLight = new THREE.PointLight(0xddeeff, 0, 120);
		lightning.flashLight.position.set(0, 15, 0);
		scene.add(lightning.flashLight);
	}

	function createFogTexture(size = 128): THREE.Texture {
		const canvas = document.createElement('canvas');
		canvas.width = size;
		canvas.height = size;
		const ctx = canvas.getContext('2d')!;
		const cx = size / 2;
		const grad = ctx.createRadialGradient(cx, cx, 0, cx, cx, cx);
		grad.addColorStop(0, 'rgba(220,225,235,0.6)');
		grad.addColorStop(0.4, 'rgba(210,215,225,0.3)');
		grad.addColorStop(0.7, 'rgba(200,205,215,0.1)');
		grad.addColorStop(1, 'rgba(190,195,205,0)');
		ctx.fillStyle = grad;
		ctx.fillRect(0, 0, size, size);
		const tex = new THREE.CanvasTexture(canvas);
		tex.needsUpdate = true;
		return tex;
	}

	return {
		update(dt: number, timeOfDay: number, stormIntensity = 0) {
			// --- Update rain (scale by storm intensity) ---
			const rainActive = stormIntensity > 0.1;
			for (const zone of rainZones) {
				zone.points.visible = rainActive;
				if (!rainActive) continue;

				const speedMult = 1 + stormIntensity * 2; // faster rain in storms
				const mat = zone.points.material as THREE.PointsMaterial;
				mat.opacity = 0.15 + stormIntensity * 0.35;
				mat.size = 0.03 + stormIntensity * 0.02;

				// Wider spread during storms
				const spreadMult = 1 + stormIntensity * 1.5;

				const pos = zone.points.geometry.attributes.position as THREE.BufferAttribute;
				const arr = pos.array as Float32Array;
				for (let i = 0; i < pos.count; i++) {
					arr[i * 3 + 1] -= zone.velocities[i] * dt * speedMult;
					if (arr[i * 3 + 1] < 0.1) {
						arr[i * 3] = zone.center.x + (Math.random() - 0.5) * zone.radius * 2 * spreadMult;
						arr[i * 3 + 1] = 8 + Math.random() * 4;
						arr[i * 3 + 2] = zone.center.z + (Math.random() - 0.5) * zone.radius * 2 * spreadMult;
					}
				}
				pos.needsUpdate = true;
			}

			// --- Update lightning (only during rain/storm) ---
			if (lightning.flashLight && mountainPositions.length > 0) {
				if (stormIntensity > 0.2) {
					lightning.nextStrike -= dt;

					// Storm: strike every 1-3s. Rain: every 8-15s
					const isStorm = stormIntensity > 0.6;

					if (lightning.nextStrike <= 0 && lightning.flashPhase === 0) {
						lightning.flashPhase = 1;
						lightning.flashTimer = 0;
						const target = mountainPositions[Math.floor(Math.random() * mountainPositions.length)];
						lightning.flashLight.position.set(target.x, 15, target.z);

						if (lightning.boltLine) {
							scene.remove(lightning.boltLine);
							lightning.boltLine.geometry.dispose();
						}
						const boltPoints = [];
						let bx = target.x + (Math.random() - 0.5) * 3;
						let bz = target.z + (Math.random() - 0.5) * 3;
						boltPoints.push(new THREE.Vector3(bx, 15, bz));
						const segments = isStorm ? 5 : 3;
						for (let seg = 0; seg < segments; seg++) {
							bx += (Math.random() - 0.5) * 2;
							bz += (Math.random() - 0.5) * 2;
							const by = 15 - (seg + 1) * (14.5 / (segments + 1));
							boltPoints.push(new THREE.Vector3(bx, by, bz));
						}
						boltPoints.push(new THREE.Vector3(bx + (Math.random() - 0.5) * 0.5, 0.5, bz + (Math.random() - 0.5) * 0.5));

						const boltGeo = new THREE.BufferGeometry().setFromPoints(boltPoints);
						const brightness = isStorm ? 5 : 3;
						const boltMat = new THREE.LineBasicMaterial({
							color: new THREE.Color(brightness, brightness, brightness + 1),
							transparent: true,
							opacity: 1.0,
							linewidth: 2
						});
						lightning.boltLine = new THREE.Line(boltGeo, boltMat);
						scene.add(lightning.boltLine);
					}

					if (lightning.flashPhase > 0) {
						lightning.flashTimer += dt;
						const t = lightning.flashTimer;
						const peak = isStorm ? 25.0 : 15.0;
						if (t < 0.05) {
							lightning.flashLight.intensity = peak;
						} else if (t < 0.1) {
							lightning.flashLight.intensity = 2.0;
						} else if (t < 0.15) {
							lightning.flashLight.intensity = peak * 0.8;
						} else if (t < 0.2) {
							lightning.flashLight.intensity = 1.0;
						} else if (t < 0.25) {
							lightning.flashLight.intensity = peak * 0.5;
						} else {
							lightning.flashLight.intensity = 0;
							lightning.flashPhase = 0;
							lightning.nextStrike = isStorm ? 1 + Math.random() * 2 : 8 + Math.random() * 7;
							if (lightning.boltLine) {
								scene.remove(lightning.boltLine);
								lightning.boltLine.geometry.dispose();
								lightning.boltLine = null;
							}
						}

						if (lightning.boltLine) {
							const mat = lightning.boltLine.material as THREE.LineBasicMaterial;
							mat.opacity = Math.max(0, 1 - t * 3);
						}
					}
				} else {
					// No lightning when calm
					lightning.flashLight.intensity = 0;
				}
			}

			// --- Update fog ---
			const h = ((timeOfDay % 24) + 24) % 24;
			// More visible at dawn/dusk, dimmer at bright day
			const isDawnDusk = (h > 5 && h < 8) || (h > 17 && h < 20);
			const isNight = h < 5 || h > 20;
			const fogOpacityMult = isDawnDusk ? 1.2 : isNight ? 0.9 : 0.6;

			for (const fog of fogSprites) {
				fog.phase += dt * 0.4;

				// Oscillate around base position (never drifts away)
				fog.sprite.position.x = fog.baseX + Math.sin(fog.phase) * fog.driftRadius;
				fog.sprite.position.z = fog.baseZ + Math.cos(fog.phase * 0.7) * fog.driftRadius * 0.6;
				fog.sprite.position.y = fog.baseY + Math.sin(fog.phase * 1.3) * 0.1;

				const mat = fog.sprite.material as THREE.SpriteMaterial;
				mat.opacity = 0.18 * fogOpacityMult;

				if (isNight) {
					mat.color.setHex(0x8899bb);
				} else if (isDawnDusk) {
					mat.color.setHex(0xddccaa);
				} else {
					mat.color.setHex(0xffffff);
				}
			}
		},

		dispose() {
			for (const zone of rainZones) {
				scene.remove(zone.points);
				zone.points.geometry.dispose();
				(zone.points.material as THREE.Material).dispose();
			}
			for (const fog of fogSprites) {
				scene.remove(fog.sprite);
				(fog.sprite.material as THREE.SpriteMaterial).map?.dispose();
				(fog.sprite.material as THREE.SpriteMaterial).dispose();
			}
			if (lightning.flashLight) {
				scene.remove(lightning.flashLight);
			}
			if (lightning.boltLine) {
				scene.remove(lightning.boltLine);
				lightning.boltLine.geometry.dispose();
			}
		}
	};
}
