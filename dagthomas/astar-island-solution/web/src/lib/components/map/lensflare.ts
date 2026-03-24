/**
 * Cinematic lens flare with internal reflections and refractions.
 * - Main glow + soft halo
 * - Ghost flares along sun→screen-center axis
 * - Hexagonal aperture reflections (iris ghosts)
 * - Chromatic refraction disc (rainbow edge)
 * - Anamorphic horizontal streak
 */
import * as THREE from 'three';

export interface LensFlareSystem {
	update(camera: THREE.PerspectiveCamera, sunPos: THREE.Vector3, nightFade: number, hour: number): void;
	setHeightFn(fn: (x: number, z: number) => number): void;
	dispose(): void;
	group: THREE.Group;
}

interface FlareElement {
	sprite: THREE.Sprite;
	offset: number;
	scale: number;
	baseOpacity: number;
	color: THREE.Color;
}

function createGlowTexture(size: number, softness: number): THREE.Texture {
	const c = document.createElement('canvas');
	c.width = size; c.height = size;
	const ctx = c.getContext('2d')!;
	const cx = size / 2;
	const grad = ctx.createRadialGradient(cx, cx, 0, cx, cx, cx);
	grad.addColorStop(0, 'rgba(255,255,255,1)');
	grad.addColorStop(softness, 'rgba(255,255,255,0.3)');
	grad.addColorStop(1, 'rgba(255,255,255,0)');
	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, size, size);
	const tex = new THREE.CanvasTexture(c);
	tex.needsUpdate = true;
	return tex;
}

function createRingTexture(size: number): THREE.Texture {
	const c = document.createElement('canvas');
	c.width = size; c.height = size;
	const ctx = c.getContext('2d')!;
	const cx = size / 2;
	const grad = ctx.createRadialGradient(cx, cx, cx * 0.6, cx, cx, cx * 0.85);
	grad.addColorStop(0, 'rgba(255,255,255,0)');
	grad.addColorStop(0.4, 'rgba(255,255,255,0.15)');
	grad.addColorStop(0.6, 'rgba(255,255,255,0.08)');
	grad.addColorStop(1, 'rgba(255,255,255,0)');
	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, size, size);
	const tex = new THREE.CanvasTexture(c);
	tex.needsUpdate = true;
	return tex;
}

/** Hexagonal aperture ghost — 6-sided iris reflection */
function createHexTexture(size: number): THREE.Texture {
	const c = document.createElement('canvas');
	c.width = size; c.height = size;
	const ctx = c.getContext('2d')!;
	const cx = size / 2, r = size * 0.4;
	ctx.clearRect(0, 0, size, size);

	// Draw hexagon outline with soft glow
	for (let pass = 0; pass < 3; pass++) {
		const pr = r + pass * 3;
		const alpha = [0.2, 0.1, 0.04][pass];
		ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
		ctx.lineWidth = 3 - pass;
		ctx.beginPath();
		for (let i = 0; i < 6; i++) {
			const a = (i / 6) * Math.PI * 2 - Math.PI / 6;
			const x = cx + Math.cos(a) * pr;
			const y = cx + Math.sin(a) * pr;
			i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
		}
		ctx.closePath();
		ctx.stroke();
	}

	// Subtle fill
	const grad = ctx.createRadialGradient(cx, cx, 0, cx, cx, r);
	grad.addColorStop(0, 'rgba(255,255,255,0.03)');
	grad.addColorStop(0.7, 'rgba(255,255,255,0.01)');
	grad.addColorStop(1, 'rgba(255,255,255,0)');
	ctx.fillStyle = grad;
	ctx.fill();

	const tex = new THREE.CanvasTexture(c);
	tex.needsUpdate = true;
	return tex;
}

/** Chromatic refraction disc — rainbow ring like a lens ghost */
function createChromaticDiscTexture(size: number): THREE.Texture {
	const c = document.createElement('canvas');
	c.width = size; c.height = size;
	const ctx = c.getContext('2d')!;
	const cx = size / 2;

	// Rainbow ring — concentric colored bands
	const colors = [
		[255, 80, 80, 0.08],   // red
		[255, 180, 50, 0.07],  // orange
		[255, 255, 100, 0.06], // yellow
		[100, 255, 100, 0.06], // green
		[80, 180, 255, 0.07],  // blue
		[180, 100, 255, 0.06], // violet
	];

	for (let i = 0; i < colors.length; i++) {
		const innerR = cx * (0.6 + i * 0.05);
		const outerR = cx * (0.65 + i * 0.05);
		const [r, g, b, a] = colors[i];
		const grad = ctx.createRadialGradient(cx, cx, innerR, cx, cx, outerR);
		grad.addColorStop(0, `rgba(${r},${g},${b},0)`);
		grad.addColorStop(0.3, `rgba(${r},${g},${b},${a})`);
		grad.addColorStop(0.7, `rgba(${r},${g},${b},${a})`);
		grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
		ctx.fillStyle = grad;
		ctx.fillRect(0, 0, size, size);
	}

	const tex = new THREE.CanvasTexture(c);
	tex.needsUpdate = true;
	return tex;
}

function createStreakTexture(): THREE.Texture {
	const c = document.createElement('canvas');
	c.width = 256; c.height = 32;
	const ctx = c.getContext('2d')!;
	const grad = ctx.createLinearGradient(0, 16, 256, 16);
	grad.addColorStop(0, 'rgba(255,255,255,0)');
	grad.addColorStop(0.3, 'rgba(255,255,255,0.15)');
	grad.addColorStop(0.5, 'rgba(255,255,255,0.4)');
	grad.addColorStop(0.7, 'rgba(255,255,255,0.15)');
	grad.addColorStop(1, 'rgba(255,255,255,0)');
	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, 256, 32);
	const tex = new THREE.CanvasTexture(c);
	tex.needsUpdate = true;
	return tex;
}

export function createLensFlareSystem(): LensFlareSystem {
	const group = new THREE.Group();
	group.renderOrder = 999;

	const glowTex = createGlowTexture(128, 0.4);
	const softGlowTex = createGlowTexture(128, 0.2);
	const ringTex = createRingTexture(128);
	const hexTex = createHexTexture(128);
	const chromaticTex = createChromaticDiscTexture(256);
	const streakTex = createStreakTexture();

	const elements: FlareElement[] = [];
	const textures = [glowTex, softGlowTex, ringTex, hexTex, chromaticTex, streakTex];

	function addFlare(tex: THREE.Texture, offset: number, scale: number, opacity: number, color: [number, number, number]) {
		const mat = new THREE.SpriteMaterial({
			map: tex, transparent: true, opacity: 0,
			blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
		});
		const sprite = new THREE.Sprite(mat);
		sprite.renderOrder = 999;
		group.add(sprite);
		elements.push({ sprite, offset, scale, baseOpacity: opacity, color: new THREE.Color(...color) });
	}

	// Main sun glow
	addFlare(glowTex, 0, 8, 0.6, [1.0, 0.95, 0.8]);
	// Soft outer halo
	addFlare(softGlowTex, 0, 18, 0.15, [1.0, 0.9, 0.7]);

	// Ghost flares along axis
	addFlare(ringTex, 0.25, 3, 0.12, [0.8, 0.9, 1.0]);
	addFlare(glowTex, 0.45, 2, 0.08, [0.9, 0.85, 1.0]);

	// Hexagonal aperture reflections (iris ghosts)
	addFlare(hexTex, 0.35, 5, 0.1, [0.7, 0.85, 1.0]);
	addFlare(hexTex, 0.6, 3.5, 0.08, [0.85, 0.8, 1.0]);
	addFlare(hexTex, 0.85, 6, 0.06, [0.6, 0.75, 1.0]);
	addFlare(hexTex, 1.15, 4, 0.05, [0.8, 0.7, 1.0]);

	// Chromatic refraction disc (rainbow ghost)
	addFlare(chromaticTex, 0.5, 7, 0.12, [1.0, 1.0, 1.0]);
	addFlare(chromaticTex, 0.75, 4, 0.08, [1.0, 1.0, 1.0]);
	addFlare(chromaticTex, 1.1, 9, 0.05, [1.0, 1.0, 1.0]);

	// Small bright dots
	addFlare(glowTex, 0.7, 1, 0.1, [1.0, 0.8, 0.6]);
	addFlare(glowTex, 1.0, 1.5, 0.08, [0.6, 0.8, 1.0]);
	addFlare(glowTex, 1.4, 2, 0.04, [0.8, 0.7, 1.0]);
	addFlare(ringTex, 1.6, 5, 0.03, [0.7, 0.8, 1.0]);

	// Horizontal anamorphic streak
	const streakMat = new THREE.SpriteMaterial({
		map: streakTex, transparent: true, opacity: 0,
		blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
	});
	const streakSprite = new THREE.Sprite(streakMat);
	streakSprite.renderOrder = 999;
	group.add(streakSprite);

	const _sunScreen = new THREE.Vector3();
	const _camFwd = new THREE.Vector3();
	let smoothIntensity = 0;
	let _heightFn: ((x: number, z: number) => number) | null = null;

	return {
		group,

		setHeightFn(fn: (x: number, z: number) => number) {
			_heightFn = fn;
		},

		update(camera: THREE.PerspectiveCamera, sunPos: THREE.Vector3, nightFade: number, hour: number) {
			_sunScreen.copy(sunPos).project(camera);

			const inFront = _sunScreen.z < 1;
			const onScreen = Math.abs(_sunScreen.x) < 1.4 && Math.abs(_sunScreen.y) < 1.4;

			// Terrain occlusion: check if terrain blocks the line of sight to the sun
			let occluded = false;
			if (_heightFn && inFront) {
				const camP = camera.position;
				const dir = sunPos.clone().sub(camP).normalize();
				// Sample 5 points along the ray toward the sun
				for (let step = 1; step <= 5; step++) {
					const d = step * 5; // check every 5 units
					const sx = camP.x + dir.x * d;
					const sz = camP.z + dir.z * d;
					const sy = camP.y + dir.y * d;
					const terrainH = _heightFn(sx, sz);
					if (sy < terrainH + 0.5) {
						occluded = true;
						break;
					}
				}
			}

			// Sun altitude
			const sunAlt = sunPos.y / 60;
			const altFactor = sunAlt > 0 ? (sunAlt < 0.3 ? 1.5 : sunAlt < 0.7 ? 1.0 : 0.7) : 0;

			// Angle between camera forward and sun direction
			camera.getWorldDirection(_camFwd);
			const sunDir = sunPos.clone().sub(camera.position).normalize();
			const dot = _camFwd.dot(sunDir);
			const angleFactor = Math.max(0, dot);

			// Edge boost
			const edgeDist = Math.max(Math.abs(_sunScreen.x), Math.abs(_sunScreen.y));
			const edgeBoost = 1 + Math.max(0, edgeDist - 0.3) * 0.4;

			const dayFactor = 1 - nightFade;
			const occlusionFactor = occluded ? 0 : 1;
			const targetIntensity = inFront && onScreen ? altFactor * dayFactor * edgeBoost * (0.5 + angleFactor * 0.5) * occlusionFactor : 0;
			smoothIntensity += (targetIntensity - smoothIntensity) * 0.08;

			if (smoothIntensity < 0.001) {
				group.visible = false;
				return;
			}
			group.visible = true;

			const isLowSun = sunAlt < 0.3 && sunAlt > 0;
			const warmShift = isLowSun ? 1.4 : 1.0;

			const sunWorld = sunPos.clone();
			// Screen center in world space (for flare axis)
			const centerWorld = camera.position.clone().add(_camFwd.multiplyScalar(100));
			const camPos = camera.position;

			for (const el of elements) {
				const pos = sunWorld.clone().lerp(centerWorld, el.offset);
				const dir = pos.clone().sub(camPos).normalize();
				el.sprite.position.copy(camPos).addScaledVector(dir, 50);

				const s = el.scale * (1 + smoothIntensity * 0.2);
				el.sprite.scale.set(s, s, 1);

				const mat = el.sprite.material as THREE.SpriteMaterial;
				mat.opacity = el.baseOpacity * smoothIntensity;
				mat.color.copy(el.color);
				if (isLowSun) {
					mat.color.r *= warmShift;
					mat.color.g *= 0.85;
				}
			}

			// Streak
			camera.getWorldDirection(_camFwd);
			const sDirW = sunPos.clone().sub(camPos).normalize();
			streakSprite.position.copy(camPos).addScaledVector(sDirW, 50);
			const streakW = 30 + smoothIntensity * 15;
			streakSprite.scale.set(streakW, 1.2, 1);
			streakMat.opacity = smoothIntensity * 0.2 * (isLowSun ? 1.4 : 0.7);
			streakMat.color.set(isLowSun ? 0xffaa66 : 0xffeedd);
		},

		dispose() {
			for (const t of textures) t.dispose();
			streakMat.dispose();
			for (const el of elements) (el.sprite.material as THREE.Material).dispose();
		},
	};
}
