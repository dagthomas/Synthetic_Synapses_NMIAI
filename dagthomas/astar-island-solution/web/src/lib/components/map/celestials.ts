// Sun glow sprite + Moon with real lunar phase calculation

import * as THREE from 'three';

// --- Moon phase calculation ---
// Reference new moon: Jan 6, 2000 18:14 UTC (Julian day 2451550.1)
const SYNODIC_MONTH = 29.53059; // days
const REF_NEW_MOON = new Date('2000-01-06T18:14:00Z').getTime();

/** Returns 0..1 where 0 = new moon, 0.5 = full moon, 1 = next new moon */
export function getMoonPhase(date: Date = new Date()): number {
	const daysSinceRef = (date.getTime() - REF_NEW_MOON) / (1000 * 60 * 60 * 24);
	const phase = ((daysSinceRef % SYNODIC_MONTH) + SYNODIC_MONTH) % SYNODIC_MONTH;
	return phase / SYNODIC_MONTH;
}

// --- Procedural textures ---

function createGlowTexture(size = 128): THREE.Texture {
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext('2d')!;
	const cx = size / 2;
	const grad = ctx.createRadialGradient(cx, cx, 0, cx, cx, cx);
	grad.addColorStop(0, 'rgba(255,255,220,1.0)');
	grad.addColorStop(0.15, 'rgba(255,240,180,0.6)');
	grad.addColorStop(0.4, 'rgba(255,200,100,0.15)');
	grad.addColorStop(1, 'rgba(255,180,80,0)');
	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, size, size);
	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	return tex;
}

function createMoonTexture(size = 128, phase: number): THREE.Texture {
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext('2d')!;
	const cx = size / 2;
	const r = cx - 4;

	// Draw full bright moon disc
	ctx.fillStyle = '#e8e4d4';
	ctx.beginPath();
	ctx.arc(cx, cx, r, 0, Math.PI * 2);
	ctx.fill();

	// Add subtle craters
	ctx.fillStyle = 'rgba(180,175,160,0.4)';
	const craters = [
		{ x: 0.35, y: 0.3, r: 0.12 },
		{ x: 0.6, y: 0.45, r: 0.08 },
		{ x: 0.45, y: 0.65, r: 0.1 },
		{ x: 0.7, y: 0.3, r: 0.06 },
		{ x: 0.3, y: 0.55, r: 0.07 },
		{ x: 0.55, y: 0.2, r: 0.05 },
		{ x: 0.65, y: 0.7, r: 0.09 }
	];
	for (const c of craters) {
		ctx.beginPath();
		ctx.arc(cx + (c.x - 0.5) * r * 2, cx + (c.y - 0.5) * r * 2, c.r * r, 0, Math.PI * 2);
		ctx.fill();
	}

	// Phase shadow — draw shadow over part of the moon
	// phase 0 = new (all shadow), 0.25 = first quarter, 0.5 = full (no shadow), 0.75 = last quarter
	if (phase < 0.03 || phase > 0.97) {
		// Near new moon: almost fully shadowed
		ctx.fillStyle = 'rgba(10,10,30,0.95)';
		ctx.beginPath();
		ctx.arc(cx, cx, r + 1, 0, Math.PI * 2);
		ctx.fill();
	} else if (Math.abs(phase - 0.5) > 0.03) {
		// Partial phase: use clipping with an ellipse to shadow one side
		ctx.save();
		ctx.globalCompositeOperation = 'source-atop';
		ctx.fillStyle = 'rgba(10,10,30,0.92)';

		// Determine shadow coverage
		// 0-0.5: waxing (shadow on left shrinking), 0.5-1: waning (shadow on right growing)
		const illumination = phase <= 0.5 ? phase * 2 : (1 - phase) * 2; // 0=none, 1=full
		const shadowSide = phase <= 0.5 ? -1 : 1; // -1=left shadow, 1=right shadow

		ctx.beginPath();
		// Shadow semicircle on one side
		const startAngle = shadowSide === -1 ? Math.PI / 2 : -Math.PI / 2;
		const endAngle = shadowSide === -1 ? -Math.PI / 2 : Math.PI / 2;
		ctx.arc(cx, cx, r + 1, startAngle, endAngle, shadowSide === -1);

		// Terminator: ellipse across the moon
		// At illumination=0, ellipse is full circle (all shadow). At illumination=1, ellipse is a line (no shadow).
		const terminatorX = (illumination - 0.5) * 2 * r * shadowSide;
		ctx.ellipse(cx + terminatorX * 0.5, cx, Math.abs(r * (1 - illumination)), r + 1, 0,
			shadowSide === -1 ? -Math.PI / 2 : Math.PI / 2,
			shadowSide === -1 ? Math.PI / 2 : -Math.PI / 2,
			shadowSide === 1
		);
		ctx.closePath();
		ctx.fill();
		ctx.restore();
	}

	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	return tex;
}

function createMoonGlowTexture(size = 128): THREE.Texture {
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const ctx = canvas.getContext('2d')!;
	const cx = size / 2;
	const grad = ctx.createRadialGradient(cx, cx, size * 0.1, cx, cx, cx);
	grad.addColorStop(0, 'rgba(200,210,255,0.3)');
	grad.addColorStop(0.3, 'rgba(180,190,230,0.1)');
	grad.addColorStop(1, 'rgba(150,160,200,0)');
	ctx.fillStyle = grad;
	ctx.fillRect(0, 0, size, size);
	const tex = new THREE.CanvasTexture(canvas);
	tex.needsUpdate = true;
	return tex;
}

// --- Celestial system ---

export interface CelestialSystem {
	sunSprite: THREE.Sprite;
	moonGroup: THREE.Group;
	moonSprite: THREE.Sprite;
	moonGlow: THREE.Sprite;
	moonLight: THREE.PointLight;
	update(sunPos: THREE.Vector3, moonPos: THREE.Vector3, nightFade: number): void;
	dispose(): void;
}

export function createCelestials(scene: THREE.Scene): CelestialSystem {
	// --- Sun glow sprite ---
	const sunTex = createGlowTexture(256);
	const sunMat = new THREE.SpriteMaterial({
		map: sunTex,
		transparent: true,
		blending: THREE.AdditiveBlending,
		depthWrite: false,
		fog: false,
		color: new THREE.Color(2.5, 2.2, 1.8) // HDR bright — triggers bloom
	});
	const sunSprite = new THREE.Sprite(sunMat);
	sunSprite.scale.set(28, 28, 1);
	sunSprite.renderOrder = 90;
	scene.add(sunSprite);

	// --- Moon ---
	const phase = getMoonPhase();
	const moonTex = createMoonTexture(256, phase);
	const moonMat = new THREE.SpriteMaterial({
		map: moonTex,
		transparent: true,
		depthWrite: false,
		fog: false
	});
	const moonSprite = new THREE.Sprite(moonMat);
	moonSprite.scale.set(8, 8, 1);
	moonSprite.renderOrder = 91;

	// Moon glow
	const moonGlowTex = createMoonGlowTexture(256);
	const moonGlowMat = new THREE.SpriteMaterial({
		map: moonGlowTex,
		transparent: true,
		blending: THREE.AdditiveBlending,
		depthWrite: false,
		fog: false,
		color: new THREE.Color(1.5, 1.6, 2.0) // HDR cool blue — subtle bloom
	});
	const moonGlow = new THREE.Sprite(moonGlowMat);
	moonGlow.scale.set(18, 18, 1);
	moonGlow.renderOrder = 89;

	// Moon light — soft bluish point light
	const moonLight = new THREE.PointLight(0x6688cc, 0, 200);

	// Moon anamorphic streak — horizontal 35mm cinema look
	const streakCanvas = document.createElement('canvas');
	streakCanvas.width = 512;
	streakCanvas.height = 16;
	const sCtx = streakCanvas.getContext('2d')!;
	const sGrad = sCtx.createLinearGradient(0, 8, 512, 8);
	sGrad.addColorStop(0, 'rgba(255,255,255,0)');
	sGrad.addColorStop(0.2, 'rgba(200,220,255,0.08)');
	sGrad.addColorStop(0.5, 'rgba(200,220,255,0.25)');
	sGrad.addColorStop(0.8, 'rgba(200,220,255,0.08)');
	sGrad.addColorStop(1, 'rgba(255,255,255,0)');
	sCtx.fillStyle = sGrad;
	sCtx.fillRect(0, 0, 512, 16);
	const streakTex = new THREE.CanvasTexture(streakCanvas);
	const moonStreakMat = new THREE.SpriteMaterial({
		map: streakTex,
		transparent: true,
		blending: THREE.AdditiveBlending,
		depthWrite: false,
		fog: false,
		color: new THREE.Color(0.6, 0.7, 1.0),
	});
	const moonStreak = new THREE.Sprite(moonStreakMat);
	moonStreak.scale.set(50, 1.2, 1);
	moonStreak.renderOrder = 88;

	const moonGroup = new THREE.Group();
	moonGroup.add(moonSprite);
	moonGroup.add(moonGlow);
	moonGroup.add(moonStreak);
	moonGroup.add(moonLight);
	scene.add(moonGroup);

	// Brightness based on phase (full moon = brightest)
	const phaseBrightness = Math.sin(phase * Math.PI); // 0 at new, 1 at full

	return {
		sunSprite,
		moonGroup,
		moonSprite,
		moonGlow,
		moonLight,

		update(sunPos: THREE.Vector3, moonPos: THREE.Vector3, nightFade: number) {
			// Sun: world-space position (large orbit radius keeps it in the sky)
			sunSprite.position.copy(sunPos);
			sunMat.opacity = Math.max(0, 1 - nightFade * 1.5);

			// Moon: world-space position
			moonGroup.position.copy(moonPos);
			const moonOpacity = Math.min(1, nightFade * 1.5) * Math.max(0.2, phaseBrightness);
			moonMat.opacity = moonOpacity;
			moonGlowMat.opacity = moonOpacity * 0.8;
			moonStreakMat.opacity = moonOpacity * 0.5;
			moonLight.intensity = moonOpacity * 1.2;
		},

		dispose() {
			sunTex.dispose();
			sunMat.dispose();
			moonTex.dispose();
			moonMat.dispose();
			moonGlowTex.dispose();
			moonGlowMat.dispose();
			scene.remove(sunSprite);
			scene.remove(moonGroup);
		}
	};
}
