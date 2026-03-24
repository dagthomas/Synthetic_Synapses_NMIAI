/**
 * Night sky: stars dome + aurora borealis curtain.
 * Both fade in/out based on nightFade from the day/night cycle.
 */
import * as THREE from 'three';

export interface NightSkySystem {
	group: THREE.Group;
	update(dt: number, nightFade: number): void;
	dispose(): void;
}

// ═══════════════════════════════════════
//  Stars — instanced points on a hemisphere
// ═══════════════════════════════════════

function createStars(): { mesh: THREE.Points; material: THREE.ShaderMaterial } {
	const COUNT = 1200;
	const positions = new Float32Array(COUNT * 3);
	const sizes = new Float32Array(COUNT);
	const twinklePhase = new Float32Array(COUNT);
	const R = 140;

	for (let i = 0; i < COUNT; i++) {
		// Uniform distribution on upper hemisphere
		const u = Math.random();
		const v = Math.random();
		const theta = 2 * Math.PI * u;
		const phi = Math.acos(1 - v * 0.85); // 0 to ~arccos(0.15) — upper dome only
		positions[i * 3] = R * Math.sin(phi) * Math.cos(theta);
		positions[i * 3 + 1] = R * Math.cos(phi); // Y is up
		positions[i * 3 + 2] = R * Math.sin(phi) * Math.sin(theta);
		sizes[i] = 0.5 + Math.random() * 1.5;
		twinklePhase[i] = Math.random() * Math.PI * 2;
	}

	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
	geo.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
	geo.setAttribute('aPhase', new THREE.BufferAttribute(twinklePhase, 1));

	const material = new THREE.ShaderMaterial({
		uniforms: {
			uTime: { value: 0 },
			uOpacity: { value: 0 },
		},
		vertexShader: /* glsl */ `
			attribute float aSize;
			attribute float aPhase;
			uniform float uTime;
			uniform float uOpacity;
			varying float vAlpha;
			void main() {
				float twinkle = 0.6 + 0.4 * sin(uTime * (1.0 + aSize * 0.3) + aPhase);
				vAlpha = uOpacity * twinkle;
				vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
				gl_PointSize = aSize * (200.0 / -mvPos.z);
				gl_Position = projectionMatrix * mvPos;
			}
		`,
		fragmentShader: /* glsl */ `
			varying float vAlpha;
			void main() {
				float d = length(gl_PointCoord - 0.5) * 2.0;
				float alpha = smoothstep(1.0, 0.2, d) * vAlpha;
				if (alpha < 0.01) discard;
				gl_FragColor = vec4(0.9, 0.92, 1.0, alpha);
			}
		`,
		transparent: true,
		depthWrite: false,
		blending: THREE.AdditiveBlending,
	});

	const mesh = new THREE.Points(geo, material);
	mesh.renderOrder = 80;
	return { mesh, material };
}

// ═══════════════════════════════════════
//  Aurora borealis — animated curtain mesh
// ═══════════════════════════════════════

const AURORA_VERT = /* glsl */ `
uniform float uTime;
varying vec2 vUv;
varying float vY;

void main() {
	vUv = uv;
	vec3 pos = position;

	// Gentle ripple along the curtain
	float wave = sin(pos.x * 0.3 + uTime * 0.4) * 1.5
	           + sin(pos.x * 0.7 - uTime * 0.25) * 0.8
	           + sin(pos.x * 0.12 + uTime * 0.15) * 2.5;
	pos.z += wave;
	pos.y += sin(pos.x * 0.15 + uTime * 0.1) * 1.0;

	vY = (pos.y - 30.0) / 35.0; // normalize height 0..1

	gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

const AURORA_FRAG = /* glsl */ `
uniform float uTime;
uniform float uOpacity;
varying vec2 vUv;
varying float vY;

// Simple noise
float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
	vec2 i = floor(p), f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(
		mix(hash(i), hash(i + vec2(1, 0)), f.x),
		mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x),
		f.y
	);
}

void main() {
	// Vertical fade: bright at base, fading up
	float baseFade = smoothstep(0.0, 0.15, vY) * smoothstep(1.0, 0.4, vY);

	// Animated curtain pattern
	vec2 uv = vUv;
	float n1 = noise(vec2(uv.x * 3.0 + uTime * 0.08, uv.y * 2.0 - uTime * 0.12));
	float n2 = noise(vec2(uv.x * 5.0 - uTime * 0.05, uv.y * 4.0 + uTime * 0.06));
	float curtain = n1 * 0.6 + n2 * 0.4;
	curtain = smoothstep(0.25, 0.7, curtain);

	// Vertical streaks
	float streak = noise(vec2(uv.x * 12.0 + uTime * 0.03, uTime * 0.02));
	streak = smoothstep(0.4, 0.8, streak) * 0.5;

	float intensity = (curtain + streak) * baseFade;

	// Aurora colors: green core, teal-cyan edges, purple-pink top
	vec3 green = vec3(0.1, 0.9, 0.3);
	vec3 teal  = vec3(0.1, 0.6, 0.8);
	vec3 purple = vec3(0.5, 0.15, 0.7);

	vec3 col = mix(green, teal, smoothstep(0.2, 0.6, vY + n1 * 0.2));
	col = mix(col, purple, smoothstep(0.5, 0.9, vY));

	// Brighten the core
	col += vec3(0.05, 0.15, 0.05) * curtain;

	float alpha = intensity * uOpacity * 0.6;
	if (alpha < 0.005) discard;

	gl_FragColor = vec4(col * (1.0 + intensity * 0.5), alpha);
}
`;

function createAurora(): { mesh: THREE.Mesh; material: THREE.ShaderMaterial } {
	// Wide curtain arc across the northern sky
	const width = 120;
	const height = 35;
	const segsX = 80;
	const segsY = 20;
	const geo = new THREE.PlaneGeometry(width, height, segsX, segsY);

	// Curve into an arc
	const positions = geo.attributes.position;
	const arcRadius = 90;
	for (let i = 0; i < positions.count; i++) {
		const x = positions.getX(i);
		const y = positions.getY(i);
		// Wrap x around a circular arc
		const angle = (x / width) * 1.2 - 0.6; // ~±35 degrees
		const r = arcRadius;
		positions.setX(i, Math.sin(angle) * r);
		positions.setZ(i, -Math.cos(angle) * r + arcRadius * 0.5);
		positions.setY(i, y + 48); // high in the sky
	}
	positions.needsUpdate = true;
	geo.computeVertexNormals();

	const material = new THREE.ShaderMaterial({
		uniforms: {
			uTime: { value: 0 },
			uOpacity: { value: 0 },
		},
		vertexShader: AURORA_VERT,
		fragmentShader: AURORA_FRAG,
		transparent: true,
		depthWrite: false,
		blending: THREE.AdditiveBlending,
		side: THREE.DoubleSide,
	});

	const mesh = new THREE.Mesh(geo, material);
	mesh.renderOrder = 81;
	return { mesh, material };
}

// ═══════════════════════════════════════
//  Combined system
// ═══════════════════════════════════════

export function createNightSky(): NightSkySystem {
	const group = new THREE.Group();
	let elapsed = 0;

	const stars = createStars();
	group.add(stars.mesh);

	const aurora = createAurora();
	group.add(aurora.mesh);

	return {
		group,

		update(dt: number, nightFade: number) {
			elapsed += dt;

			// Stars fade in fully during night
			const starOpacity = Math.min(1, nightFade * 1.8) * 0.9;
			stars.material.uniforms.uTime.value = elapsed;
			stars.material.uniforms.uOpacity.value = starOpacity;
			stars.mesh.visible = starOpacity > 0.01;

			// Aurora only appears during deep night (nightFade > 0.7)
			const auroraFade = Math.max(0, (nightFade - 0.7) / 0.3);
			const auroraOpacity = auroraFade * auroraFade; // ease-in
			aurora.material.uniforms.uTime.value = elapsed;
			aurora.material.uniforms.uOpacity.value = auroraOpacity;
			aurora.mesh.visible = auroraOpacity > 0.01;

			// Slowly rotate aurora for variety
			aurora.mesh.rotation.y = Math.sin(elapsed * 0.02) * 0.15;
		},

		dispose() {
			stars.mesh.geometry.dispose();
			stars.material.dispose();
			aurora.mesh.geometry.dispose();
			aurora.material.dispose();
		},
	};
}
