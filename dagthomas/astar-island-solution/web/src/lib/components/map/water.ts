/**
 * Simple water plane — single flat surface covering the whole map.
 * Clean uniform color with subtle animated specular highlights.
 */
import * as THREE from 'three';
import { WATER_LEVEL } from './terrain';

const MAX_RIPPLES = 8;

export interface WaterSystem {
	mesh: THREE.Mesh;
	causticMesh: THREE.Mesh;
	update(dt: number, sunDir?: THREE.Vector3): void;
	addRipple(x: number, z: number): void;
	dispose(): void;
}

const WATER_VERT = /* glsl */ `
uniform float uTime;
uniform vec4 uRipples[${MAX_RIPPLES}];

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vFogDepth;

#define PI 3.14159265359

void main() {
	vec3 pos = position;

	// Very gentle vertical waves only (no horizontal displacement)
	float w = 0.0;
	w += sin(pos.x * 0.8 + uTime * 0.6) * 0.008;
	w += sin(pos.z * 1.1 - uTime * 0.4) * 0.006;
	w += sin((pos.x + pos.z) * 0.5 + uTime * 0.3) * 0.004;
	pos.y += w;

	// Player ripples
	for (int i = 0; i < ${MAX_RIPPLES}; i++) {
		if (uRipples[i].w > 0.0) {
			float age = uTime - uRipples[i].z;
			if (age > 0.0 && age < 4.0) {
				float dist = length(pos.xz - uRipples[i].xy);
				float ring = sin((dist - age * 3.0) * 6.0);
				float env = exp(-dist * 0.6) * exp(-age * 0.9) * uRipples[i].w;
				pos.y += ring * env * 0.03;
			}
		}
	}

	// Simple normal from wave derivatives
	float eps = 0.1;
	float hR = sin((pos.x + eps) * 0.8 + uTime * 0.6) * 0.008
	         + sin(pos.z * 1.1 - uTime * 0.4) * 0.006
	         + sin((pos.x + eps + pos.z) * 0.5 + uTime * 0.3) * 0.004;
	float hU = sin(pos.x * 0.8 + uTime * 0.6) * 0.008
	         + sin((pos.z + eps) * 1.1 - uTime * 0.4) * 0.006
	         + sin((pos.x + pos.z + eps) * 0.5 + uTime * 0.3) * 0.004;
	vNormal = normalize(vec3(w - hR, eps, w - hU));

	vWorldPos = (modelMatrix * vec4(pos, 1.0)).xyz;
	vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
	vFogDepth = -mvPos.z;
	gl_Position = projectionMatrix * mvPos;
}
`;

const WATER_FRAG = /* glsl */ `
uniform float uTime;
uniform vec3 uSunDir;
uniform vec3 uWaterColor;
uniform float uOpacity;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vFogDepth;

#ifdef USE_FOG
	uniform vec3 fogColor;
	uniform float fogDensity;
#endif

void main() {
	vec3 N = normalize(vNormal);
	vec3 V = normalize(cameraPosition - vWorldPos);

	// Fresnel for slight sky reflection at grazing angles
	float cosTheta = max(dot(V, N), 0.0);
	float fresnel = pow(1.0 - cosTheta, 4.0) * 0.3 + 0.05;

	// Specular highlight from sun
	vec3 H = normalize(V + uSunDir);
	float spec = pow(max(dot(N, H), 0.0), 80.0) * 0.6;

	// Simple color: base water color with slight sky blend at edges
	vec3 skyTint = vec3(0.55, 0.70, 0.85);
	vec3 color = mix(uWaterColor, skyTint, fresnel);
	color += vec3(spec);

	gl_FragColor = vec4(color, uOpacity);

	#ifdef USE_FOG
		float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
		gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
	#endif
}
`;

// Caustic floor shaders
const CAUSTIC_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
	vUv = uv;
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const CAUSTIC_FRAG = /* glsl */ `
uniform float uTime;
varying vec2 vUv;

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float noise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(
		mix(hash(i), hash(i + vec2(1, 0)), f.x),
		mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x),
		f.y
	);
}

void main() {
	vec2 uv = vUv * 25.0;
	float c1 = noise(uv + uTime * 0.2);
	float c2 = noise(uv * 1.3 - uTime * 0.15);
	float caustic = c1 * c2 * 3.0;
	vec3 col = vec3(0.15, 0.35, 0.40) * caustic;
	float alpha = caustic * 0.25;
	gl_FragColor = vec4(col, alpha);
}
`;

export function createWater(cols: number, rows: number, _grid?: number[][]): WaterSystem {
	// Single plane covering the whole map with subdivisions for wave animation
	const padding = 2;
	const segsX = (cols + padding * 2) * 2;
	const segsZ = (rows + padding * 2) * 2;
	const geometry = new THREE.PlaneGeometry(
		cols + padding * 2, rows + padding * 2, segsX, segsZ
	);
	geometry.rotateX(-Math.PI / 2);

	const rippleData = Array.from({ length: MAX_RIPPLES }, () => new THREE.Vector4(0, 0, -100, 0));

	const uniforms: Record<string, THREE.IUniform> = {
		uTime:       { value: 0 },
		uSunDir:     { value: new THREE.Vector3(0.5, 0.7, 0.3).normalize() },
		uWaterColor: { value: new THREE.Color(0x1a6e7a) },
		uOpacity:    { value: 0.75 },
		uRipples:    { value: rippleData },
		fogColor:    { value: new THREE.Color(0x8ab4cc) },
		fogDensity:  { value: 0.035 },
	};

	const material = new THREE.ShaderMaterial({
		uniforms,
		vertexShader: WATER_VERT,
		fragmentShader: WATER_FRAG,
		transparent: true,
		depthWrite: false,
		side: THREE.DoubleSide,
		fog: true,
	});

	const mesh = new THREE.Mesh(geometry, material);
	mesh.position.y = WATER_LEVEL;
	mesh.renderOrder = 2;

	// Caustic floor
	const causticGeo = new THREE.PlaneGeometry(cols + padding * 2, rows + padding * 2, 1, 1);
	causticGeo.rotateX(-Math.PI / 2);

	const causticUniforms: Record<string, THREE.IUniform> = {
		uTime: { value: 0 },
	};
	const causticMat = new THREE.ShaderMaterial({
		uniforms: causticUniforms,
		vertexShader: CAUSTIC_VERT,
		fragmentShader: CAUSTIC_FRAG,
		transparent: true,
		depthWrite: false,
		blending: THREE.AdditiveBlending,
	});

	const causticMesh = new THREE.Mesh(causticGeo, causticMat);
	causticMesh.position.y = -0.30;
	causticMesh.renderOrder = 1;

	let elapsed = 0;
	let rippleIdx = 0;

	return {
		mesh,
		causticMesh,

		update(dt: number, sunDir?: THREE.Vector3) {
			elapsed += dt;
			uniforms.uTime.value = elapsed;
			causticUniforms.uTime.value = elapsed;
			if (sunDir) {
				uniforms.uSunDir.value.copy(sunDir).normalize();
			}
		},

		addRipple(x: number, z: number) {
			rippleData[rippleIdx].set(x, z, elapsed, 0.7 + Math.random() * 0.5);
			rippleIdx = (rippleIdx + 1) % MAX_RIPPLES;
		},

		dispose() {
			geometry.dispose();
			material.dispose();
			causticGeo.dispose();
			causticMat.dispose();
		}
	};
}
