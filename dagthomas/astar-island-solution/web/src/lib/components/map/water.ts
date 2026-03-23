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
uniform float uRadius;

varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vFogDepth;

#ifdef USE_FOG
	uniform vec3 fogColor;
	uniform float fogDensity;
#endif

void main() {
	// Circular clip — discard pixels outside the island radius
	float dist = length(vWorldPos.xz);
	if (dist > uRadius) discard;
	// Soft edge fade near the rim
	float edgeFade = smoothstep(uRadius, uRadius - 0.5, dist);

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

	gl_FragColor = vec4(color, uOpacity * edgeFade);

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
	// Round water disc — PlaneGeometry for wave subdivisions, circular clip in fragment shader
	const radius = Math.sqrt(cols * cols + rows * rows) / 2 + 2;
	const size = radius * 2;
	const segs = Math.ceil(size) * 2;
	const geometry = new THREE.PlaneGeometry(size, size, segs, segs);
	geometry.rotateX(-Math.PI / 2);

	const rippleData = Array.from({ length: MAX_RIPPLES }, () => new THREE.Vector4(0, 0, -100, 0));

	const uniforms: Record<string, THREE.IUniform> = {
		uTime:       { value: 0 },
		uSunDir:     { value: new THREE.Vector3(0.5, 0.7, 0.3).normalize() },
		uWaterColor: { value: new THREE.Color(0x1a6e7a) },
		uOpacity:    { value: 0.75 },
		uRipples:    { value: rippleData },
		uRadius:     { value: radius },
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

	// Caustic floor (circular)
	const causticGeo = new THREE.CircleGeometry(radius, 48);
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

// ═══════════════════════════════════════
//  Moat + River/Waterfall system
// ═══════════════════════════════════════

export interface MoatSystem {
	group: THREE.Group;
	update(dt: number, sunDir?: THREE.Vector3): void;
	dispose(): void;
}

/* Simplified water vert — no ripples */
const MOAT_VERT = /* glsl */ `
uniform float uTime;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vFogDepth;
void main() {
	vec3 pos = position;
	float w = sin(pos.x * 0.8 + uTime * 0.6) * 0.008
	        + sin(pos.z * 1.1 - uTime * 0.4) * 0.006
	        + sin((pos.x + pos.z) * 0.5 + uTime * 0.3) * 0.004;
	pos.y += w;
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

/* Water frag with rounded-rectangle inner/outer clipping */
const MOAT_FRAG = /* glsl */ `
uniform float uTime;
uniform vec3 uSunDir;
uniform vec3 uWaterColor;
uniform float uOpacity;
uniform vec2 uInnerHalf;
uniform vec2 uOuterHalf;
uniform float uCornerR;
varying vec3 vWorldPos;
varying vec3 vNormal;
varying float vFogDepth;
#ifdef USE_FOG
	uniform vec3 fogColor;
	uniform float fogDensity;
#endif
// Signed distance to rounded rectangle (negative = inside)
float sdRoundBox(vec2 p, vec2 b, float r) {
	vec2 d = abs(p) - b + r;
	return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}
void main() {
	vec2 p = vWorldPos.xz;
	float innerDist = sdRoundBox(p, uInnerHalf, uCornerR * 0.5);
	if (innerDist < 0.0) discard;
	float outerDist = sdRoundBox(p, uOuterHalf, uCornerR);
	if (outerDist > 0.0) discard;
	vec3 N = normalize(vNormal);
	vec3 V = normalize(cameraPosition - vWorldPos);
	float cosTheta = max(dot(V, N), 0.0);
	float fresnel = pow(1.0 - cosTheta, 4.0) * 0.3 + 0.05;
	vec3 H = normalize(V + uSunDir);
	float spec = pow(max(dot(N, H), 0.0), 80.0) * 0.6;
	vec3 skyTint = vec3(0.55, 0.70, 0.85);
	vec3 color = mix(uWaterColor, skyTint, fresnel) + vec3(spec);
	float shoreFade = smoothstep(0.0, 0.5, innerDist);
	gl_FragColor = vec4(color, uOpacity * shoreFade);
	#ifdef USE_FOG
		float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
		gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
	#endif
}
`;

/* River / waterfall vert */
const RIVER_VERT = /* glsl */ `
uniform float uTime;
varying vec2 vUv;
varying float vFogDepth;
void main() {
	vUv = uv;
	vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
	vFogDepth = -mvPos.z;
	gl_Position = projectionMatrix * mvPos;
}
`;

/* River / waterfall frag — scrolling flow + foam */
const RIVER_FRAG = /* glsl */ `
uniform float uTime;
uniform vec3 uWaterColor;
varying vec2 vUv;
varying float vFogDepth;
#ifdef USE_FOG
	uniform vec3 fogColor;
	uniform float fogDensity;
#endif
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
	vec2 i = floor(p), f = fract(p);
	f = f * f * (3.0 - 2.0 * f);
	return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
	           mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
	vec2 uv = vUv;
	uv.y -= uTime * 0.6;
	float n = noise(uv * vec2(3.0, 8.0));
	float foam = smoothstep(0.3, 0.65, n);
	vec3 col = mix(uWaterColor, vec3(0.8, 0.9, 0.95), foam * 0.5);
	float edgeFade = smoothstep(0.0, 0.15, vUv.x) * smoothstep(1.0, 0.85, vUv.x);
	gl_FragColor = vec4(col, 0.75 * edgeFade);
	#ifdef USE_FOG
		float fogFactor = 1.0 - exp(-fogDensity * fogDensity * vFogDepth * vFogDepth);
		gl_FragColor.rgb = mix(gl_FragColor.rgb, fogColor, fogFactor);
	#endif
}
`;

/** Build a continuous river→waterfall strip following the cone surface */
function buildRiverFallGeo(
	dir: { dx: number; dz: number; startDist: number },
	width: number, topRadius: number, bottomRadius: number,
	coneHeight: number, moatY: number, fallHeight: number
): THREE.BufferGeometry {
	const riverSegs = 8, fallSegs = 16;
	const total = riverSegs + fallSegs;
	const riverLen = topRadius - dir.startDist - 0.3;
	const px = -dir.dz, pz = dir.dx; // perpendicular

	const pos: number[] = [], uvs: number[] = [], idx: number[] = [];
	for (let j = 0; j <= total; j++) {
		let x: number, y: number, z: number, w = width;
		if (j <= riverSegs) {
			const t = j / riverSegs;
			const d = dir.startDist + riverLen * t;
			x = dir.dx * d; z = dir.dz * d; y = moatY;
		} else {
			const t = (j - riverSegs) / fallSegs;
			y = moatY - t * fallHeight;
			const ncy = Math.max(0, Math.min(1, (y + coneHeight - 0.05) / coneHeight));
			const cr = bottomRadius + (topRadius - bottomRadius) * ncy + 0.1;
			x = dir.dx * cr; z = dir.dz * cr;
			w = width * (1 - t * 0.4);
		}
		const hw = w / 2;
		pos.push(x + px * hw, y, z + pz * hw, x - px * hw, y, z - pz * hw);
		uvs.push(0, j / total, 1, j / total);
	}
	for (let j = 0; j < total; j++) {
		const a = j * 2;
		idx.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
	}
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
	geo.setIndex(idx);
	geo.computeVertexNormals();
	return geo;
}

export function createMoatAndWaterfalls(cols: number, rows: number): MoatSystem {
	const group = new THREE.Group();
	const disposables: Array<{ dispose(): void }> = [];
	let elapsed = 0;

	const moatWidth = 2.5;
	const halfW = cols / 2, halfH = rows / 2;
	const innerW = halfW - 0.3, innerH = halfH - 0.3;
	const outerW = halfW + moatWidth, outerH = halfH + moatWidth;
	const moatY = 0.08;
	const riverWidth = 1.2;
	const topRadius = Math.sqrt(cols * cols + rows * rows) / 2 + 4;
	const bottomRadius = 2.0, coneHeight = 26.0, fallHeight = 12.0;

	// --- Moat ring (PlaneGeometry with inner/outer discard) ---
	const ps = Math.max(outerW, outerH) * 2 + 2;
	const segs = Math.ceil(ps) * 2;
	const moatGeo = new THREE.PlaneGeometry(ps, ps, segs, segs);
	moatGeo.rotateX(-Math.PI / 2);

	const moatU: Record<string, THREE.IUniform> = {
		uTime:       { value: 0 },
		uSunDir:     { value: new THREE.Vector3(0.5, 0.7, 0.3).normalize() },
		uWaterColor: { value: new THREE.Color(0x1a6e7a) },
		uOpacity:    { value: 0.72 },
		uInnerHalf:  { value: new THREE.Vector2(innerW, innerH) },
		uOuterHalf:  { value: new THREE.Vector2(outerW, outerH) },
		uCornerR:    { value: 2.0 },
		fogColor:    { value: new THREE.Color(0x8ab4cc) },
		fogDensity:  { value: 0.035 },
	};
	const moatMat = new THREE.ShaderMaterial({
		uniforms: moatU, vertexShader: MOAT_VERT, fragmentShader: MOAT_FRAG,
		transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: true,
	});
	const moatMesh = new THREE.Mesh(moatGeo, moatMat);
	moatMesh.position.y = moatY;
	moatMesh.renderOrder = 3;
	group.add(moatMesh);
	disposables.push(moatGeo, moatMat);

	// --- River-waterfall strips (4 cardinal directions) ---
	const riverU: Record<string, THREE.IUniform> = {
		uTime:       { value: 0 },
		uWaterColor: { value: new THREE.Color(0x2a8e9a) },
		fogColor:    { value: new THREE.Color(0x8ab4cc) },
		fogDensity:  { value: 0.035 },
	};
	const dirs = [
		{ dx: 0, dz: -1, startDist: outerH },
		{ dx: 0, dz:  1, startDist: outerH },
		{ dx:-1, dz:  0, startDist: outerW },
		{ dx: 1, dz:  0, startDist: outerW },
	];
	for (const d of dirs) {
		const g = buildRiverFallGeo(d, riverWidth, topRadius, bottomRadius, coneHeight, moatY, fallHeight);
		const m = new THREE.ShaderMaterial({
			uniforms: riverU, vertexShader: RIVER_VERT, fragmentShader: RIVER_FRAG,
			transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: true,
		});
		const mesh = new THREE.Mesh(g, m);
		mesh.renderOrder = 4;
		group.add(mesh);
		disposables.push(g, m);
	}

	return {
		group,
		update(dt, sunDir) {
			elapsed += dt;
			moatU.uTime.value = elapsed;
			riverU.uTime.value = elapsed;
			if (sunDir) moatU.uSunDir.value.copy(sunDir).normalize();
		},
		dispose() { for (const d of disposables) d.dispose(); },
	};
}
