/**
 * Global wind system — gentle breeze with time-varying gusts.
 * Drives cloud movement and vegetation sway via shared uniforms.
 */
import * as THREE from 'three';

export type WeatherState = 'clear' | 'cloudy' | 'rain' | 'storm';

export interface WindSystem {
	/** Normalized XZ wind direction */
	direction: THREE.Vector2;
	/** Wind strength 0–1 */
	strength: number;
	/** Accumulated time for shader uniforms */
	time: number;
	/** Current weather state */
	weather: WeatherState;
	/** Storm intensity 0–1 (0 = calm, 1 = peak storm) */
	stormIntensity: number;
	/** Shared uniforms — attach to materials for GPU sway */
	uniforms: {
		uWindTime: { value: number };
		uWindDir: { value: THREE.Vector2 };
		uWindStrength: { value: number };
	};
	update(dt: number): void;
}

const MAX_DRIFT_DEG = 35; // max degrees wind can drift from base heading
const MAX_DRIFT_RAD = MAX_DRIFT_DEG * Math.PI / 180;

// Weather state durations (seconds)
const STATE_DURATIONS: Record<WeatherState, [number, number]> = {
	clear:  [60, 120],
	cloudy: [30, 60],
	rain:   [40, 80],
	storm:  [15, 35],
};

// Wind base strength per weather state
const STATE_WIND: Record<WeatherState, [number, number]> = {
	clear:  [0.15, 0.40],
	cloudy: [0.30, 0.50],
	rain:   [0.40, 0.65],
	storm:  [0.70, 1.00],
};

function randRange(lo: number, hi: number): number {
	return lo + Math.random() * (hi - lo);
}

export function createWind(): WindSystem {
	const direction = new THREE.Vector2(0.7, 0.7).normalize();
	let strength = 0.4;
	let time = 0;

	// Random initial heading
	const baseAngle = Math.random() * Math.PI * 2;
	let angle = baseAngle;
	let driftTarget = 0;
	let driftCurrent = 0;
	let nextShiftTime = 8 + Math.random() * 12;

	// Weather state machine — start at cloudy so first rain comes within ~30s
	let weather: WeatherState = 'cloudy';
	let stateTimer = randRange(15, 30);
	let stormIntensity = 0;
	let targetStormIntensity = 0;

	const uniforms = {
		uWindTime: { value: 0 },
		uWindDir: { value: direction },
		uWindStrength: { value: strength }
	};

	function advanceWeather() {
		switch (weather) {
			case 'clear':  weather = 'cloudy'; break;
			case 'cloudy': weather = 'rain'; break;
			case 'rain':
				// 30% chance of storm
				weather = Math.random() < 0.3 ? 'storm' : 'cloudy';
				break;
			case 'storm':  weather = 'rain'; break;
		}
		stateTimer = randRange(...STATE_DURATIONS[weather]);
		targetStormIntensity = weather === 'storm' ? 1.0 : weather === 'rain' ? 0.3 : 0;
	}

	return {
		direction,
		strength,
		time,
		weather,
		stormIntensity,
		uniforms,

		update(dt: number) {
			time += dt;

			// Weather state transitions
			stateTimer -= dt;
			if (stateTimer <= 0) advanceWeather();
			this.weather = weather;

			// Smooth storm intensity transitions
			stormIntensity += (targetStormIntensity - stormIntensity) * Math.min(1, dt * 0.15);
			this.stormIntensity = stormIntensity;

			// Direction drift
			nextShiftTime -= dt;
			if (nextShiftTime <= 0) {
				driftTarget = (Math.random() * 2 - 1) * MAX_DRIFT_RAD;
				nextShiftTime = weather === 'storm' ? 3 + Math.random() * 5 : 8 + Math.random() * 15;
			}
			driftCurrent += (driftTarget - driftCurrent) * Math.min(1, dt * 0.3);
			angle = baseAngle + driftCurrent;
			direction.set(Math.cos(angle), Math.sin(angle));

			// Wind strength based on weather state
			const [wLo, wHi] = STATE_WIND[weather];
			const baseStrength = (wLo + wHi) / 2;
			strength = baseStrength
				+ Math.sin(time * 0.4) * 0.12
				+ Math.sin(time * 1.1 + 2.0) * 0.08
				+ Math.sin(time * 2.7 + 5.0) * 0.04;
			// Storm gusts — sharp spikes
			if (weather === 'storm') {
				strength += Math.max(0, Math.sin(time * 3.5)) * 0.25;
			}
			strength = Math.max(wLo, Math.min(wHi + 0.15, strength));

			this.strength = strength;
			this.time = time;

			uniforms.uWindTime.value = time;
			uniforms.uWindDir.value = direction;
			uniforms.uWindStrength.value = strength;
		}
	};
}

/** Wind sway vertex shader injection — GLSL code inserted after #include <begin_vertex>.
 *  Uses a spatial "breath wave" that travels along the wind direction so trees
 *  sway in staggered waves rather than all at once. */
const WIND_SWAY_GLSL = /* glsl */ `
	{
		float localHeight = max(0.0, transformed.y);

		// World-space position of this instance (or object)
		#ifdef USE_INSTANCING
			float worldX = instanceMatrix[3][0];
			float worldZ = instanceMatrix[3][2];
		#else
			float worldX = modelMatrix[3][0];
			float worldZ = modelMatrix[3][2];
		#endif

		// Spatial breath wave — travels along wind direction
		// dot(worldPos, windDir) projects position onto wind axis
		// so trees downwind sway later than upwind trees
		float windDot = worldX * uWindDir.x + worldZ * uWindDir.y;
		float breathWave = sin(uWindTime * 1.2 - windDot * 0.25) * 0.5 + 0.5; // 0-1 pulse

		// Per-instance variation (small random offset so neighbors differ)
		float instancePhase = fract(worldX * 12.9898 + worldZ * 78.233) * 6.283;

		// Two gentle sway frequencies
		float phase = uWindTime * 1.5 + instancePhase;
		float sway1 = sin(phase) * 0.35;
		float sway2 = sin(phase * 0.6 + 2.1) * 0.18;

		// Height factor: cubic ramp so trunk base stays still, only canopy/top bends
		// localHeight near 0 (trunk base) → factor ≈ 0, higher up → factor grows fast
		float normH = clamp(localHeight * 3.0, 0.0, 1.0);
		float heightFactor = normH * normH * normH; // cubic: 0 at base, 1 at top

		// Combine: strength × breath pulse × height
		float swayAmount = (sway1 + sway2) * uWindStrength * breathWave * heightFactor;
		transformed.x += swayAmount * uWindDir.x;
		transformed.z += swayAmount * uWindDir.y;
	}
`;

/**
 * Inject wind sway into a MeshStandardMaterial via onBeforeCompile.
 * Works with both regular meshes and InstancedMesh.
 */
export function applyWindSway(
	material: THREE.MeshStandardMaterial,
	windUniforms: WindSystem['uniforms']
): void {
	material.onBeforeCompile = (shader) => {
		// Add wind uniforms
		shader.uniforms.uWindTime = windUniforms.uWindTime;
		shader.uniforms.uWindDir = windUniforms.uWindDir;
		shader.uniforms.uWindStrength = windUniforms.uWindStrength;

		// Declare uniforms at top of vertex shader
		shader.vertexShader = shader.vertexShader.replace(
			'void main() {',
			`uniform float uWindTime;
			uniform vec2 uWindDir;
			uniform float uWindStrength;
			void main() {`
		);

		// Inject sway after begin_vertex (which sets transformed = position)
		shader.vertexShader = shader.vertexShader.replace(
			'#include <begin_vertex>',
			`#include <begin_vertex>\n${WIND_SWAY_GLSL}`
		);
	};
	// Force shader recompilation
	material.needsUpdate = true;
}
