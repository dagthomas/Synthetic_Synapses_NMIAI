/**
 * Atmospheric enhancements for first-person mode:
 * - Player-following fill light (warm/cool based on time)
 *
 * Vegetation particles removed — replaced by GLB model scatter system.
 */
import * as THREE from 'three';

export interface AtmosphereSystem {
	playerLight: THREE.PointLight;
	update(camera: THREE.PerspectiveCamera, dt: number, timeOfDay: number): void;
	dispose(): void;
}

export function createAtmosphere(
	scene: THREE.Scene,
	_grid: number[][],
	_heightFn: (x: number, z: number) => number
): AtmosphereSystem {
	// Player-following fill light
	const playerLight = new THREE.PointLight(0xffeedd, 0.5, 10, 1.2);
	playerLight.castShadow = false;
	scene.add(playerLight);

	const _camDir = new THREE.Vector3();

	return {
		playerLight,

		update(camera: THREE.PerspectiveCamera, _dt: number, timeOfDay: number) {
			camera.getWorldDirection(_camDir);
			playerLight.position.copy(camera.position);
			playerLight.position.y += 0.2;
			playerLight.position.addScaledVector(_camDir, -0.4);

			const isNight = timeOfDay < 6 || timeOfDay > 18;
			const isDawnDusk = (timeOfDay >= 5 && timeOfDay <= 7) || (timeOfDay >= 17 && timeOfDay <= 19);

			if (isNight) {
				playerLight.intensity = 1.0;
				playerLight.color.setHex(0x8899cc);
				playerLight.distance = 12;
			} else if (isDawnDusk) {
				playerLight.intensity = 0.6;
				playerLight.color.setHex(0xffbb66);
				playerLight.distance = 10;
			} else {
				playerLight.intensity = 0.35;
				playerLight.color.setHex(0xffeedd);
				playerLight.distance = 8;
			}
		},

		dispose() {
			scene.remove(playerLight);
			playerLight.dispose();
		}
	};
}
