/**
 * First-person camera controller — raw mouse input, WASD, sprint,
 * terrain ground-following with smooth vertical lerp and head bob.
 */
import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

export interface FPController {
	controls: PointerLockControls;
	update(dt: number): void;
	lock(): void;
	unlock(): void;
	setHeightFn(fn: (x: number, z: number) => number): void;
	dispose(): void;
	isLocked(): boolean;
}

const MOVE_SPEED = 5.0;
const SPRINT_SPEED = 12.0;
const EYE_HEIGHT = 0.45;
const HEAD_BOB_SPEED = 9.0;
const HEAD_BOB_AMOUNT = 0.014;

export function createFPController(
	camera: THREE.PerspectiveCamera,
	domElement: HTMLElement
): FPController {
	const controls = new PointerLockControls(camera, domElement);

	const keys = {
		forward: false,
		backward: false,
		left: false,
		right: false,
		sprint: false
	};

	let heightFn: ((x: number, z: number) => number) | null = null;
	let bobPhase = 0;
	let isMoving = false;

	// Reusable vectors to avoid GC
	const _direction = new THREE.Vector3();
	const _forward = new THREE.Vector3();
	const _right = new THREE.Vector3();

	function onKeyDown(e: KeyboardEvent) {
		if (!controls.isLocked) return;
		switch (e.code) {
			case 'KeyW': case 'ArrowUp': keys.forward = true; break;
			case 'KeyS': case 'ArrowDown': keys.backward = true; break;
			case 'KeyA': case 'ArrowLeft': keys.left = true; break;
			case 'KeyD': case 'ArrowRight': keys.right = true; break;
			case 'ShiftLeft': case 'ShiftRight': keys.sprint = true; break;
		}
		if (['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code)) {
			e.preventDefault();
		}
	}

	function onKeyUp(e: KeyboardEvent) {
		switch (e.code) {
			case 'KeyW': case 'ArrowUp': keys.forward = false; break;
			case 'KeyS': case 'ArrowDown': keys.backward = false; break;
			case 'KeyA': case 'ArrowLeft': keys.left = false; break;
			case 'KeyD': case 'ArrowRight': keys.right = false; break;
			case 'ShiftLeft': case 'ShiftRight': keys.sprint = false; break;
		}
	}

	document.addEventListener('keydown', onKeyDown);
	document.addEventListener('keyup', onKeyUp);

	return {
		controls,

		update(dt: number) {
			if (!controls.isLocked) return;

			const speed = keys.sprint ? SPRINT_SPEED : MOVE_SPEED;
			const distance = speed * dt;

			camera.getWorldDirection(_forward);
			_forward.y = 0;
			_forward.normalize();
			_right.crossVectors(_forward, camera.up).normalize();

			_direction.set(0, 0, 0);
			if (keys.forward) _direction.add(_forward);
			if (keys.backward) _direction.sub(_forward);
			if (keys.right) _direction.add(_right);
			if (keys.left) _direction.sub(_right);

			isMoving = _direction.lengthSq() > 0.001;

			if (isMoving) {
				_direction.normalize();
				camera.position.addScaledVector(_direction, distance);
			}

			if (heightFn) {
				const groundY = heightFn(camera.position.x, camera.position.z);
				let targetY = groundY + EYE_HEIGHT;

				if (isMoving) {
					bobPhase += dt * HEAD_BOB_SPEED * (keys.sprint ? 1.4 : 1.0);
					targetY += Math.sin(bobPhase) * HEAD_BOB_AMOUNT;
				} else {
					// Smoothly decay bob
					bobPhase *= 0.9;
				}

				camera.position.y += (targetY - camera.position.y) * Math.min(1, dt * 14);
			}
		},

		lock() {
			// Raw mouse input — bypasses OS acceleration for snappy mouselook
			try {
				(domElement as any).requestPointerLock({ unadjustedMovement: true });
			} catch {
				controls.lock(); // fallback
			}
		},

		unlock() {
			controls.unlock();
		},

		setHeightFn(fn: (x: number, z: number) => number) {
			heightFn = fn;
		},

		dispose() {
			document.removeEventListener('keydown', onKeyDown);
			document.removeEventListener('keyup', onKeyUp);
			controls.dispose();
		},

		isLocked(): boolean {
			return controls.isLocked;
		}
	};
}
