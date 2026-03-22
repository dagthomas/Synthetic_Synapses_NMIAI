/**
 * First-person camera controller with WASD movement, pointer lock mouse look,
 * and terrain ground-following.
 */
import * as THREE from 'three';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

export interface FPController {
	controls: PointerLockControls;
	/** Call every frame with delta time */
	update(dt: number): void;
	/** Lock pointer and start FP mode */
	lock(): void;
	/** Unlock pointer */
	unlock(): void;
	/** Set the ground-height function */
	setHeightFn(fn: (x: number, z: number) => number): void;
	/** Clean up listeners */
	dispose(): void;
	/** Whether pointer is currently locked */
	isLocked(): boolean;
}

const MOVE_SPEED = 4.0;
const SPRINT_SPEED = 8.0;
const EYE_HEIGHT = 0.5; // world-scale eye height above terrain
const HEAD_BOB_SPEED = 8.0;
const HEAD_BOB_AMOUNT = 0.012;

export function createFPController(
	camera: THREE.PerspectiveCamera,
	domElement: HTMLElement
): FPController {
	const controls = new PointerLockControls(camera, domElement);

	// Movement state
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

	function onKeyDown(e: KeyboardEvent) {
		if (!controls.isLocked) return;
		switch (e.code) {
			case 'KeyW':
			case 'ArrowUp':
				keys.forward = true;
				break;
			case 'KeyS':
			case 'ArrowDown':
				keys.backward = true;
				break;
			case 'KeyA':
			case 'ArrowLeft':
				keys.left = true;
				break;
			case 'KeyD':
			case 'ArrowRight':
				keys.right = true;
				break;
			case 'ShiftLeft':
			case 'ShiftRight':
				keys.sprint = true;
				break;
		}
		// Prevent default for movement keys to avoid page scroll
		if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
			e.preventDefault();
		}
	}

	function onKeyUp(e: KeyboardEvent) {
		switch (e.code) {
			case 'KeyW':
			case 'ArrowUp':
				keys.forward = false;
				break;
			case 'KeyS':
			case 'ArrowDown':
				keys.backward = false;
				break;
			case 'KeyA':
			case 'ArrowLeft':
				keys.left = false;
				break;
			case 'KeyD':
			case 'ArrowRight':
				keys.right = false;
				break;
			case 'ShiftLeft':
			case 'ShiftRight':
				keys.sprint = false;
				break;
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

			// Build movement vector in camera's XZ plane
			const direction = new THREE.Vector3();
			const forward = new THREE.Vector3();
			camera.getWorldDirection(forward);
			forward.y = 0;
			forward.normalize();

			const right = new THREE.Vector3();
			right.crossVectors(forward, camera.up).normalize();

			if (keys.forward) direction.add(forward);
			if (keys.backward) direction.sub(forward);
			if (keys.right) direction.add(right);
			if (keys.left) direction.sub(right);

			isMoving = direction.lengthSq() > 0.001;

			if (isMoving) {
				direction.normalize();
				camera.position.addScaledVector(direction, distance);
			}

			// Ground following
			if (heightFn) {
				const groundY = heightFn(camera.position.x, camera.position.z);
				let targetY = groundY + EYE_HEIGHT;

				// Head bob while moving
				if (isMoving) {
					bobPhase += dt * HEAD_BOB_SPEED * (keys.sprint ? 1.3 : 1.0);
					targetY += Math.sin(bobPhase) * HEAD_BOB_AMOUNT;
				} else {
					bobPhase = 0;
				}

				// Smooth vertical following
				camera.position.y += (targetY - camera.position.y) * Math.min(1, dt * 12);
			}
		},

		lock() {
			controls.lock();
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
