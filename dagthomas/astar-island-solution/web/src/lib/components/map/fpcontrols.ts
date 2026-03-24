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
	setOnExit(fn: () => void): void;
	setFreefly(enabled: boolean): void;
	dispose(): void;
	isLocked(): boolean;
}

const MOVE_SPEED = 1.8;
const SPRINT_SPEED = 4.5;
const EYE_HEIGHT = 0.45;
const HEAD_BOB_SPEED = 7.0;
const HEAD_BOB_AMOUNT = 0.025;
const HEAD_SWAY_AMOUNT = 0.002; // lateral sway (subtle)

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
		sprint: false,
		up: false,
		down: false
	};

	let heightFn: ((x: number, z: number) => number) | null = null;
	let onExitFn: (() => void) | null = null;
	let bobPhase = 0;
	let isMoving = false;
	let freefly = false;

	// Reusable vectors to avoid GC
	const _direction = new THREE.Vector3();
	const _forward = new THREE.Vector3();
	const _right = new THREE.Vector3();

	function onKeyDown(e: KeyboardEvent) {
		// Escape exits FP/flythrough mode entirely
		if (e.code === 'Escape' && onExitFn) {
			e.preventDefault();
			onExitFn();
			return;
		}
		if (!controls.isLocked) return;
		switch (e.code) {
			case 'KeyW': case 'ArrowUp': keys.forward = true; break;
			case 'KeyS': case 'ArrowDown': keys.backward = true; break;
			case 'KeyA': case 'ArrowLeft': keys.left = true; break;
			case 'KeyD': case 'ArrowRight': keys.right = true; break;
			case 'ShiftLeft': case 'ShiftRight': keys.sprint = true; break;
			case 'Space': keys.up = true; break;
			case 'KeyQ': case 'ControlLeft': case 'ControlRight': keys.down = true; break;
		}
		if (['KeyW','KeyA','KeyS','KeyD','KeyQ','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Space'].includes(e.code)) {
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
			case 'Space': keys.up = false; break;
			case 'KeyQ': case 'ControlLeft': case 'ControlRight': keys.down = false; break;
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
					const bobSpeed = keys.sprint ? 1.4 : 1.0;
					bobPhase += dt * HEAD_BOB_SPEED * bobSpeed;
					// Vertical bob — footstep rhythm
					targetY += Math.sin(bobPhase * 2) * HEAD_BOB_AMOUNT;
					// Lateral sway — shoulder rock
					camera.getWorldDirection(_forward);
					_right.crossVectors(_forward, camera.up).normalize();
					camera.position.addScaledVector(
						_right,
						Math.sin(bobPhase) * HEAD_SWAY_AMOUNT * bobSpeed
					);
				} else {
					bobPhase *= 0.92;
				}

				camera.position.y += (targetY - camera.position.y) * Math.min(1, dt * 10);
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

		setOnExit(fn: () => void) {
			onExitFn = fn;
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
