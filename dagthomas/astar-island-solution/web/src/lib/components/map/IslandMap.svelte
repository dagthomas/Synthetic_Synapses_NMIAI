<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import * as THREE from 'three';
	import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
	import { findClusters, type Cluster } from './clusters';
	import type { Settlement } from '$lib/types';
	import { TerrainCode } from '$lib/types';
	import { mulberry32, cellSeed } from './prng';
	import { computeDayNight } from './daynight';
	import { createCloudSystem, type CloudSystem } from './clouds';
	import { findWaterfallSources, createWaterfallSystem, type WaterfallSystem } from './waterfalls';
	import { createCelestials, type CelestialSystem } from './celestials';
	import { createWildlifeSystem, type WildlifeSystem } from './wildlife';
	import { preloadModels, placeModel } from './models';
	import { createWeatherSystem, type WeatherSystem } from './weather';
	import { createTerrain, type TerrainSystem } from './terrain';
	import { createFPController, type FPController } from './fpcontrols';
	import { createPostFX, type PostFXPipeline } from './postfx';
	import { createAtmosphere, type AtmosphereSystem } from './atmosphere';
	import { Sky } from 'three/addons/objects/Sky.js';
	import { createFlythrough, type FlythroughSystem } from './flythrough';
	import { createScatter, type ScatterSystem } from './scatter';
	import { createCreatures, type CreatureSystem } from './creatures';
	import { createWater, type WaterSystem } from './water';
	import { LightProbeGenerator } from 'three/addons/lights/LightProbeGenerator.js';

	let {
		grid,
		settlements = [],
		showScores = false,
		timeOfDay = 12,
		freezeCamera = false,
		seedLabel = '',
		roundLabel = '',
		onCaptureFn = undefined,
		onFlythroughChange = undefined
	}: {
		grid: number[][];
		settlements?: Settlement[];
		showScores?: boolean;
		timeOfDay?: number;
		freezeCamera?: boolean;
		seedLabel?: string;
		roundLabel?: string;
		onCaptureFn?: (fn: () => string) => void;
		onFlythroughChange?: (active: boolean) => void;
	} = $props();

	let container: HTMLDivElement;
	let renderer: THREE.WebGLRenderer;
	let scene: THREE.Scene;
	let _realSceneAdd: (...args: THREE.Object3D[]) => THREE.Scene; // saved once in onMount
	let camera: THREE.PerspectiveCamera;
	let controls: OrbitControls;
	let animationId: number;
	let oceanMesh: THREE.Mesh | null = null;
	let oceanTime = 0;
	let lastTime = 0;

	let hemiLight: THREE.HemisphereLight | null = null;
	let sunLight: THREE.DirectionalLight | null = null;
	let ambientLight: THREE.AmbientLight | null = null;

	let cloudSystem: CloudSystem | null = null;
	let waterfallSystem: WaterfallSystem | null = null;
	let celestialSystem: CelestialSystem | null = null;
	let wildlifeSystem: WildlifeSystem | null = null;
	let weatherSystem: WeatherSystem | null = null;
	let fillLight: THREE.DirectionalLight | null = null;
	let lightProbe: THREE.LightProbe | null = null;
	let cubeCamera: THREE.CubeCamera | null = null;
	let cubeRenderTarget: THREE.WebGLCubeRenderTarget | null = null;
	let lightProbeNeedsUpdate = false;
	let settlementLights: THREE.PointLight[] = [];
	let fireParticles: THREE.Mesh[] = [];
	let settlementPositions: THREE.Vector3[] = [];
	let lastInteraction = 0;
	let autoRotateAngle = 0;
	let buildGeneration = 0; // incremented each build to cancel stale async work
	let prevTimeOfDay = -1;
	let cachedDN: ReturnType<typeof computeDayNight> | null = null;
	let lastShadowUpdate = 0;
	let fpsFrameCount = 0;
	let fpsLastTime = 0;
	let fpsDisplay = $state('');
	let drawCallsDisplay = $state('');
	let displayTime = 12;

	// First-person mode state
	let fpMode = $state(false);
	let fpController: FPController | null = $state(null);
	let terrainSystem: TerrainSystem | null = null;
	let blockyTerrainGroup: THREE.Group | null = null;
	let fpFog: THREE.FogExp2 | null = null;
	let fpOverlay: HTMLDivElement = $state() as HTMLDivElement;
	let postFX: PostFXPipeline | null = null;
	let atmosphere: AtmosphereSystem | null = null;
	let fpSky: Sky | null = null;
	let fpSkyUniforms: Record<string, { value: any }> | null = null;
	let flythrough: FlythroughSystem | null = null;
	let flythroughActive = $state(false);
	let flythroughLoading = $state(false);
	let flythroughLoadProgress = $state(0);
	let scatter: ScatterSystem | null = null;
	let creatureSystem: CreatureSystem | null = null;
	let waterSystem: WaterSystem | null = null;
	let fpAudio: HTMLAudioElement | null = null;
	let pointerLocked = $state(false);
	let lastCamX: number | null = null;
	let lastCamZ: number | null = null;
	let rippleCooldown = 0;

	// --- Cross-fade transition state ---
	let worldGroup: THREE.Group | null = null;
	let prevWorldGroup: THREE.Group | null = null;
	let prevTerrainSystem: TerrainSystem | null = null;
	let prevWaterSystem: WaterSystem | null = null;
	let transitionAlpha = 1.0;
	let transitioning = false;
	const TRANSITION_DURATION = 2.0;

	// --- Terrain colors matching reference image ---
	// Reference: sandy tan base, green grass patches, blue water, gray stone
	const terrainMaterials: Record<number, THREE.MeshStandardMaterial> = {};
	function getTerrainMaterial(code: number): THREE.MeshStandardMaterial {
		if (!terrainMaterials[code]) {
			const colors: Record<number, number> = {
				0: 0xc9b882,   // Empty: sandy tan
				1: 0xc4a862,   // Settlement: warm tan
				2: 0x4a9ec9,   // Port: blue water
				3: 0x9a9080,   // Ruin: gray-brown stone
				4: 0x5a8a3a,   // Forest: green
				5: 0x8a8a82,   // Mountain: gray rock
				10: 0x3a88b8,  // Ocean: deep blue
				11: 0x7ab648   // Plains: light green
			};
			terrainMaterials[code] = new THREE.MeshStandardMaterial({
				color: colors[code] ?? 0xc9b882,
				roughness: 0.85,
				metalness: 0.02
			});
		}
		return terrainMaterials[code];
	}

	function getTerrainHeight(code: number): number {
		const heights: Record<number, number> = {
			0: 0.06, 1: 0.12, 2: 0.04, 3: 0.08, 4: 0.10,
			5: 0.20, 10: 0.05, 11: 0.08
		};
		return heights[code] ?? 0.06;
	}

	/** Ground height helper — uses heightmap terrain */
	function gY(wx: number, wz: number): number {
		return terrainSystem ? terrainSystem.getHeightAt(wx, wz) : 0.1;
	}

	/** Cleanup grid-dependent systems only (terrain, creatures, scatter, water) */
	function cleanupGridSystems() {
		if (terrainSystem) { terrainSystem.dispose(); terrainSystem = null; }
		if (creatureSystem) { creatureSystem.dispose(); creatureSystem = null; }
		if (scatter) { scatter.dispose(); scatter = null; }
		if (waterSystem) { waterSystem.dispose(); waterSystem = null; }
		blockyTerrainGroup = null;
		settlementLights = [];
		fireParticles = [];
		settlementPositions = [];
	}

	/** Set opacity on all meshes in a group for cross-fade transitions */
	function setGroupOpacity(group: THREE.Group | null, alpha: number) {
		if (!group) return;
		const fullyOpaque = alpha >= 0.999;
		group.traverse((obj) => {
			if (!(obj instanceof THREE.Mesh) && !(obj instanceof THREE.InstancedMesh)) return;
			const mat = obj.material;
			if (!mat) return;
			const mats = Array.isArray(mat) ? mat : [mat];
			for (const m of mats) {
				if (m.isMeshStandardMaterial || m.isMeshBasicMaterial) {
					m.transparent = !fullyOpaque;
					m.opacity = alpha;
					m.needsUpdate = true;
				} else if (m.isShaderMaterial && m.uniforms?.uOpacity) {
					// Scale the shader's base opacity by alpha (water base opacity is 0.72)
					m.uniforms.uOpacity.value = 0.72 * alpha;
					m.transparent = true; // water is always transparent
				}
			}
		});
	}

	/** Dispose all geometry and materials in a group */
	function disposeGroup(group: THREE.Group) {
		group.traverse((obj) => {
			if (obj instanceof THREE.Mesh || obj instanceof THREE.InstancedMesh) {
				obj.geometry?.dispose();
				const mat = obj.material;
				if (mat) {
					const mats = Array.isArray(mat) ? mat : [mat];
					for (const m of mats) m.dispose();
				}
			}
		});
	}

	// === First-person mode toggle ===

	/** Visual-only FP setup: terrain, sky, fog, atmosphere, shadows, camera.
	 *  Does NOT create FP controller or lock pointer (used by flythrough).
	 *  Does NOT change postFX mode or scene.background — caller activates those
	 *  after camera/flythrough are ready to avoid black frames during loading. */
	function enterFPMode_visual() {
		if (!scene || !camera || !renderer || !grid?.length) return;
		fpMode = true;

		// Terrain already visible — just ensure it exists
		if (!terrainSystem) {
			terrainSystem = createTerrain(grid);
			scene.add(terrainSystem.mesh);
		}

		// Camera: wide FOV, close near plane for immersion
		camera.fov = 100;
		camera.near = 0.01;
		camera.far = 300;
		camera.rotation.order = 'YXZ';
		camera.updateProjectionMatrix();

		// Position camera at center of map at eye height, reset rotation
		const groundY = terrainSystem ? terrainSystem.getHeightAt(0, 0) : 0.1;
		camera.position.set(0, groundY + 0.45, 0);
		camera.rotation.set(0, 0, 0);

		// Preetham atmospheric sky shader — replaces flat sky color
		if (!fpSky) {
			fpSky = new Sky();
			fpSky.scale.setScalar(450);
			fpSkyUniforms = fpSky.material.uniforms;
			fpSkyUniforms['turbidity'].value = 4;
			fpSkyUniforms['rayleigh'].value = 2;
			fpSkyUniforms['mieCoefficient'].value = 0.005;
			fpSkyUniforms['mieDirectionalG'].value = 0.8;
		}
		scene.add(fpSky);

		// Height-based fog for depth
		fpFog = new THREE.FogExp2(0x8ab4cc, 0.035);
		scene.fog = fpFog;

		// Atmosphere (player light, grass, mist, dust)
		if (!atmosphere && terrainSystem) {
			atmosphere = createAtmosphere(scene, grid, terrainSystem.getHeightAt);
		}

		// Higher quality shadows for ground-level (4K for RTX class GPUs)
		if (sunLight) {
			sunLight.shadow.mapSize.width = 4096;
			sunLight.shadow.mapSize.height = 4096;
			sunLight.shadow.camera.near = 0.5;
			sunLight.shadow.camera.far = 60;
			sunLight.shadow.camera.left = -20;
			sunLight.shadow.camera.right = 20;
			sunLight.shadow.camera.top = 20;
			sunLight.shadow.camera.bottom = -20;
			sunLight.shadow.needsUpdate = true;
		}

		// Boost ambient for ground-level (shadows are darker at eye level)
		if (ambientLight) ambientLight.intensity *= 1.3;

		// Play ambient music
		if (!fpAudio) {
			fpAudio = new Audio('/song.mp3');
			fpAudio.loop = true;
			fpAudio.volume = 0.4;
		}
		fpAudio.play().catch(() => {/* autoplay blocked, user needs interaction first */});

		// Disable orbit controls
		if (controls) controls.enabled = false;
	}

	function enterFPMode() {
		enterFPMode_visual();
		// Activate postFX and sky background for walking FP mode
		if (postFX) postFX.setMode('fp');
		scene.background = null;

		// Create FP controller (interactive walking mode)
		if (!fpController) {
			fpController = createFPController(camera, renderer.domElement);
			fpController.setOnExit(() => {
				if (flythroughActive) stopFlythrough();
				exitFPMode();
			});
		}
		if (terrainSystem) {
			fpController.setHeightFn(terrainSystem.getHeightAt);
		}

		// Lock pointer
		fpController.lock();

		// Listen for pointer lock/unlock to toggle overlay reactively
		fpController.controls.addEventListener('unlock', onFPUnlock);
		fpController.controls.addEventListener('lock', onFPLock);
	}

	function exitFPMode() {
		fpMode = false;
		if (postFX) postFX.setMode('orbit');

		// Remove sky shader, restore flat background
		if (fpSky) {
			scene.remove(fpSky);
		}

		// Restore orbit camera
		camera.fov = 60;
		camera.near = 0.1;
		camera.far = 250;
		camera.rotation.order = 'XYZ'; // Restore default for orbit controls
		camera.updateProjectionMatrix();
		camera.position.set(30, 28, 30);
		camera.lookAt(0, 0, 0);

		// Remove fog
		scene.fog = null;
		fpFog = null;

		// Dispose atmosphere
		if (atmosphere) {
			atmosphere.dispose();
			atmosphere = null;
		}

		// Restore ambient intensity
		if (ambientLight) ambientLight.intensity = 0.45;

		// Scatter + creatures persist across modes

		// Stop music
		if (fpAudio) { fpAudio.pause(); fpAudio.currentTime = 0; }

		// Restore shadow quality
		if (sunLight) {
			sunLight.shadow.mapSize.width = 2048;
			sunLight.shadow.mapSize.height = 2048;
			sunLight.shadow.camera.near = 1;
			sunLight.shadow.camera.far = 100;
			sunLight.shadow.camera.left = -30;
			sunLight.shadow.camera.right = 30;
			sunLight.shadow.camera.top = 30;
			sunLight.shadow.camera.bottom = -30;
			sunLight.shadow.needsUpdate = true;
		}

		// Re-enable orbit controls
		if (controls) {
			controls.enabled = true;
			controls.target.set(0, 0, 0);
			controls.update();
		}

		// Unlock FP
		if (fpController) {
			fpController.controls.removeEventListener('unlock', onFPUnlock);
			fpController.controls.removeEventListener('lock', onFPLock);
			fpController.unlock();
		}
		pointerLocked = false;

		lastInteraction = performance.now() / 1000;
		autoRotateAngle = Math.atan2(camera.position.z, camera.position.x);

		// Force sky/lighting restoration (applyDayNight cache would skip otherwise)
		applyDayNight(timeOfDay, true);
	}

	function onFPUnlock() {
		pointerLocked = false;
	}

	function onFPLock() {
		pointerLocked = true;
	}

	function handleFPToggle() {
		if (fpMode) {
			exitFPMode();
		} else {
			enterFPMode();
		}
	}

	function handleFlythroughToggle() {
		if (flythroughActive) {
			stopFlythrough();
		} else {
			startFlythrough();
		}
	}

	function onGlobalKeyDown(e: KeyboardEvent) {
		if (e.code === 'Escape') {
			if (flythroughActive) { stopFlythrough(); exitFPMode(); }
			else if (fpMode) exitFPMode();
		}
	}

	async function startFlythrough() {
		if (!scene || !camera || !grid?.length || flythroughLoading) return;

		flythroughLoading = true;
		flythroughLoadProgress = 0;
		await nextFrame(); // let loading overlay paint

		// Step 1: Terrain
		flythroughLoadProgress = 0.1;
		if (!terrainSystem) {
			terrainSystem = createTerrain(grid);
			scene.add(terrainSystem.mesh);
		}
		await nextFrame();
		flythroughLoadProgress = 0.4;

		// Step 2: FP visual setup (sky, fog, atmosphere, shadows)
		if (!fpMode) enterFPMode_visual();
		// Unlock pointer if FP controller exists — flythrough is hands-free
		if (fpController) {
			fpController.controls.removeEventListener('unlock', onFPUnlock);
			fpController.controls.removeEventListener('lock', onFPLock);
			fpController.unlock();
		}
		pointerLocked = false;
		await nextFrame();
		flythroughLoadProgress = 0.7;

		// Step 3: Build flythrough path
		if (!flythrough) {
			flythrough = createFlythrough(grid, terrainSystem.getHeightAt);
		}
		await nextFrame();
		flythroughLoadProgress = 0.9;

		// Step 4: Start flythrough — now activate postFX and sky background
		if (flythrough) {
			flythrough.start(camera);
			scene.background = null; // Sky shader takes over
			flythroughActive = true;
			if (postFX) postFX.setMode('flythrough');
			onFlythroughChange?.(true);
		}

		flythroughLoading = false;

		// Go fullscreen
		try { container?.requestFullscreen?.(); } catch {};
	}

	function stopFlythrough() {
		if (flythrough) {
			flythrough.stop();
		}
		flythroughActive = false;
		onFlythroughChange?.(false);
		// Return to FP walking mode
		if (fpController) {
			fpController.controls.addEventListener('unlock', onFPUnlock);
			fpController.controls.addEventListener('lock', onFPLock);
		}
	}

	// --- Floating island cone with earthy underside ---
	function buildIslandSlab(cols: number, rows: number) {
		const rng = mulberry32(12345);
		const topRadius = Math.max(cols, rows) / 2 + 2;
		const bottomRadius = 1.5;
		const coneHeight = 10.0;
		const radialSegs = 48;
		const heightSegs = 20;

		// Inverted cone: wide at top (terrain level), narrows to a point below
		const geo = new THREE.CylinderGeometry(topRadius, bottomRadius, coneHeight, radialSegs, heightSegs, false);

		// Perturb vertices: heavy noise for organic mud/roots/carved stone look
		const pos = geo.attributes.position;
		for (let i = 0; i < pos.count; i++) {
			const y = pos.getY(i);
			const normalizedY = (y + coneHeight / 2) / coneHeight; // 0 at bottom, 1 at top
			const x = pos.getX(i);
			const z = pos.getZ(i);

			// Heavy noise for carved stone effect, more in the middle
			const noiseScale = Math.sin(normalizedY * Math.PI) * 1.2;
			// Vertical ridges (root-like grooves)
			const angle = Math.atan2(z, x);
			const ridgeNoise = Math.sin(angle * 8 + normalizedY * 12) * 0.3 * (1 - normalizedY);
			const jitterX = (rng() - 0.5) * noiseScale + Math.cos(angle) * ridgeNoise;
			const jitterZ = (rng() - 0.5) * noiseScale + Math.sin(angle) * ridgeNoise;
			const jitterY = (rng() - 0.5) * 0.25;

			pos.setX(i, pos.getX(i) + jitterX);
			pos.setZ(i, pos.getZ(i) + jitterZ);
			if (normalizedY < 0.93) {
				pos.setY(i, pos.getY(i) + jitterY);
			}
		}
		pos.needsUpdate = true;
		geo.computeVertexNormals();

		// Vertex colors: grass rim → wet mud → dark earth with roots → gray stone
		const colors = new Float32Array(pos.count * 3);
		const grassCol = [0.32, 0.50, 0.20];
		const mudCol   = [0.40, 0.30, 0.18]; // wet dark mud
		const rootCol  = [0.30, 0.22, 0.12]; // dark root/earth
		const stoneCol = [0.28, 0.26, 0.24]; // carved stone

		for (let i = 0; i < pos.count; i++) {
			const y = pos.getY(i);
			const t = (y + coneHeight / 2) / coneHeight;
			let r: number, g: number, b: number;
			if (t > 0.88) {
				const blend = (t - 0.88) / 0.12;
				r = mudCol[0] + (grassCol[0] - mudCol[0]) * blend;
				g = mudCol[1] + (grassCol[1] - mudCol[1]) * blend;
				b = mudCol[2] + (grassCol[2] - mudCol[2]) * blend;
			} else if (t > 0.55) {
				const blend = (t - 0.55) / 0.33;
				r = rootCol[0] + (mudCol[0] - rootCol[0]) * blend;
				g = rootCol[1] + (mudCol[1] - rootCol[1]) * blend;
				b = rootCol[2] + (mudCol[2] - rootCol[2]) * blend;
			} else if (t > 0.2) {
				const blend = (t - 0.2) / 0.35;
				r = stoneCol[0] + (rootCol[0] - stoneCol[0]) * blend;
				g = stoneCol[1] + (rootCol[1] - stoneCol[1]) * blend;
				b = stoneCol[2] + (rootCol[2] - stoneCol[2]) * blend;
			} else {
				r = stoneCol[0]; g = stoneCol[1]; b = stoneCol[2];
			}
			// More aggressive noise for mottled mud/stone look
			const noise = (rng() - 0.5) * 0.10;
			colors[i * 3] = Math.max(0, r + noise);
			colors[i * 3 + 1] = Math.max(0, g + noise);
			colors[i * 3 + 2] = Math.max(0, b + noise);
		}
		geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

		const mat = new THREE.MeshStandardMaterial({
			vertexColors: true,
			roughness: 0.9,
			metalness: 0.02,
			flatShading: true
		});

		const mesh = new THREE.Mesh(geo, mat);
		mesh.position.set(0, -coneHeight / 2 - 0.15, 0); // top flush with terrain base
		mesh.receiveShadow = true;
		mesh.castShadow = true;
		scene.add(mesh);

		// A few hanging stalactites at the bottom
		const stalMat = new THREE.MeshStandardMaterial({ color: 0x5a5248, roughness: 0.9, flatShading: true });
		for (let i = 0; i < 6; i++) {
			const angle = rng() * Math.PI * 2;
			const dist = rng() * bottomRadius * 0.6;
			const sx = Math.cos(angle) * dist;
			const sz = Math.sin(angle) * dist;
			const stalH = 0.3 + rng() * 0.8;
			const stalR = 0.08 + rng() * 0.15;
			const stalGeo = new THREE.ConeGeometry(stalR, stalH, 5);
			const stal = new THREE.Mesh(stalGeo, stalMat);
			stal.position.set(sx, -coneHeight - 0.15 - stalH / 2, sz);
			stal.rotation.x = Math.PI;
			stal.castShadow = true;
			scene.add(stal);
		}
	}

	/** Minimap Svelte action: draws grid cells and camera position */
	const MINIMAP_COLORS: Record<number, string> = {
		0: '#c9b882', 1: '#c4a862', 2: '#4a9ec9', 3: '#9a9080',
		4: '#5a8a3a', 5: '#8a8a82', 10: '#2a6888', 11: '#7ab648'
	};
	let minimapCanvas: HTMLCanvasElement | null = null;
	let minimapInterval: ReturnType<typeof setInterval> | undefined;

	function drawMinimap(canvas: HTMLCanvasElement, g: number[][]) {
		minimapCanvas = canvas;
		const draw = () => {
			if (!g?.length) return;
			const ctx = canvas.getContext('2d');
			if (!ctx) return;
			const rows = g.length, cols = g[0].length;
			const s = 3; // pixels per cell
			for (let z = 0; z < rows; z++) {
				for (let x = 0; x < cols; x++) {
					ctx.fillStyle = MINIMAP_COLORS[g[z][x]] ?? '#666';
					ctx.fillRect(x * s, z * s, s, s);
				}
			}
			// Camera dot
			if (camera) {
				const cx = (camera.position.x + cols / 2) * s;
				const cz = (camera.position.z + rows / 2) * s;
				ctx.fillStyle = '#ff4444';
				ctx.beginPath();
				ctx.arc(cx, cz, 3, 0, Math.PI * 2);
				ctx.fill();
				// Direction indicator
				const fwd = new THREE.Vector3();
				camera.getWorldDirection(fwd);
				ctx.strokeStyle = '#ff4444';
				ctx.lineWidth = 1.5;
				ctx.beginPath();
				ctx.moveTo(cx, cz);
				ctx.lineTo(cx + fwd.x * 8, cz + fwd.z * 8);
				ctx.stroke();
			}
		};
		draw();
		minimapInterval = setInterval(draw, 200); // refresh 5x/sec
		return {
			update(newGrid: number[][]) { g = newGrid; draw(); },
			destroy() { if (minimapInterval) clearInterval(minimapInterval); }
		};
	}

	/** Yield one frame so the render loop can draw while we build */
	function nextFrame(): Promise<void> {
		return new Promise(resolve => requestAnimationFrame(() => resolve()));
	}

	async function buildScene() {
		if (!grid?.length || !container) return;
		displayTime = timeOfDay;
		const gen = ++buildGeneration; // track this build — abort if superseded
		const isFirstBuild = !hemiLight; // true on very first build only
		const isFlythroughTransition = flythroughActive && worldGroup !== null;

		if (!isFlythroughTransition) {
			// Normal build: clean everything immediately
			cleanupGridSystems();

			// Remove grid-dependent scene children
			// Keep lights, celestials, clouds, weather, wildlife, fpSky, prevWorldGroup
			const keep = new Set<THREE.Object3D>();
			if (hemiLight) keep.add(hemiLight);
			if (sunLight) keep.add(sunLight);
			if (ambientLight) keep.add(ambientLight);
			if (fillLight) keep.add(fillLight);
			if (celestialSystem) { keep.add(celestialSystem.sunSprite); keep.add(celestialSystem.moonGroup); }
			if (cloudSystem) for (const s of cloudSystem.sprites) keep.add(s);
			if (wildlifeSystem) { /* birds are individual sprites — keep them */ }
			if (fpSky) keep.add(fpSky);

			for (let i = scene.children.length - 1; i >= 0; i--) {
				const child = scene.children[i];
				if (!keep.has(child)) {
					scene.remove(child);
					if (child instanceof THREE.Mesh) { child.geometry.dispose(); }
				}
			}
		} else {
			// Flythrough transition: keep old world for cross-fade, move references
			prevWorldGroup = worldGroup;
			prevTerrainSystem = terrainSystem;
			prevWaterSystem = waterSystem;
			// Null out so new build gets fresh systems
			terrainSystem = null;
			waterSystem = null;
			creatureSystem = null;
			scatter = null;
			blockyTerrainGroup = null;
			settlementLights = [];
			fireParticles = [];
			settlementPositions = [];
		}

		const rows = grid.length;
		const cols = grid[0].length;
		const offsetX = -cols / 2;
		const offsetZ = -rows / 2;

		// Create a new group to hold all world content
		const newGroup = new THREE.Group();
		// Redirect scene.add to newGroup during build
		scene.add = (...args: THREE.Object3D[]) => { newGroup.add(...args); return scene; };

		// === Lighting (first build only — added to real scene, not group) ===
		if (isFirstBuild) {
			hemiLight = new THREE.HemisphereLight(0x97c5e8, 0x8a7a5a, 0.6);
			_realSceneAdd(hemiLight);

			sunLight = new THREE.DirectionalLight(0xffeedd, 1.6);
			sunLight.position.set(20, 30, 10);
			sunLight.castShadow = true;
			sunLight.shadow.mapSize.width = 1024;
			sunLight.shadow.mapSize.height = 1024;
			sunLight.shadow.camera.near = 1;
			sunLight.shadow.camera.far = 100;
			sunLight.shadow.camera.left = -30;
			sunLight.shadow.camera.right = 30;
			sunLight.shadow.camera.top = 30;
			sunLight.shadow.camera.bottom = -30;
			_realSceneAdd(sunLight);

			ambientLight = new THREE.AmbientLight(0x909088, 0.3); // reduced — light probe provides soft fill
			_realSceneAdd(ambientLight);

			fillLight = new THREE.DirectionalLight(0x99aabb, 0.2);
			fillLight.position.set(-20, 15, -10);
			_realSceneAdd(fillLight);

			// Light probe for soft indirect lighting from environment
			lightProbe = new THREE.LightProbe();
			lightProbe.intensity = 0.6;
			_realSceneAdd(lightProbe);

			cubeRenderTarget = new THREE.WebGLCubeRenderTarget(128);
			cubeCamera = new THREE.CubeCamera(0.1, 200, cubeRenderTarget);
			lightProbeNeedsUpdate = true;

			// Temporarily restore scene.add so celestials go to real scene
			scene.add = _realSceneAdd;
			celestialSystem = createCelestials(scene);
			scene.add = (...args: THREE.Object3D[]) => { newGroup.add(...args); return scene; };
		}

		applyDayNight(timeOfDay);

		// === Square island slab with earth layers ===
		buildIslandSlab(cols, rows);

		// Find clusters
		const clusters = findClusters(grid);

		// === Multi-layered pine trees (instanced) ===
		const treeCount = countTrees(clusters);
		const treeTrunkGeo = new THREE.CylinderGeometry(0.04, 0.07, 0.5, 5);
		const treeCanopy1Geo = new THREE.ConeGeometry(0.28, 0.45, 7);
		const treeCanopy2Geo = new THREE.ConeGeometry(0.22, 0.4, 7);
		const treeCanopy3Geo = new THREE.ConeGeometry(0.15, 0.35, 7);

		const treeTrunkMat = new THREE.MeshStandardMaterial({ color: 0x5a4030, roughness: 0.9 });
		const treeCanopyMat = new THREE.MeshStandardMaterial({ color: 0x2d6b1e, roughness: 0.8 });

		const trunkInst = new THREE.InstancedMesh(treeTrunkGeo, treeTrunkMat, treeCount);
		const canopy1Inst = new THREE.InstancedMesh(treeCanopy1Geo, treeCanopyMat, treeCount);
		const canopy2Inst = new THREE.InstancedMesh(treeCanopy2Geo, treeCanopyMat, treeCount);
		const canopy3Inst = new THREE.InstancedMesh(treeCanopy3Geo, treeCanopyMat, treeCount);
		trunkInst.castShadow = true;
		canopy1Inst.castShadow = true;
		canopy2Inst.castShadow = true;
		canopy3Inst.castShadow = true;
		let treeIdx = 0;

		// Build heightmap terrain (always visible — replaces blocky tiles)
		terrainSystem = createTerrain(grid);
		scene.add(terrainSystem.mesh);

		// Water surface + caustic floor
		waterSystem = createWater(cols, rows, grid);
		scene.add(waterSystem.mesh);

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }

		// Add 3D cluster objects
		const dummy = new THREE.Object3D();

		for (const cluster of clusters) {
			switch (cluster.terrainType) {
				case TerrainCode.FOREST:
					treeIdx = addForest(cluster, offsetX, offsetZ, trunkInst, canopy1Inst, canopy2Inst, canopy3Inst, treeIdx, dummy);
					break;
				case TerrainCode.SETTLEMENT:
					addSettlement(cluster, offsetX, offsetZ, clusters.filter(c => c.terrainType === TerrainCode.SETTLEMENT));
					break;
				case TerrainCode.PORT:
					addPort(cluster, offsetX, offsetZ);
					break;
				case TerrainCode.RUIN:
					addRuin(cluster, offsetX, offsetZ);
					break;
			}
		}

		trunkInst.instanceMatrix.needsUpdate = true;
		canopy1Inst.instanceMatrix.needsUpdate = true;
		canopy2Inst.instanceMatrix.needsUpdate = true;
		canopy3Inst.instanceMatrix.needsUpdate = true;
		trunkInst.count = treeIdx;
		canopy1Inst.count = treeIdx;
		canopy2Inst.count = treeIdx;
		canopy3Inst.count = treeIdx;
		scene.add(trunkInst);
		scene.add(canopy1Inst);
		scene.add(canopy2Inst);
		scene.add(canopy3Inst);

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }

		const settlementClusters = clusters.filter(c => c.terrainType === TerrainCode.SETTLEMENT);

		// === Ambient systems (first build only — grid-independent, add to real scene) ===
		if (isFirstBuild) {
			scene.add = _realSceneAdd;
			cloudSystem = createCloudSystem(scene);
			scene.add = (...args: THREE.Object3D[]) => { newGroup.add(...args); return scene; };
		}

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }

		// === Roads between settlements ===
		buildRoads(settlementClusters, offsetX, offsetZ);

		// === Living world: creatures walking between settlements ===
		if (terrainSystem && settlementClusters.length >= 2) {
			createCreatures(scene, grid, terrainSystem.getHeightAt, settlementClusters).then(cs => {
				if (gen !== buildGeneration) { cs.dispose(); return; }
				creatureSystem = cs;
				newGroup.add(cs.group);
			}).catch(() => {});
		}

		// === Wildlife: birds (first build only — grid-independent) ===
		if (isFirstBuild) {
			scene.add = _realSceneAdd;
			wildlifeSystem = createWildlifeSystem(scene);
			scene.add = (...args: THREE.Object3D[]) => { newGroup.add(...args); return scene; };
		}

		// Restore scene.add
		scene.add = _realSceneAdd;

		// Add the new world group to the real scene
		if (isFlythroughTransition) {
			// Cross-fade: new group starts invisible, fades in
			setGroupOpacity(newGroup, 0);
			_realSceneAdd(newGroup);
			transitioning = true;
			transitionAlpha = 0;
		} else {
			_realSceneAdd(newGroup);
		}
		worldGroup = newGroup;

		// Restore FP/flythrough state if active during rebuild
		if (fpMode && fpSky) {
			scene.add(fpSky);
			scene.background = null;
		}
		if (fpMode && terrainSystem) {
			terrainSystem.mesh.visible = true;
		}
		if (flythroughActive && terrainSystem) {
			if (flythrough) {
				// Smooth transition: bridge camera path to new terrain
				flythrough.transitionToNewPath(grid, terrainSystem.getHeightAt);
			} else {
				flythrough = createFlythrough(grid, terrainSystem.getHeightAt);
				flythrough.start(camera);
			}
		}

		// Refresh light probe after scene rebuild
		lightProbeNeedsUpdate = true;
	}

	function buildRoads(settlements: Cluster[], ox: number, oz: number) {
		if (settlements.length < 2) return;
		const roadMat = new THREE.MeshStandardMaterial({ color: 0xa89060, roughness: 0.95 });

		// Connect each settlement to its nearest neighbor (minimum spanning tree approach)
		const connected = new Set<number>();
		const edges: { from: number; to: number; dist: number }[] = [];

		// Calculate all distances
		for (let i = 0; i < settlements.length; i++) {
			for (let j = i + 1; j < settlements.length; j++) {
				const dx = settlements[i].centerX - settlements[j].centerX;
				const dz = settlements[i].centerY - settlements[j].centerY;
				edges.push({ from: i, to: j, dist: Math.sqrt(dx * dx + dz * dz) });
			}
		}
		edges.sort((a, b) => a.dist - b.dist);

		// Kruskal-like: connect nearest pairs, up to N-1 edges
		const parent = settlements.map((_, i) => i);
		function find(x: number): number { return parent[x] === x ? x : (parent[x] = find(parent[x])); }
		const roadEdges: { from: number; to: number }[] = [];

		for (const e of edges) {
			const pf = find(e.from);
			const pt = find(e.to);
			if (pf !== pt) {
				parent[pf] = pt;
				roadEdges.push(e);
				if (roadEdges.length >= settlements.length - 1) break;
			}
		}

		// Draw roads as segmented strips that follow terrain height
		for (const edge of roadEdges) {
			const a = settlements[edge.from];
			const b = settlements[edge.to];
			const ax = a.centerX + ox + 0.5;
			const az = a.centerY + oz + 0.5;
			const bx = b.centerX + ox + 0.5;
			const bz = b.centerY + oz + 0.5;

			const dx = bx - ax;
			const dz = bz - az;
			const totalLen = Math.sqrt(dx * dx + dz * dz);
			const segments = Math.max(2, Math.ceil(totalLen));
			const angle = Math.atan2(dz, dx);

			for (let s = 0; s < segments; s++) {
				const t0 = s / segments;
				const t1 = (s + 1) / segments;
				const mx = ax + dx * (t0 + t1) / 2;
				const mz = az + dz * (t0 + t1) / 2;
				const segLen = totalLen / segments;
				const segY = gY(mx, mz) + 0.02;

				// Tilt segment to match terrain slope
				const y0 = gY(ax + dx * t0, az + dz * t0);
				const y1 = gY(ax + dx * t1, az + dz * t1);
				const slopeAngle = Math.atan2(y1 - y0, segLen);

				const segGeo = new THREE.BoxGeometry(segLen + 0.02, 0.015, 0.18);
				const seg = new THREE.Mesh(segGeo, roadMat);
				seg.position.set(mx, segY, mz);
				seg.rotation.y = -angle;
				seg.rotation.z = slopeAngle;
				seg.receiveShadow = true;
				scene.add(seg);
			}
		}
	}

	function countTrees(clusters: Cluster[]): number {
		let count = 0;
		for (const c of clusters) {
			if (c.terrainType !== TerrainCode.FOREST) continue;
			const treesPerCell = c.size >= 9 ? 4 : c.size >= 4 ? 3 : c.size >= 2 ? 2 : 1;
			count += c.cells.length * treesPerCell;
		}
		return Math.max(count, 1);
	}

	// --- Multi-layered pine tree (3 canopy cones stacked) ---
	function addForest(
		cluster: Cluster, ox: number, oz: number,
		trunkInst: THREE.InstancedMesh,
		canopy1: THREE.InstancedMesh, canopy2: THREE.InstancedMesh, canopy3: THREE.InstancedMesh,
		startIdx: number, dummy: THREE.Object3D
	): number {
		let idx = startIdx;
		const treesPerCell = cluster.size >= 9 ? 4 : cluster.size >= 4 ? 3 : cluster.size >= 2 ? 2 : 1;
		const baseHeight = cluster.size >= 9 ? 2.2 : cluster.size >= 4 ? 1.5 : cluster.size >= 2 ? 0.9 : 0.5;

		for (const cell of cluster.cells) {
			const rng = mulberry32(cellSeed(cell.x, cell.y, 200));
			for (let t = 0; t < treesPerCell; t++) {
				const px = cell.x + ox + 0.15 + rng() * 0.7;
				const pz = cell.y + oz + 0.15 + rng() * 0.7;
				const h = baseHeight * (0.7 + rng() * 0.3);
				const s = 0.8 + rng() * 0.4; // width variation
				const groundH = gY(px, pz);

				// Trunk
				dummy.position.set(px, groundH + h * 0.25, pz);
				dummy.scale.set(s * 0.8, h * 1.0, s * 0.8);
				dummy.rotation.set(0, 0, 0);
				dummy.updateMatrix();
				trunkInst.setMatrixAt(idx, dummy.matrix);

				// Bottom canopy layer (widest)
				dummy.position.set(px, groundH + h * 0.45, pz);
				dummy.scale.set(s * 1.3, h * 0.9, s * 1.3);
				dummy.updateMatrix();
				canopy1.setMatrixAt(idx, dummy.matrix);

				// Middle canopy layer
				dummy.position.set(px, groundH + h * 0.65, pz);
				dummy.scale.set(s * 1.1, h * 0.8, s * 1.1);
				dummy.updateMatrix();
				canopy2.setMatrixAt(idx, dummy.matrix);

				// Top canopy layer (narrowest)
				dummy.position.set(px, groundH + h * 0.82, pz);
				dummy.scale.set(s * 0.85, h * 0.7, s * 0.85);
				dummy.updateMatrix();
				canopy3.setMatrixAt(idx, dummy.matrix);

				idx++;
			}
		}
		return idx;
	}

	function addMountain(cluster: Cluster, ox: number, oz: number) {
		const rng = mulberry32(cellSeed(Math.round(cluster.centerX), Math.round(cluster.centerY), 100));
		const cx = cluster.centerX + ox + 0.5;
		const cz = cluster.centerY + oz + 0.5;

		// Try GLB models first
		if (cluster.size >= 7) {
			const m = placeModel('mountains', new THREE.Vector3(cx, 0.1, cz), rng() * Math.PI * 2, 0.55);
			if (m) { scene.add(m); return; }
		} else if (cluster.size >= 4) {
			const m = placeModel('mountainGroup', new THREE.Vector3(cx, 0.1, cz), rng() * Math.PI * 2, 0.60);
			if (m) { scene.add(m); return; }
		} else if (cluster.size >= 2) {
			const m = placeModel('mountain', new THREE.Vector3(cx, 0.1, cz), rng() * Math.PI * 2, 0.75);
			if (m) { scene.add(m); return; }
		}

		// Fallback: procedural mountain
		const h = cluster.size >= 7 ? 4.5 : cluster.size >= 4 ? 2.8 : cluster.size >= 2 ? 1.4 : 0.6;
		const r = cluster.size >= 7 ? 1.4 : cluster.size >= 4 ? 0.9 : Math.max(0.3, Math.sqrt(cluster.size) * 0.35);

		const peakGeo = new THREE.ConeGeometry(r, h, 10);
		const peakPos = peakGeo.attributes.position;
		for (let i = 0; i < peakPos.count; i++) {
			const y = peakPos.getY(i);
			if (y < h * 0.45 && y > -h * 0.45) {
				peakPos.setX(i, peakPos.getX(i) + (rng() - 0.5) * r * 0.3);
				peakPos.setZ(i, peakPos.getZ(i) + (rng() - 0.5) * r * 0.3);
			}
		}
		peakPos.needsUpdate = true;
		peakGeo.computeVertexNormals();

		const peak = new THREE.Mesh(peakGeo, new THREE.MeshStandardMaterial({ color: 0x7a7a72, roughness: 0.85, flatShading: true }));
		peak.position.set(cx, 0.3 + h / 2, cz);
		peak.castShadow = true;
		scene.add(peak);

		if (cluster.size >= 2) {
			const snowH = cluster.size >= 7 ? h * 0.3 : h * 0.22;
			const snowR = cluster.size >= 7 ? r * 0.5 : r * 0.38;
			const snow = new THREE.Mesh(
				new THREE.ConeGeometry(snowR, snowH, 8),
				new THREE.MeshStandardMaterial({ color: 0xf0ece0, roughness: 0.4 })
			);
			snow.position.set(cx, 0.3 + h * 0.85, cz);
			scene.add(snow);
		}

		// Boulders — use GLB rocks if available
		if (cluster.size >= 3) {
			for (let i = 0; i < Math.min(cluster.size, 8); i++) {
				const cell = cluster.cells[Math.min(i, cluster.cells.length - 1)];
				const bx = cell.x + ox + 0.2 + rng() * 0.6;
				const bz = cell.y + oz + 0.2 + rng() * 0.6;
				const rockModel = placeModel(rng() > 0.5 ? 'rock' : 'rocks', new THREE.Vector3(bx, 0.1, bz), rng() * Math.PI * 2, 0.20 + rng() * 0.25);
				if (rockModel) { scene.add(rockModel); }
				else {
					const bs = 0.1 + rng() * 0.25;
					const boulder = new THREE.Mesh(
						new THREE.DodecahedronGeometry(bs, 1),
						new THREE.MeshStandardMaterial({ color: 0x8a8878, roughness: 0.9, flatShading: true })
					);
					boulder.position.set(bx, 0.15 + bs * 0.5, bz);
					boulder.rotation.set(rng() * 3, rng() * 3, rng() * 3);
					boulder.castShadow = true;
					scene.add(boulder);
				}
			}
		}
	}

	function addSettlement(cluster: Cluster, ox: number, oz: number, allSettlements: Cluster[]) {
		const rng = mulberry32(cellSeed(Math.round(cluster.centerX), Math.round(cluster.centerY), 300));
		const isLarge = cluster.size >= 7;
		const isMedium = cluster.size >= 4;

		// Check if this settlement is near others → makes it a "town"
		const cx = cluster.centerX;
		const cy = cluster.centerY;
		let nearbyPop = cluster.size;
		for (const other of allSettlements) {
			if (other === cluster) continue;
			const dx = other.centerX - cx;
			const dy = other.centerY - cy;
			if (Math.sqrt(dx * dx + dy * dy) < 5) nearbyPop += other.size;
		}
		const isTown = nearbyPop >= 12 || isLarge;

		// Building pool based on settlement type
		type MN = import('./models').ModelName;
		const villageBuildings: MN[] = ['pfHut', 'hut', 'hut2', 'pfSmallFarm', 'pfCrops'];
		const townBuildings: MN[] = ['pfHouse', 'pfHouse2', 'pfHouse3', 'pfHouse4', 'pfHouse5', 'pfFarm', 'pfSmallFarm'];
		const townSpecials: MN[] = ['pfMarket', 'townCenter', 'pfFortress'];

		for (let i = 0; i < cluster.cells.length; i++) {
			const cell = cluster.cells[i];
			const px = cell.x + ox + 0.5;
			const pz = cell.y + oz + 0.5;
			const isCenter = Math.round(cluster.centerX) === cell.x && Math.round(cluster.centerY) === cell.y;
			const jx = (rng() - 0.5) * 0.25;
			const jz = (rng() - 0.5) * 0.25;
			const rot = rng() * Math.PI * 2;

			if (isTown) {
				if (isCenter && isLarge) {
					// Town center: market or fortress
					const special = townSpecials[Math.floor(rng() * townSpecials.length)];
					const m = placeModel(special, new THREE.Vector3(px, 0.1, pz), rot, 0.55);
					if (m) scene.add(m);
				} else if (isCenter && isMedium) {
					const m = placeModel('pfMarket', new THREE.Vector3(px, 0.1, pz), rot, 0.48);
					if (m) scene.add(m);
				} else {
					// Mix of houses and farms
					const pick = townBuildings[Math.floor(rng() * townBuildings.length)];
					const scale = pick.startsWith('pf') ? 0.38 + rng() * 0.10 : 0.50 + rng() * 0.12;
					const m = placeModel(pick, new THREE.Vector3(px + jx, 0.1, pz + jz), rot, scale);
					if (m) scene.add(m);
					else {
						// Procedural fallback
						const wallMat = new THREE.MeshStandardMaterial({ color: 0x8a6a40, roughness: 0.8 });
						const hutS = 0.45 + rng() * 0.15;
						const hut = new THREE.Mesh(new THREE.BoxGeometry(hutS, 0.32, hutS), wallMat);
						hut.position.set(px + jx, 0.28, pz + jz);
						hut.castShadow = true;
						scene.add(hut);
						const roof = new THREE.Mesh(
							new THREE.ConeGeometry(hutS * 0.8, 0.24, 4),
							new THREE.MeshStandardMaterial({ color: 0x7a6a52, roughness: 0.9 })
						);
						roof.position.copy(hut.position);
						roof.position.y += 0.28;
						roof.rotation.y = Math.PI / 4;
						scene.add(roof);
					}
				}
			} else {
				// Village: mostly huts, small farms, crops
				if (isCenter && isMedium) {
					// Central building for medium village
					const m = placeModel('house', new THREE.Vector3(px, 0.1, pz), rot, 0.55);
					if (m) scene.add(m);
				} else {
					const pick = villageBuildings[Math.floor(rng() * villageBuildings.length)];
					const scale = pick.startsWith('pf') ? 0.36 + rng() * 0.08 : 0.28 + rng() * 0.10;
					const m = placeModel(pick, new THREE.Vector3(px + jx, 0.1, pz + jz), rot, scale);
					if (m) scene.add(m);
					else {
						const wallMat = new THREE.MeshStandardMaterial({ color: 0x8a6a40, roughness: 0.8 });
						const hutS = 0.40 + rng() * 0.12;
						const hut = new THREE.Mesh(new THREE.BoxGeometry(hutS, 0.28, hutS), wallMat);
						hut.position.set(px + jx, 0.26, pz + jz);
						hut.castShadow = true;
						scene.add(hut);
						const roof = new THREE.Mesh(
							new THREE.ConeGeometry(hutS * 0.8, 0.16, 4),
							new THREE.MeshStandardMaterial({ color: 0x7a6a52, roughness: 0.9 })
						);
						roof.position.copy(hut.position);
						roof.position.y += 0.19;
						roof.rotation.y = Math.PI / 4;
						scene.add(roof);
					}
				}
			}

			// Fire lights disabled for performance

			if (i === 0) {
				settlementPositions.push(new THREE.Vector3(px, 0.2, pz));
			}
		}

		// Walls/fortress for towns
		if (isTown && isLarge) {
			const fcx = cluster.centerX + ox + 0.5;
			const fcz = cluster.centerY + oz + 0.5;
			const fenceR = Math.sqrt(cluster.size) * 0.55 + 0.5;
			const wallCount = 10;
			for (let i = 0; i < wallCount; i++) {
				const angle = (i / wallCount) * Math.PI * 2;
				const wx = fcx + Math.cos(angle) * fenceR;
				const wz = fcz + Math.sin(angle) * fenceR;
				const wm = placeModel('woodenWall', new THREE.Vector3(wx, 0.1, wz), angle + Math.PI / 2, 0.22);
				if (wm) scene.add(wm);
			}
		} else if (isMedium) {
			const fcx = cluster.centerX + ox + 0.5;
			const fcz = cluster.centerY + oz + 0.5;
			const fenceR = Math.sqrt(cluster.size) * 0.55 + 0.4;
			const wallCount = 6;
			for (let i = 0; i < wallCount; i++) {
				const angle = (i / wallCount) * Math.PI * 2;
				const wx = fcx + Math.cos(angle) * fenceR;
				const wz = fcz + Math.sin(angle) * fenceR;
				const wm = placeModel('woodenWall', new THREE.Vector3(wx, 0.1, wz), angle + Math.PI / 2, 0.18);
				if (wm) scene.add(wm);
			}
		}

		// Wooden walls — try GLB, fallback to procedural fence
		if (isMedium) {
			const cx = cluster.centerX + ox + 0.5;
			const cz = cluster.centerY + oz + 0.5;
			const fenceR = Math.sqrt(cluster.size) * 0.55 + 0.4;
			const wallCount = isLarge ? 8 : 5;
			for (let i = 0; i < wallCount; i++) {
				const angle = (i / wallCount) * Math.PI * 2;
				const wx = cx + Math.cos(angle) * fenceR;
				const wz = cz + Math.sin(angle) * fenceR;
				const wm = placeModel('woodenWall', new THREE.Vector3(wx, 0.1, wz), angle + Math.PI / 2, 0.20);
				if (wm) { scene.add(wm); }
				else {
					const fenceMat = new THREE.MeshStandardMaterial({ color: 0x6a5030, roughness: 0.9 });
					const postH = 0.22;
					const post = new THREE.Mesh(new THREE.BoxGeometry(0.04, postH, 0.04), fenceMat);
					post.position.set(wx, 0.12 + postH / 2, wz);
					scene.add(post);
				}
			}
		}
	}

	function addPort(cluster: Cluster, ox: number, oz: number) {
		const rng = mulberry32(cellSeed(Math.round(cluster.centerX), Math.round(cluster.centerY), 400));

		for (const cell of cluster.cells) {
			const px = cell.x + ox + 0.5;
			const pz = cell.y + oz + 0.5;
			const dm = placeModel('dock', new THREE.Vector3(px, 0.05, pz), rng() * Math.PI * 2);
			if (dm) { scene.add(dm); }
			else {
				const dockMat = new THREE.MeshStandardMaterial({ color: 0x6a5038, roughness: 0.9 });
				const dock = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.04, 0.4), dockMat);
				dock.position.set(px, 0.02, pz);
				scene.add(dock);
			}
		}

		// Port building / ship
		if (cluster.size >= 2) {
			const cx = cluster.centerX + ox + 0.5;
			const cz = cluster.centerY + oz + 0.5;
			const pm = placeModel('port', new THREE.Vector3(cx, 0.05, cz + 0.8), rng() * Math.PI * 2, 0.25);
			if (pm) { scene.add(pm); }
			else {
				// Procedural ship fallback
				const hullMat = new THREE.MeshStandardMaterial({ color: 0x5a3a20, roughness: 0.85 });
				const hull = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.12, 0.22), hullMat);
				hull.position.set(cx, -0.02, cz + 1.2);
				hull.castShadow = true;
				scene.add(hull);
				const mast = new THREE.Mesh(
					new THREE.CylinderGeometry(0.02, 0.02, 0.5, 4),
					new THREE.MeshStandardMaterial({ color: 0x6a5038 })
				);
				mast.position.set(cx, 0.25, cz + 1.2);
				scene.add(mast);
			}
		}
	}

	function addRuin(cluster: Cluster, ox: number, oz: number) {
		// Standing stones like reference image (gray monoliths)
		const stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8a80, roughness: 0.85, flatShading: true });

		for (const cell of cluster.cells) {
			const rng = mulberry32(cellSeed(cell.x, cell.y, 500));
			const count = 1 + Math.floor(rng() * 2);
			for (let i = 0; i < count; i++) {
				const h = 0.2 + rng() * 0.35;
				const w = 0.06 + rng() * 0.06;
				// Standing stone (tall thin box)
				const stoneGeo = new THREE.BoxGeometry(w, h, w * 0.6);
				const stone = new THREE.Mesh(stoneGeo, stoneMat);
				const sx = cell.x + ox + 0.2 + rng() * 0.6;
				const sz = cell.y + oz + 0.2 + rng() * 0.6;
				stone.position.set(sx, gY(sx, sz) + h / 2, sz);
				stone.rotation.y = rng() * Math.PI;
				// Slight lean
				stone.rotation.x = (rng() - 0.5) * 0.15;
				stone.rotation.z = (rng() - 0.5) * 0.15;
				stone.castShadow = true;
				scene.add(stone);
			}

			// Rubble stones
			const rubbleMat = new THREE.MeshStandardMaterial({ color: 0x9a9a90, roughness: 0.9, flatShading: true });
			for (let i = 0; i < 2; i++) {
				const rs = 0.04 + rng() * 0.06;
				const rubbleGeo = new THREE.DodecahedronGeometry(rs, 0);
				const rubble = new THREE.Mesh(rubbleGeo, rubbleMat);
				rubble.position.set(
					cell.x + ox + 0.2 + rng() * 0.6,
					0.1 + rs,
					cell.y + oz + 0.2 + rng() * 0.6
				);
				rubble.rotation.set(rng() * 3, rng() * 3, rng() * 3);
				scene.add(rubble);
			}
		}
	}


	function applyDayNight(hour: number, force = false) {
		// Skip if time hasn't changed (within 0.01 resolution)
		const rounded = Math.round(hour * 100);
		if (!force && rounded === prevTimeOfDay && cachedDN) return cachedDN;
		prevTimeOfDay = rounded;

		const dn = computeDayNight(hour);
		cachedDN = dn;

		if (sunLight) {
			sunLight.position.copy(dn.sunPosition);
			sunLight.color.copy(dn.sunColor);
			sunLight.intensity = dn.sunIntensity;

			// Throttle shadow re-renders — orbit=5s, FP=2s
			const now = performance.now();
			const shadowInterval = fpMode ? 2000 : 5000;
			if (force || now - lastShadowUpdate > shadowInterval) {
				sunLight.shadow.needsUpdate = true;
				lastShadowUpdate = now;
				lightProbeNeedsUpdate = true;
			}
		}
		if (ambientLight) {
			ambientLight.color.copy(dn.ambientColor);
			ambientLight.intensity = dn.ambientIntensity;
		}
		if (hemiLight) {
			hemiLight.color.copy(dn.hemiSkyColor);
			hemiLight.groundColor.copy(dn.hemiGroundColor);
			hemiLight.intensity = dn.hemiIntensity;
		}
		// Only set flat background when Preetham sky is NOT active
		if (scene && !fpSky?.parent) {
			scene.background = dn.skyColor;
		}
		if (postFX) postFX.updateAtmosphere(hour, flythroughActive ? 'flythrough' : fpMode ? 'fp' : 'orbit');
		return dn;
	}

	function animate() {
		animationId = requestAnimationFrame(animate);
		const now = performance.now() / 1000;
		const dt = lastTime ? Math.min(now - lastTime, 0.1) : 0.016;
		lastTime = now;

		// (ocean rendered as transparent blocks, no separate plane)

		// Smooth time transitions — lerp toward target to avoid frame drops
		const timeDiff = timeOfDay - displayTime;
		if (Math.abs(timeDiff) > 0.005) {
			displayTime += timeDiff * Math.min(1, dt * 5);
		} else {
			displayTime = timeOfDay;
		}
		const dn = applyDayNight(displayTime);

		// Update celestials every frame (world-space, large orbit radius)
		if (celestialSystem && cachedDN) {
			celestialSystem.update(cachedDN.sunPosition, cachedDN.moonPosition, cachedDN.nightFade);
		}
		// Update light probe for soft indirect lighting (throttled with shadows)
		if (lightProbeNeedsUpdate && cubeCamera && cubeRenderTarget && lightProbe && renderer) {
			lightProbeNeedsUpdate = false;
			cubeCamera.position.set(0, 2, 0);
			cubeCamera.update(renderer, scene);
			const probeResult = LightProbeGenerator.fromCubeRenderTarget(renderer, cubeRenderTarget);
			if (probeResult instanceof Promise) {
				probeResult.then(p => { if (lightProbe) { lightProbe.copy(p); lightProbe.intensity = 0.6; } });
			} else {
				lightProbe.copy(probeResult as THREE.LightProbe);
				lightProbe.intensity = 0.6;
			}
		}

		if (cloudSystem) cloudSystem.update(dt, dn.skyColor);
		// waterfallSystem removed for performance
		if (prevWaterSystem) prevWaterSystem.update(dt, cachedDN?.sunPosition); // keep old water animating during cross-fade
		if (waterSystem) {
			waterSystem.update(dt, cachedDN?.sunPosition);
			// Player ripples when walking in water (FP mode)
			if (fpMode && camera.position.y < 0.15) {
				const speed = Math.sqrt(
					(camera.position.x - (lastCamX ?? camera.position.x)) ** 2
					+ (camera.position.z - (lastCamZ ?? camera.position.z)) ** 2
				);
				if (speed > 0.005) {
					rippleCooldown -= dt;
					if (rippleCooldown <= 0) {
						waterSystem.addRipple(camera.position.x, camera.position.z);
						rippleCooldown = 0.25;
					}
				}
			}
			lastCamX = camera.position.x;
			lastCamZ = camera.position.z;
		}
		if (wildlifeSystem) wildlifeSystem.update(dt, displayTime);
		if (creatureSystem) creatureSystem.update(dt, displayTime, camera);

		// Cross-fade transition: animate opacity between old and new world
		if (transitioning) {
			transitionAlpha = Math.min(1.0, transitionAlpha + dt / TRANSITION_DURATION);
			setGroupOpacity(worldGroup, transitionAlpha);
			if (prevWorldGroup) setGroupOpacity(prevWorldGroup, 1.0 - transitionAlpha);
			if (transitionAlpha >= 1.0) {
				transitioning = false;
				if (prevWorldGroup) {
					disposeGroup(prevWorldGroup);
					scene.remove(prevWorldGroup);
					prevWorldGroup = null;
				}
				if (prevTerrainSystem) { prevTerrainSystem.dispose(); prevTerrainSystem = null; }
				if (prevWaterSystem) { prevWaterSystem.dispose(); prevWaterSystem = null; }
				// Restore full opacity on new world
				setGroupOpacity(worldGroup, 1.0);
			}
		}

		// Settlement fires disabled for performance

		// First-person mode: update FP controller or flythrough

		if (fpMode && fpController) {
			// Flythrough overrides FP controls
			if (flythroughActive && flythrough) {
				flythrough.update(camera, dt);
			} else {
				fpController.update(dt);
			}

			// Sky dome follows camera so it always surrounds the viewer
			if (fpSky) fpSky.position.copy(camera.position);

			// Sync Preetham sky with sun position from daynight system
			if (fpSky && fpSkyUniforms && cachedDN) {
				// Use the daynight sun position as a direction for the sky shader
				// sunPosition uniform expects a unit direction vector
				fpSkyUniforms['sunPosition'].value
					.copy(cachedDN.sunPosition)
					.normalize();

				// Adjust sky parameters based on time of day
				const h = ((timeOfDay % 24) + 24) % 24;
				const isDawnDusk = (h >= 5 && h <= 7) || (h >= 17 && h <= 19);
				const isNight = h < 5 || h > 19;
				if (isDawnDusk) {
					fpSkyUniforms['turbidity'].value = 8;
					fpSkyUniforms['rayleigh'].value = 3;
					fpSkyUniforms['mieCoefficient'].value = 0.01;
				} else if (isNight) {
					fpSkyUniforms['turbidity'].value = 2;
					fpSkyUniforms['rayleigh'].value = 0.5;
					fpSkyUniforms['mieCoefficient'].value = 0.001;
				} else {
					fpSkyUniforms['turbidity'].value = 4;
					fpSkyUniforms['rayleigh'].value = 2;
					fpSkyUniforms['mieCoefficient'].value = 0.005;
				}
			}

			// Animate fog — color derived from sky, density from camera height
			if (fpFog) {
				fpFog.color.copy(cachedDN!.fogColor);
				const camHeight = camera.position.y;
				fpFog.density = 0.035 + Math.max(0, (0.3 - camHeight) * 0.025);
			}

			// Update atmosphere (player light, grass, mist, dust)
			if (atmosphere) {
				atmosphere.update(camera, dt, timeOfDay);
			}

		} else {
			// Gentle idle auto-rotation after 10 seconds of no interaction
			const idleTime = now - lastInteraction;
			if (!freezeCamera && idleTime > 10 && controls) {
				autoRotateAngle += dt * 0.08; // slow orbit
				const dist = camera.position.length();
				const baseY = camera.position.y;
				camera.position.x = Math.cos(autoRotateAngle) * dist * 0.7;
				camera.position.z = Math.sin(autoRotateAngle) * dist * 0.7;
				camera.position.y = baseY + Math.sin(autoRotateAngle * 0.3) * 0.5;
				camera.lookAt(0, 0, 0);
			}

			controls.update();
		}

		// Post-processing: bloom, DOF, vignette, ACES, SMAA — always active
		if (postFX) {
			postFX.render(dt);
		} else {
			renderer.render(scene, camera);
		}

		// FPS + draw call counter (updated every 30 frames)
		fpsFrameCount++;
		if (fpsFrameCount >= 30) {
			const elapsed = now - fpsLastTime;
			if (elapsed > 0) {
				fpsDisplay = `${Math.round(fpsFrameCount / elapsed)} FPS`;
				drawCallsDisplay = `${renderer.info.render.calls} draws · ${(renderer.info.render.triangles / 1000).toFixed(0)}k tris`;
			}
			fpsFrameCount = 0;
			fpsLastTime = now;
		}
	}

	function handleResize() {
		if (!container || !camera || !renderer) return;
		camera.aspect = container.clientWidth / container.clientHeight;
		camera.updateProjectionMatrix();
		renderer.setSize(container.clientWidth, container.clientHeight);
		if (postFX) postFX.resize(container.clientWidth, container.clientHeight);
	}

	onMount(() => {
		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
		renderer.setPixelRatio(window.devicePixelRatio);
		renderer.setSize(container.clientWidth, container.clientHeight);
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFSoftShadowMap;
		renderer.shadowMap.autoUpdate = false; // manual shadow updates on time change only
		renderer.toneMapping = THREE.NoToneMapping; // postFX handles tone mapping
		container.appendChild(renderer.domElement);

		scene = new THREE.Scene();
		_realSceneAdd = scene.add.bind(scene);
		scene.background = new THREE.Color(0x87ceeb);

		camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 250);
		camera.position.set(30, 28, 30);
		camera.lookAt(0, 0, 0);

		// Post-processing pipeline (bloom, DOF, vignette, ACES, SMAA) — always active
		postFX = createPostFX(renderer, scene, camera);
		postFX.resize(container.clientWidth, container.clientHeight);

		controls = new OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.05;
		controls.maxPolarAngle = Math.PI * 0.45;
		controls.minDistance = 10;
		controls.maxDistance = 90;

		// Track user interaction to pause auto-rotation
		lastInteraction = performance.now() / 1000;
		const resetIdle = () => {
			lastInteraction = performance.now() / 1000;
			// Sync autoRotateAngle to current camera position so resume is seamless
			autoRotateAngle = Math.atan2(camera.position.z, camera.position.x);
		};
		renderer.domElement.addEventListener('pointerdown', resetIdle);
		renderer.domElement.addEventListener('pointermove', resetIdle);
		renderer.domElement.addEventListener('wheel', resetIdle);

		// Preload GLB models then build scene
		preloadModels().then(() => {
			buildScene().then(() => applyDayNight(timeOfDay, true));
		}).catch(() => {
			buildScene().then(() => applyDayNight(timeOfDay, true));
		});
		animate();
		window.addEventListener('resize', handleResize);
		window.addEventListener('keydown', onGlobalKeyDown);

		// Register capture function for parent component
		if (onCaptureFn) {
			onCaptureFn(() => {
				renderer.render(scene, camera);
				return renderer.domElement.toDataURL('image/png');
			});
		}
	});

	onDestroy(() => {
		if (typeof window === 'undefined') return;
		if (animationId) cancelAnimationFrame(animationId);
		// Full cleanup on destroy
		cleanupGridSystems();
		if (prevWorldGroup) { disposeGroup(prevWorldGroup); prevWorldGroup = null; }
		if (worldGroup) { disposeGroup(worldGroup); worldGroup = null; }
		if (prevTerrainSystem) { prevTerrainSystem.dispose(); prevTerrainSystem = null; }
		if (prevWaterSystem) { prevWaterSystem.dispose(); prevWaterSystem = null; }
		if (cloudSystem) { cloudSystem.dispose(); cloudSystem = null; }
		if (waterfallSystem) { waterfallSystem.dispose(); waterfallSystem = null; }
		if (celestialSystem) { celestialSystem.dispose(); celestialSystem = null; }
		if (wildlifeSystem) { wildlifeSystem.dispose(); wildlifeSystem = null; }
		if (weatherSystem) { weatherSystem.dispose(); weatherSystem = null; }
		if (fillLight) { scene.remove(fillLight); fillLight = null; }
		if (lightProbe) { scene.remove(lightProbe); lightProbe = null; }
		if (cubeRenderTarget) { cubeRenderTarget.dispose(); cubeRenderTarget = null; }
		cubeCamera = null;
		if (fpController) { fpController.dispose(); fpController = null; }
		if (postFX) { postFX.dispose(); postFX = null; }
		if (atmosphere) { atmosphere.dispose(); atmosphere = null; }
		if (fpSky) { fpSky.geometry.dispose(); (fpSky.material as THREE.Material).dispose(); fpSky = null; }
		if (scatter) { scatter.dispose(); scatter = null; }
		if (creatureSystem) { creatureSystem.dispose(); creatureSystem = null; }
		if (flythrough) { flythrough.dispose(); flythrough = null; }
		if (renderer) renderer.dispose();
		window.removeEventListener('resize', handleResize);
		window.removeEventListener('keydown', onGlobalKeyDown);
	});

	$effect(() => { grid; settlements; if (scene) buildScene(); });
</script>

<div class="relative w-full h-full">
	<div bind:this={container} class="w-full h-full rounded-lg overflow-hidden border border-cyber-border"></div>

	<!-- Performance stats overlay (hidden during flythrough) -->
	{#if fpsDisplay && !flythroughActive}
		<div class="absolute bottom-2 left-2 z-10 px-2 py-1 text-[10px] font-mono text-cyber-muted bg-cyber-bg/70 rounded backdrop-blur-sm pointer-events-none">
			{fpsDisplay} · {drawCallsDisplay}
		</div>
	{/if}

	<!-- FP mode buttons (hidden during flythrough) -->
	{#if !flythroughActive}
		<div class="absolute top-3 right-3 z-10 flex gap-2">
			<button
				class="px-3 py-1.5 text-[11px] font-medium rounded border transition-colors backdrop-blur-sm
					border-cyber-border text-cyber-muted bg-cyber-surface/60 hover:border-neon-gold/40 hover:text-cyber-fg"
				onclick={handleFlythroughToggle}
				title="Epic FPV flythrough"
			>
				Flythrough
			</button>
			<button
				class="px-3 py-1.5 text-[11px] font-medium rounded border transition-colors backdrop-blur-sm
					{fpMode
						? 'border-neon-cyan text-neon-cyan bg-cyber-surface/80 hover:bg-neon-cyan/20'
						: 'border-cyber-border text-cyber-muted bg-cyber-surface/60 hover:border-neon-cyan/40 hover:text-cyber-fg'
					}"
				onclick={handleFPToggle}
				title={fpMode ? 'Exit first-person' : 'Enter first-person mode'}
			>
				{fpMode ? 'Exit FP' : 'First Person'}
			</button>
		</div>
	{/if}

	<!-- Flythrough loading overlay -->
	{#if flythroughLoading}
		<div class="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md rounded-lg">
			<h2 class="text-2xl font-thin tracking-widest uppercase text-white/80 mb-6">
				Preparing Flythrough
			</h2>
			<div class="w-64 h-1 bg-white/10 rounded-full overflow-hidden">
				<div class="h-full bg-cyan-400/80 rounded-full transition-all duration-300"
					style="width: {flythroughLoadProgress * 100}%"></div>
			</div>
			<p class="mt-3 text-xs text-white/40">
				{#if flythroughLoadProgress < 0.4}Building terrain...
				{:else if flythroughLoadProgress < 0.7}Setting up atmosphere...
				{:else}Charting flight path...{/if}
			</p>
		</div>
	{/if}

	<!-- FP mode HUD overlay -->
	{#if fpMode}
		<!-- Crosshair -->
		<div class="absolute inset-0 pointer-events-none flex items-center justify-center z-10">
			<div class="w-5 h-5 relative opacity-40">
				<div class="absolute top-1/2 left-0 w-full h-px bg-white -translate-y-px"></div>
				<div class="absolute left-1/2 top-0 h-full w-px bg-white -translate-x-px"></div>
			</div>
		</div>

		<!-- Controls hint (bottom) — not during flythrough -->
		{#if !pointerLocked && !flythroughActive}
			<div
				bind:this={fpOverlay}
				class="absolute inset-0 z-20 flex items-center justify-center bg-black/50 backdrop-blur-sm cursor-pointer rounded-lg"
				onclick={() => fpController?.lock()}
				role="button"
				tabindex="0"
				onkeydown={(e) => e.key === 'Enter' && fpController?.lock()}
			>
				<div class="text-center text-cyber-fg">
					<p class="text-lg font-medium mb-2">Click to enter first-person view</p>
					<p class="text-xs text-cyber-muted">WASD to move &middot; Mouse to look &middot; Shift to sprint &middot; Esc to pause</p>
				</div>
			</div>
		{/if}

		<!-- FP mode indicator (walking only) -->
		{#if !flythroughActive}
			<div class="absolute bottom-3 left-3 z-10 text-[10px] text-cyber-muted/60 pointer-events-none">
				FP MODE &middot; WASD + Mouse &middot; Shift = Sprint
			</div>
		{/if}

		<!-- Flythrough cinematic HUD -->
		{#if flythroughActive}
			<div class="absolute inset-0 pointer-events-none z-20 flex flex-col items-center"
				style="animation: hudFadeIn 2s ease-out forwards">
				<div class="mt-10 text-center">
					<h1 class="text-5xl font-thin tracking-[0.4em] uppercase"
						style="color: rgba(255,255,255,0.85);
							text-shadow: 0 0 40px rgba(255,255,255,0.4), 0 0 80px rgba(135,206,235,0.25), 0 0 120px rgba(135,206,235,0.1);
							filter: blur(0.3px)">
						ASTAR ISLAND
					</h1>
					{#if roundLabel || seedLabel}
						<p class="mt-3 text-base tracking-[0.25em] uppercase"
							style="color: rgba(255,255,255,0.55);
								text-shadow: 0 0 20px rgba(255,255,255,0.3), 0 0 40px rgba(135,206,235,0.15);
								transition: opacity 0.8s ease">
							{#if roundLabel}{roundLabel}{/if}
							{#if roundLabel && seedLabel} &middot; {/if}
							{#if seedLabel}{seedLabel}{/if}
						</p>
					{/if}
				</div>
			</div>

			<!-- Minimap + terrain breakdown (bottom-right) -->
			{#if grid}
				<div class="absolute bottom-4 right-4 z-20 pointer-events-none flex gap-3 items-end" style="animation: hudFadeIn 2s ease-out forwards">
					<!-- Terrain breakdown -->
					<div class="text-[9px] font-mono text-white/50 leading-relaxed text-right">
						{@const counts = (() => {
							const c: Record<number, number> = {};
							let total = 0;
							for (const row of grid) for (const cell of row) { c[cell] = (c[cell] || 0) + 1; total++; }
							return { c, total };
						})()}
						{#each Object.entries(counts.c).sort((a, b) => Number(b[1]) - Number(a[1])) as [code, count]}
							{@const pct = ((count as number) / counts.total * 100).toFixed(0)}
							{@const names: Record<string, string> = {'0':'Sand','1':'Town','2':'Port','3':'Ruin','4':'Forest','5':'Mountain','10':'Ocean','11':'Plains'}}
							{#if (count as number) / counts.total > 0.02}
								<div>{names[code] ?? '?'} {pct}%</div>
							{/if}
						{/each}
					</div>
					<!-- Minimap canvas -->
					<canvas
						class="rounded border border-white/20"
						width={grid[0].length * 3}
						height={grid.length * 3}
						style="width: {Math.min(120, grid[0].length * 3)}px; height: {Math.min(120, grid.length * 3)}px; image-rendering: pixelated;"
						use:drawMinimap={grid}
					></canvas>
				</div>
			{/if}

			<!-- Exit hint (click anywhere) -->
			<div class="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 text-[10px] text-white/30 pointer-events-none">
				Press ESC to exit
			</div>
		{/if}
	{/if}
</div>

<style>
	@keyframes hudFadeIn {
		from { opacity: 0; transform: translateY(-10px); }
		to { opacity: 1; transform: translateY(0); }
	}
</style>
