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
	let settlementLights: THREE.PointLight[] = [];
	let fireParticles: THREE.Mesh[] = [];
	let settlementPositions: THREE.Vector3[] = [];
	let lastInteraction = 0;
	let autoRotateAngle = 0;
	let buildGeneration = 0; // incremented each build to cancel stale async work
	let prevTimeOfDay = -1;
	let cachedDN: ReturnType<typeof computeDayNight> | null = null;
	let lastShadowUpdate = 0;
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
	let scatter: ScatterSystem | null = null;
	let creatureSystem: CreatureSystem | null = null;
	let fpAudio: HTMLAudioElement | null = null;
	let pointerLocked = $state(false);

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

	function cleanupSystems() {
		if (cloudSystem) { cloudSystem.dispose(); cloudSystem = null; }
		if (waterfallSystem) { waterfallSystem.dispose(); waterfallSystem = null; }
		if (celestialSystem) { celestialSystem.dispose(); celestialSystem = null; }
		if (wildlifeSystem) { wildlifeSystem.dispose(); wildlifeSystem = null; }
		if (weatherSystem) { weatherSystem.dispose(); weatherSystem = null; }
		if (terrainSystem) { terrainSystem.dispose(); terrainSystem = null; }
		if (creatureSystem) { creatureSystem.dispose(); creatureSystem = null; }
		if (scatter) { scatter.dispose(); scatter = null; }
		if (fillLight) { scene.remove(fillLight); fillLight = null; }
		blockyTerrainGroup = null;
		settlementLights = [];
		fireParticles = [];
		settlementPositions = [];
	}

	// === First-person mode toggle ===
	function enterFPMode() {
		if (!scene || !camera || !renderer || !grid?.length) return;
		fpMode = true;

		// Build heightmap terrain if needed
		if (!terrainSystem) {
			terrainSystem = createTerrain(grid);
			scene.add(terrainSystem.mesh);
		}
		terrainSystem.mesh.visible = true;

		// Hide blocky terrain
		if (blockyTerrainGroup) blockyTerrainGroup.visible = false;

		// Create FP controller
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

		// Camera: wide FOV, close near plane for immersion
		camera.fov = 100;
		camera.near = 0.01;
		camera.far = 300;
		camera.rotation.order = 'YXZ'; // Match PointerLockControls' Euler order
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
		// Hide the flat color background — sky shader renders it
		scene.background = null;

		// Height-based fog for depth
		fpFog = new THREE.FogExp2(0x8ab4cc, 0.035);
		scene.fog = fpFog;

		// Atmosphere (player light, grass, mist, dust)
		if (!atmosphere && terrainSystem) {
			atmosphere = createAtmosphere(scene, grid, terrainSystem.getHeightAt);
		}

		// Higher quality shadows for ground-level (2K — good balance of quality vs perf)
		if (sunLight) {
			sunLight.shadow.mapSize.width = 2048;
			sunLight.shadow.mapSize.height = 2048;
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

		// Lock pointer
		fpController.lock();

		// Listen for pointer lock/unlock to toggle overlay reactively
		fpController.controls.addEventListener('unlock', onFPUnlock);
		fpController.controls.addEventListener('lock', onFPLock);
	}

	function exitFPMode() {
		fpMode = false;

		// Show blocky terrain, hide heightmap
		if (blockyTerrainGroup) blockyTerrainGroup.visible = true;
		if (terrainSystem) terrainSystem.mesh.visible = false;

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
		camera.position.set(28, 22, 28);
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

	function startFlythrough() {
		if (!scene || !camera || !grid?.length) return;

		// Enter FP visual mode first (sky, postfx, terrain)
		if (!fpMode) enterFPMode();
		// Unlock pointer — flythrough is hands-free
		if (fpController) {
			fpController.controls.removeEventListener('unlock', onFPUnlock);
			fpController.controls.removeEventListener('lock', onFPLock);
			fpController.unlock();
		}
		pointerLocked = false;

		// Build flythrough path if needed
		if (!flythrough && terrainSystem) {
			flythrough = createFlythrough(grid, terrainSystem.getHeightAt);
		}

		if (flythrough) {
			flythrough.start(camera);
			flythroughActive = true;
			onFlythroughChange?.(true);
		}
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

	// --- Square floating island slab with earthy cross-section ---
	function buildIslandSlab(cols: number, rows: number) {
		const rng = mulberry32(12345);
		const w = cols + 1.5;
		const d = rows + 1.5;
		const hw = w / 2;
		const hd = d / 2;

		// Earth cross-section layers — grass rim sits below water level
		// so ocean cells show water on top, land terrain blocks poke above
		const layers = [
			{ yTop: -0.08, yBot: -0.20, color: 0x5a8a3a, roughness: 0.85 },  // Grass rim (below water surface at -0.05)
			{ yTop: -0.20, yBot: -0.8, color: 0x9a7840, roughness: 0.9 },    // Brown earth
			{ yTop: -0.8, yBot: -1.8, color: 0x7a6545, roughness: 0.92 },     // Deep dirt
			{ yTop: -1.8, yBot: -3.2, color: 0x5a5248, roughness: 0.95 },     // Rock
		];

		for (let li = 0; li < layers.length; li++) {
			const layer = layers[li];
			// Each lower layer slightly inset for natural taper
			const inset = li * 0.15;
			const lw = w - inset * 2;
			const ld = d - inset * 2;
			const lh = layer.yTop - layer.yBot;

			// Use subdivided box for rough side texture
			const segsW = Math.max(1, Math.round(lw * 5));
			const segsD = Math.max(1, Math.round(ld * 5));
			const segsH = Math.max(2, Math.round(lh * 8));
			const geo = new THREE.BoxGeometry(lw, lh, ld, segsW, segsH, segsD);

			// Perturb side vertices (not top/bottom faces) for organic roughness
			const pos = geo.attributes.position;
			const normal = geo.attributes.normal;
			for (let i = 0; i < pos.count; i++) {
				const nx = normal.getX(i);
				const ny = normal.getY(i);
				const nz = normal.getZ(i);
				// Only perturb side faces (normal.y ≈ 0)
				if (Math.abs(ny) < 0.5) {
					const jitter = (rng() - 0.5) * 0.30 * (1 + li * 0.35);
					pos.setX(i, pos.getX(i) + nx * jitter);
					pos.setZ(i, pos.getZ(i) + nz * jitter);
					// Slight vertical noise too
					pos.setY(i, pos.getY(i) + (rng() - 0.5) * 0.08);
				}
			}
			pos.needsUpdate = true;
			geo.computeVertexNormals();

			const mat = new THREE.MeshStandardMaterial({
				color: layer.color,
				roughness: layer.roughness,
				metalness: 0.02,
				flatShading: true
			});
			const mesh = new THREE.Mesh(geo, mat);
			mesh.position.set(0, (layer.yTop + layer.yBot) / 2, 0);
			mesh.receiveShadow = true;
			mesh.castShadow = true;
			scene.add(mesh);
		}

		// Hanging stalactites under the island
		const stalMat = new THREE.MeshStandardMaterial({ color: 0x5a5248, roughness: 0.9, flatShading: true });
		for (let i = 0; i < 12; i++) {
			const sx = (rng() - 0.5) * w * 0.8;
			const sz = (rng() - 0.5) * d * 0.8;
			const stalH = 0.5 + rng() * 1.5;
			const stalR = 0.15 + rng() * 0.35;
			const stalGeo = new THREE.ConeGeometry(stalR, stalH, 5);
			const stal = new THREE.Mesh(stalGeo, stalMat);
			stal.position.set(sx, -3.2 - stalH / 2, sz);
			stal.rotation.x = Math.PI;
			stal.castShadow = true;
			scene.add(stal);
		}

		// Small rocks embedded in the sides
		const rockMat = new THREE.MeshStandardMaterial({ color: 0x6a6055, roughness: 0.95, flatShading: true });
		for (let i = 0; i < 20; i++) {
			const side = Math.floor(rng() * 4);
			let rx: number, rz: number;
			if (side === 0) { rx = -hw + (rng() - 0.5) * 0.3; rz = (rng() - 0.5) * d; }
			else if (side === 1) { rx = hw + (rng() - 0.5) * 0.3; rz = (rng() - 0.5) * d; }
			else if (side === 2) { rx = (rng() - 0.5) * w; rz = -hd + (rng() - 0.5) * 0.3; }
			else { rx = (rng() - 0.5) * w; rz = hd + (rng() - 0.5) * 0.3; }
			const ry = -0.5 - rng() * 2.0;
			const rs = 0.1 + rng() * 0.25;
			const rockGeo = new THREE.DodecahedronGeometry(rs, 0);
			const rock = new THREE.Mesh(rockGeo, rockMat);
			rock.position.set(rx, ry, rz);
			rock.rotation.set(rng() * 3, rng() * 3, rng() * 3);
			scene.add(rock);
		}
	}

	/** Yield one frame so the render loop can draw while we build */
	function nextFrame(): Promise<void> {
		return new Promise(resolve => requestAnimationFrame(() => resolve()));
	}

	async function buildScene() {
		if (!grid?.length || !container) return;
		displayTime = timeOfDay;
		const gen = ++buildGeneration; // track this build — abort if superseded

		cleanupSystems();
		while (scene.children.length > 0) {
			const child = scene.children[0];
			scene.remove(child);
			if (child instanceof THREE.Mesh) { child.geometry.dispose(); }
		}

		const rows = grid.length;
		const cols = grid[0].length;
		const offsetX = -cols / 2;
		const offsetZ = -rows / 2;

		// === Lighting — warm & bright like reference ===
		hemiLight = new THREE.HemisphereLight(0x97c5e8, 0x8a7a5a, 0.6);
		scene.add(hemiLight);

		sunLight = new THREE.DirectionalLight(0xffeedd, 1.6);
		sunLight.position.set(20, 30, 10);
		sunLight.castShadow = true;
		sunLight.shadow.mapSize.width = 2048;
		sunLight.shadow.mapSize.height = 2048;
		sunLight.shadow.camera.near = 1;
		sunLight.shadow.camera.far = 100;
		sunLight.shadow.camera.left = -30;
		sunLight.shadow.camera.right = 30;
		sunLight.shadow.camera.top = 30;
		sunLight.shadow.camera.bottom = -30;
		scene.add(sunLight);

		ambientLight = new THREE.AmbientLight(0x909088, 0.45);
		scene.add(ambientLight);

		// Secondary fill light — sky bounce from opposite side
		fillLight = new THREE.DirectionalLight(0x99aabb, 0.2);
		fillLight.position.set(-20, 15, -10);
		scene.add(fillLight);

		applyDayNight(timeOfDay);

		// === Square island slab with earth layers ===
		buildIslandSlab(cols, rows);

		// === Celestials ===
		celestialSystem = createCelestials(scene);

		// Find clusters
		const clusters = findClusters(grid);

		// === Multi-layered pine trees (instanced) ===
		// Each tree = trunk + 3 canopy cone layers
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

		// Build terrain blocks (wrapped in group for FP mode toggling)
		blockyTerrainGroup = new THREE.Group();
		const blockGeo = new THREE.BoxGeometry(0.95, 0.15, 0.95);
		const typeCounts: Record<number, number> = {};
		for (const row of grid) {
			for (const cell of row) {
				typeCounts[cell] = (typeCounts[cell] || 0) + 1;
			}
		}

		// Ocean material — transparent blue blocks
		const oceanMat = new THREE.MeshStandardMaterial({
			color: 0x3088b8,
			transparent: true,
			opacity: 0.65,
			roughness: 0.2,
			metalness: 0.15
		});

		for (const [typeStr, count] of Object.entries(typeCounts)) {
			const type = Number(typeStr);
			const isOcean = type === TerrainCode.OCEAN;
			const mat = isOcean ? oceanMat : getTerrainMaterial(type);
			const instanced = new THREE.InstancedMesh(blockGeo, mat, count);
			instanced.receiveShadow = true;
			let idx = 0;
			const dummyT = new THREE.Object3D();
			for (let y = 0; y < rows; y++) {
				for (let x = 0; x < cols; x++) {
					if (grid[y][x] !== type) continue;
					const h = getTerrainHeight(type);
					dummyT.position.set(x + offsetX + 0.5, h / 2, y + offsetZ + 0.5);
					dummyT.scale.set(1, h / 0.15, 1);
					dummyT.updateMatrix();
					instanced.setMatrixAt(idx++, dummyT.matrix);
				}
			}
			instanced.instanceMatrix.needsUpdate = true;
			blockyTerrainGroup.add(instanced);
		}
		scene.add(blockyTerrainGroup);

		await nextFrame();
		if (gen !== buildGeneration) return; // superseded by newer build

		// Build heightmap terrain (hidden initially, used in FP mode)
		terrainSystem = createTerrain(grid);
		terrainSystem.mesh.visible = false;
		scene.add(terrainSystem.mesh);

		// Ground under ocean cells — earth blocks so water doesn't float over void
		const oceanCellCount = typeCounts[TerrainCode.OCEAN] || 0;
		if (oceanCellCount > 0) {
			const groundUnderWaterGeo = new THREE.BoxGeometry(0.95, 0.20, 0.95);
			const groundUnderWaterMat = new THREE.MeshStandardMaterial({
				color: 0x6a5a3a, // brown earth, visible through translucent water
				roughness: 0.90,
				metalness: 0.02,
			});
			const groundInst = new THREE.InstancedMesh(groundUnderWaterGeo, groundUnderWaterMat, oceanCellCount);
			groundInst.receiveShadow = true;
			let gIdx = 0;
			const dummyG = new THREE.Object3D();
			for (let y = 0; y < rows; y++) {
				for (let x = 0; x < cols; x++) {
					if (grid[y][x] !== TerrainCode.OCEAN) continue;
					// Earth block sitting just under the seabed block
					dummyG.position.set(x + offsetX + 0.5, -0.33, y + offsetZ + 0.5);
					dummyG.scale.set(1, 1, 1);
					dummyG.updateMatrix();
					groundInst.setMatrixAt(gIdx++, dummyG.matrix);
				}
			}
			groundInst.instanceMatrix.needsUpdate = true;
			blockyTerrainGroup.add(groundInst);
		}

		// Add 3D cluster objects
		const dummy = new THREE.Object3D();

		for (const cluster of clusters) {
			switch (cluster.terrainType) {
				case TerrainCode.MOUNTAIN:
					addMountain(cluster, offsetX, offsetZ);
					break;
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
		if (gen !== buildGeneration) return;

		cloudSystem = createCloudSystem(scene);

		const waterfallSources = findWaterfallSources(grid, offsetX, offsetZ);
		waterfallSystem = createWaterfallSystem(scene, waterfallSources);

		const settlementClusters = clusters.filter(c => c.terrainType === TerrainCode.SETTLEMENT);
		wildlifeSystem = createWildlifeSystem(scene);

		// === Weather system ===
		const mountainClusters = clusters.filter(c => c.terrainType === TerrainCode.MOUNTAIN);
		const mountainPositions = mountainClusters.map(c => ({
			x: c.centerX + offsetX + 0.5,
			z: c.centerY + offsetZ + 0.5,
			radius: Math.sqrt(c.size) * 0.6
		}));

		// Find interior water cells (not on perimeter) = lakes
		const lakePositions: { x: number; z: number }[] = [];
		for (let y = 1; y < rows - 1; y++) {
			for (let x = 1; x < cols - 1; x++) {
				if (grid[y][x] === TerrainCode.OCEAN) {
					// Check if surrounded by non-ocean (interior water)
					const neighbors = [grid[y-1]?.[x], grid[y+1]?.[x], grid[y]?.[x-1], grid[y]?.[x+1]];
					const hasLand = neighbors.some(n => n !== undefined && n !== TerrainCode.OCEAN);
					if (hasLand) {
						lakePositions.push({ x: x + offsetX + 0.5, z: y + offsetZ + 0.5 });
					}
				}
			}
		}

		weatherSystem = createWeatherSystem(scene, mountainPositions, lakePositions);

		await nextFrame();
		if (gen !== buildGeneration) return;

		// === Creatures (humans, raiders, animals) — visible in orbit + FP ===
		if (creatureSystem) { creatureSystem.dispose(); creatureSystem = null; }
		const orbitHeightFn = (x: number, z: number) => {
			const gx = Math.floor(x - offsetX);
			const gz = Math.floor(z - offsetZ);
			if (gz >= 0 && gz < rows && gx >= 0 && gx < cols) {
				return getTerrainHeight(grid[gz][gx]) / 2;
			}
			return 0.06;
		};
		createCreatures(scene, grid, orbitHeightFn, settlementClusters).then(cs => {
			if (gen === buildGeneration) creatureSystem = cs;
		});

		// === Scatter vegetation (trees, flowers, rocks) — always visible ===
		if (scatter) { scatter.dispose(); scatter = null; }
		createScatter(grid, terrainSystem.getHeightAt, 3000).then(s => {
			if (gen !== buildGeneration) return;
			scatter = s;
			scene.add(scatter.group);
		});

		// === Roads between settlements ===
		buildRoads(settlementClusters, offsetX, offsetZ);

		// Restore FP/flythrough state if active during rebuild
		if (fpMode && fpSky) {
			scene.add(fpSky);
			scene.background = null;
		}
		if (fpMode && terrainSystem) {
			terrainSystem.mesh.visible = true;
			if (blockyTerrainGroup) blockyTerrainGroup.visible = false;
		}
		if (flythroughActive && terrainSystem) {
			flythrough = createFlythrough(grid, terrainSystem.getHeightAt);
			flythrough.start(camera);
		}
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

		// Draw roads as thin box segments
		for (const edge of roadEdges) {
			const a = settlements[edge.from];
			const b = settlements[edge.to];
			const ax = a.centerX + ox + 0.5;
			const az = a.centerY + oz + 0.5;
			const bx = b.centerX + ox + 0.5;
			const bz = b.centerY + oz + 0.5;

			const dx = bx - ax;
			const dz = bz - az;
			const len = Math.sqrt(dx * dx + dz * dz);
			const angle = Math.atan2(dz, dx);

			// Road surface
			const roadGeo = new THREE.BoxGeometry(len, 0.015, 0.18);
			const road = new THREE.Mesh(roadGeo, roadMat);
			road.position.set(ax + dx / 2, 0.11, az + dz / 2);
			road.rotation.y = -angle;
			road.receiveShadow = true;
			scene.add(road);

			// Road edge markers (small stones along the road)
			const rng = mulberry32(Math.floor(ax * 100 + az * 7));
			const stoneCount = Math.floor(len * 2);
			const stoneMat = new THREE.MeshStandardMaterial({ color: 0x8a8878, roughness: 0.9 });
			for (let i = 0; i < stoneCount; i++) {
				const t = (i + 0.5) / stoneCount;
				const sx = ax + dx * t + (rng() - 0.5) * 0.15;
				const sz = az + dz * t + (rng() - 0.5) * 0.15;
				const side = rng() > 0.5 ? 0.12 : -0.12;
				const cosA = Math.cos(-angle);
				const sinA = Math.sin(-angle);
				const stoneGeo = new THREE.DodecahedronGeometry(0.02 + rng() * 0.02, 0);
				const stone = new THREE.Mesh(stoneGeo, stoneMat);
				stone.position.set(sx - sinA * side, 0.12, sz + cosA * side);
				scene.add(stone);
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

				// Trunk
				dummy.position.set(px, 0.1 + h * 0.25, pz);
				dummy.scale.set(s * 0.8, h * 1.0, s * 0.8);
				dummy.rotation.set(0, 0, 0);
				dummy.updateMatrix();
				trunkInst.setMatrixAt(idx, dummy.matrix);

				// Bottom canopy layer (widest)
				dummy.position.set(px, 0.1 + h * 0.45, pz);
				dummy.scale.set(s * 1.3, h * 0.9, s * 1.3);
				dummy.updateMatrix();
				canopy1.setMatrixAt(idx, dummy.matrix);

				// Middle canopy layer
				dummy.position.set(px, 0.1 + h * 0.65, pz);
				dummy.scale.set(s * 1.1, h * 0.8, s * 1.1);
				dummy.updateMatrix();
				canopy2.setMatrixAt(idx, dummy.matrix);

				// Top canopy layer (narrowest)
				dummy.position.set(px, 0.1 + h * 0.82, pz);
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

			// Fire lights — more for towns
			const maxFires = isTown ? 5 : 3;
			if (i < maxFires && settlementLights.length < 24) {
				const fireLight = new THREE.PointLight(0xff6622, 1.2, 6, 1.5);
				fireLight.position.set(px + (rng() - 0.5) * 0.3, 0.45, pz + (rng() - 0.5) * 0.3);
				fireLight.castShadow = false;
				scene.add(fireLight);
				settlementLights.push(fireLight);

				const fireMat = new THREE.MeshBasicMaterial({ color: 0xff6622, transparent: true, opacity: 0.85 });
				const fireGeo = new THREE.SphereGeometry(0.06, 6, 6);
				const fireMesh = new THREE.Mesh(fireGeo, fireMat);
				fireMesh.position.copy(fireLight.position);
				fireMesh.position.y -= 0.1;
				scene.add(fireMesh);
				fireParticles.push(fireMesh);
			}

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
				stone.position.set(
					cell.x + ox + 0.2 + rng() * 0.6,
					0.1 + h / 2,
					cell.y + oz + 0.2 + rng() * 0.6
				);
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

			// Throttle shadow re-renders to max once per 300ms — prevents slider lag
			const now = performance.now();
			if (force || now - lastShadowUpdate > 2000) {
				sunLight.shadow.needsUpdate = true;
				lastShadowUpdate = now;
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
		if (cloudSystem) cloudSystem.update(dt, dn.skyColor);
		if (waterfallSystem) waterfallSystem.update(dt);
		if (wildlifeSystem) wildlifeSystem.update(dt, timeOfDay);
		if (weatherSystem) weatherSystem.update(dt, timeOfDay);

		// Settlement fires — flicker and scale with nighttime
		const isNight = timeOfDay < 6 || timeOfDay > 18;
		const nightBlend = isNight ? 1.0 : (timeOfDay < 7 ? (7 - timeOfDay) : timeOfDay > 17 ? (timeOfDay - 17) : 0);
		const fireIntensity = nightBlend * 1.2;

		for (let li = 0; li < settlementLights.length; li++) {
			const light = settlementLights[li];
			const flicker = Math.sin(now * 8 + li * 2.3) * 0.3 + Math.sin(now * 13 + li * 5.1) * 0.15;
			// Fires visible day and night but much brighter at night
			light.intensity = 0.3 + fireIntensity * (0.8 + flicker);
			light.distance = 4 + fireIntensity * 3;
		}

		// Fire particle glow animation
		for (let fi = 0; fi < fireParticles.length; fi++) {
			const fp = fireParticles[fi];
			const mat = fp.material as THREE.MeshBasicMaterial;
			const pulse = 0.5 + Math.sin(now * 6 + fi * 3) * 0.3;
			mat.opacity = (0.3 + fireIntensity * 0.6) * pulse;
			fp.scale.setScalar(0.8 + Math.sin(now * 10 + fi) * 0.3);
			fp.position.y += Math.sin(now * 4 + fi * 2) * 0.0003;
		}

		// First-person mode: update FP controller or flythrough
		// Update creatures in all modes (walking, raiding, roaming)
		if (creatureSystem) {
			creatureSystem.update(dt, timeOfDay);
		}

		// Frustum + distance culling on scatter objects (all modes)
		if (scatter) scatter.updateCulling(camera);

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
				// Sample sky color from daynight for fog tint
				const dn = computeDayNight(timeOfDay);
				fpFog.color.copy(dn.fogColor);
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
		scene.background = new THREE.Color(0x87ceeb);

		camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 250);
		camera.position.set(28, 22, 28);
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
		cleanupSystems();
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

	<!-- FP mode buttons -->
	<div class="absolute top-3 right-3 z-10 flex gap-2">
		<button
			class="px-3 py-1.5 text-[11px] font-medium rounded border transition-colors backdrop-blur-sm
				{flythroughActive
					? 'border-neon-gold text-neon-gold bg-cyber-surface/80 hover:bg-neon-gold/20 animate-pulse'
					: 'border-cyber-border text-cyber-muted bg-cyber-surface/60 hover:border-neon-gold/40 hover:text-cyber-fg'
				}"
			onclick={handleFlythroughToggle}
			title={flythroughActive ? 'Stop flythrough' : 'Epic FPV flythrough'}
		>
			{flythroughActive ? 'Stop Tour' : 'Flythrough'}
		</button>
		<button
			class="px-3 py-1.5 text-[11px] font-medium rounded border transition-colors backdrop-blur-sm
				{fpMode && !flythroughActive
					? 'border-neon-cyan text-neon-cyan bg-cyber-surface/80 hover:bg-neon-cyan/20'
					: 'border-cyber-border text-cyber-muted bg-cyber-surface/60 hover:border-neon-cyan/40 hover:text-cyber-fg'
				}"
			onclick={handleFPToggle}
			title={fpMode ? 'Exit first-person' : 'Enter first-person mode'}
		>
			{fpMode ? 'Exit FP' : 'First Person'}
		</button>
	</div>

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

		<!-- FP/flythrough mode indicator -->
		<div class="absolute bottom-3 left-3 z-10 text-[10px] text-cyber-muted/60 pointer-events-none">
			{#if flythroughActive}
				FPV FLYTHROUGH &middot; Click "Stop Tour" to exit
			{:else}
				FP MODE &middot; WASD + Mouse &middot; Shift = Sprint
			{/if}
		</div>

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
		{/if}
	{/if}
</div>

<style>
	@keyframes hudFadeIn {
		from { opacity: 0; transform: translateY(-10px); }
		to { opacity: 1; transform: translateY(0); }
	}
</style>
