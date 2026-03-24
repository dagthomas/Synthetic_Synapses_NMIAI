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
	import { createTerrain, computeTerrainMorphData, applyNormalMap, updateTerrainHeightFn, getCurrentSeason, getSeasonalTreeTint, WATER_LEVEL, type TerrainSystem, type Season } from './terrain';
	import { createFPController, type FPController } from './fpcontrols';
	import { createPostFX, type PostFXPipeline } from './postfx';
	import { createAtmosphere, type AtmosphereSystem } from './atmosphere';
	import { Sky } from 'three/addons/objects/Sky.js';
	import { createFlythrough, type FlythroughSystem } from './flythrough';
	import { createScatter, type ScatterSystem } from './scatter';
	import { createCreatures, type CreatureSystem } from './creatures';
	import { createWater, createMoatAndWaterfalls, type WaterSystem, type MoatSystem } from './water';
	import { LightProbeGenerator } from 'three/addons/lights/LightProbeGenerator.js';
	import { createWind, applyWindSway, type WindSystem } from './wind';
	import { createNightSky, type NightSkySystem } from './nightsky';

	let {
		grid,
		settlements = [],
		showScores = false,
		showGrid = false,
		showPrediction = false,
		showTerrain = true,
		timeOfDay = 12,
		freezeCamera = false,
		seedLabel = '',
		roundLabel = '',
		roundId = undefined,
		roundNumber = undefined,
		seedIndex = undefined,
		onCaptureFn = undefined,
		onFlythroughChange = undefined,
		season = getCurrentSeason()
	}: {
		grid: number[][];
		settlements?: Settlement[];
		showScores?: boolean;
		showGrid?: boolean;
		showPrediction?: boolean;
		showTerrain?: boolean;
		timeOfDay?: number;
		freezeCamera?: boolean;
		seedLabel?: string;
		roundLabel?: string;
		roundId?: string | null;
		roundNumber?: number;
		seedIndex?: number;
		onCaptureFn?: (fn: () => string) => void;
		onFlythroughChange?: (active: boolean) => void;
		season?: Season;
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
	let activeWaterfalls: {
		points: THREE.Points; topH: number; botH: number; midX: number; midZ: number;
		peakX: number; peakZ: number; peakH: number;
		edgeX: number; edgeZ: number; edgeH: number;
		watX: number; watZ: number; watH: number;
	}[] = [];
	let fireParticles: THREE.Mesh[] = [];
	let settlementPositions: THREE.Vector3[] = [];
	let lastInteraction = 0;
	let autoRotateAngle = 0;

	// Idle cinema mode — random still shots with handheld sway, camera looks toward center
	const IDLE_SHOT_DURATION = 10;
	const IDLE_TRANSITION_DURATION = 2;
	let idleCinemaActive = false;
	let idleShotTimer = 0;
	let idleShotPhase: 'hold' | 'transition' = 'hold';
	let idleTransitionProgress = 0;
	let idleCamFrom = { px: 0, py: 0, pz: 0, lx: 0, ly: 0, lz: 0 };
	let idleCamTo = { px: 0, py: 0, pz: 0, lx: 0, ly: 0, lz: 0 };
	let idleCamCurrent = { px: 0, py: 0, pz: 0, lx: 0, ly: 0, lz: 0 };
	let idleHandheldTime = 0;

	function generateIdleShot(): { px: number; py: number; pz: number; lx: number; ly: number; lz: number } {
		const rows = grid?.length ?? 40;
		const cols = grid?.[0]?.length ?? 40;
		// Try up to 10 random positions, prefer land (height > 0)
		let bestPx = 0, bestPz = 0, bestH = -999;
		for (let attempt = 0; attempt < 10; attempt++) {
			const angle = Math.random() * Math.PI * 2;
			const radius = 6 + Math.random() * 12;
			const px = Math.cos(angle) * radius;
			const pz = Math.sin(angle) * radius;
			const h = gY(px, pz);
			if (h > bestH) {
				bestPx = px;
				bestPz = pz;
				bestH = h;
			}
			if (h > 0.05) break; // found land, good enough
		}
		// Camera 10m above ground, minimum 1m above water
		const py = Math.max(bestH + 10, 1);
		// Look toward center, ground level clamped above water
		const centerY = Math.max(gY(0, 0), 0.1);
		return { px: bestPx, py, pz: bestPz, lx: 0, ly: centerY, lz: 0 };
	}

	function smoothstep(t: number): number {
		return t * t * (3 - 2 * t);
	}
	let buildGeneration = 0; // incremented each build to cancel stale async work
	let morphFrameCount = 0;
	let terrainWorker: Worker | null = null;

	function getTerrainWorker(): Worker {
		if (terrainWorker) return terrainWorker;
		// Inline Blob worker — self-contained, no import issues
		// Seasonal biome colors and snow thresholds passed via message
		const code = `
const BH={0:.12,1:.16,2:.04,3:.16,4:.2,5:.36,10:-.55,11:.14};
const SBC={summer:{0:[.82,.74,.5],1:[.7,.6,.38],2:[.45,.55,.52],3:[.56,.52,.44],4:[.2,.44,.14],5:[.5,.48,.44],10:[.1,.25,.42],11:[.36,.62,.18]},spring:{0:[.82,.74,.5],1:[.68,.6,.4],2:[.45,.55,.52],3:[.52,.52,.42],4:[.28,.52,.18],5:[.5,.48,.44],10:[.1,.26,.44],11:[.42,.65,.22]},autumn:{0:[.8,.72,.48],1:[.68,.56,.36],2:[.44,.52,.5],3:[.58,.5,.4],4:[.55,.35,.12],5:[.5,.48,.44],10:[.1,.22,.38],11:[.62,.52,.2]},winter:{0:[.78,.74,.58],1:[.64,.58,.46],2:[.48,.54,.54],3:[.54,.52,.48],4:[.3,.28,.24],5:[.58,.56,.54],10:[.08,.2,.38],11:[.45,.4,.32]}};
const SNT={summer:{s:2.4,f:3.5},spring:{s:2,f:3.2},autumn:{s:1.8,f:2.8},winter:{s:.8,f:1.6}};
const SC=[.94,.94,.97],WC=[.58,.52,.38],DC=[.65,.58,.42];
const NA={0:.018,1:.012,2:.01,3:.03,4:.035,5:.12,10:.008,11:.02};
function h(x,y){const n=Math.sin(x*127.1+y*311.7)*43758.5453;return n-Math.floor(n)}
function sn(x,z){const ix=Math.floor(x),iz=Math.floor(z),fx=x-ix,fz=z-iz,sx=fx*fx*(3-2*fx),sz=fz*fz*(3-2*fz);return h(ix,iz)*(1-sx)*(1-sz)+h(ix+1,iz)*sx*(1-sz)+h(ix,iz+1)*(1-sx)*sz+h(ix+1,iz+1)*sx*sz}
function fbm(x,z,o,l,g){let v=0,a=1,f=1,m=0;for(let i=0;i<o;i++){v+=sn(x*f,z*f)*a;m+=a;a*=g;f*=l}return v/m}
function rn(x,z,o){let v=0,a=1,f=1,m=0;for(let i=0;i<o;i++){let n=sn(x*f,z*f);n=1-Math.abs(n*2-1);n*=n;v+=n*a;m+=a;a*=.5;f*=2.1}return v/m}
function wfbm(x,z,o){return fbm(x+fbm(x+5.2,z+1.3,3,2,.5)*1.5,z+fbm(x+1.7,z+9.2,3,2,.5)*1.5,o,2,.5)}
function bmf(g){const r=g.length,c=g[0].length,f=new Float32Array(r*c);for(let y=0;y<r;y++)for(let x=0;x<c;x++){let n=0;for(let dy=-3;dy<=3;dy++)for(let dx=-3;dx<=3;dx++){const cx=x+dx,cy=y+dy;if(cx>=0&&cx<c&&cy>=0&&cy<r&&g[cy][cx]===5)n++}f[y*c+x]=n}return f}
function smf(f,c,r,gx,gz){const cx=Math.max(0,Math.min(c-1.001,gx)),cz=Math.max(0,Math.min(r-1.001,gz)),ix=Math.floor(cx),iz=Math.floor(cz),fx=cx-ix,fz=cz-iz,ix1=Math.min(ix+1,c-1),iz1=Math.min(iz+1,r-1);return f[iz*c+ix]*(1-fx)*(1-fz)+f[iz*c+ix1]*fx*(1-fz)+f[iz1*c+ix]*(1-fx)*fz+f[iz1*c+ix1]*fx*fz}
function bbh(g,gx,gz){const r=g.length,c=g[0].length,s2=1.8;let t=0,w=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const cx=Math.round(gx)+dx,cy=Math.round(gz)+dy;if(cx<0||cx>=c||cy<0||cy>=r)continue;const d2=(gx-cx)*(gx-cx)+(gz-cy)*(gz-cy),wt=Math.exp(-d2/s2);t+=(BH[g[cy][cx]]??0.06)*wt;w+=wt}return w>0?t/w:.06}
function sfbm(x,z){return fbm(x,z,2,2,.5)}
function ch(g,mf,c,r,gx,gz){let ht=bbh(g,gx,gz);const tix=Math.floor(Math.max(0,Math.min(c-1,gx+.5))),tiy=Math.floor(Math.max(0,Math.min(r-1,gz+.5))),code=g[tiy]?.[tix]??0;let mp=0;for(let dy=-3;dy<=3;dy++)for(let dx=-3;dx<=3;dx++){const cx=Math.round(gx)+dx,cy=Math.round(gz)+dy;if(cx<0||cx>=c||cy<0||cy>=r||g[cy][cx]!==5)continue;const d=Math.sqrt((gx-cx)*(gx-cx)+(gz-cy)*(gz-cy)),R=2.8;if(d>=R)continue;const t=d/R;mp=Math.max(mp,(1-t*t)*(1-t*t))}if(mp>0){const dn=smf(mf,c,r,gx,gz),ps=dn>=12?4:dn>=6?3.2:dn>=3?2.2:1.5;ht+=mp*ps}ht+=(sfbm(gx*.8,gz*.8)-.5)*2*(NA[code]??.015);if(mp>.2)ht+=rn(gx*.5,gz*.5,2)*mp*.6;return ht}
function bc(g,gx,gz,BC){const r=g.length,c=g[0].length,s2=1.4;let R=0,G=0,B=0,w=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const cx=Math.round(gx)+dx,cy=Math.round(gz)+dy;if(cx<0||cx>=c||cy<0||cy>=r)continue;const d2=(gx-cx)*(gx-cx)+(gz-cy)*(gz-cy),wt=Math.exp(-d2/s2),cl=BC[g[cy][cx]]??DC;R+=cl[0]*wt;G+=cl[1]*wt;B+=cl[2]*wt;w+=wt}return w>0?[R/w,G/w,B/w]:[...DC]}
self.onmessage=function(e){const g=e.data.grid,sn=e.data.season||'summer',BC=SBC[sn]||SBC.summer,st=SNT[sn]||SNT.summer,r=g.length,c=g[0].length,mf=bmf(g),sd=2,sx=c*sd,sz=r*sd,vc=(sx+1)*(sz+1),hs=new Float32Array(vc),cs=new Float32Array(vc*3);for(let iz=0;iz<=sz;iz++)for(let ix=0;ix<=sx;ix++){const i=iz*(sx+1)+ix,wx=(ix/sx-.5)*c,wz=(iz/sz-.5)*r,gx=wx+c/2-.5,gz=wz+r/2-.5,ht=ch(g,mf,c,r,gx,gz);hs[i]=ht;let[R,G,B]=bc(g,gx,gz,BC);if(ht>st.s){const t=Math.min(1,(ht-st.s)/(st.f-st.s));R=R*(1-t)+SC[0]*t;G=G*(1-t)+SC[1]*t;B=B*(1-t)+SC[2]*t}if(ht>-.15&&ht<.12){const wt=1-Math.max(0,Math.min(1,(ht+.15)/.27));R=R*(1-wt*.6)+WC[0]*wt*.6;G=G*(1-wt*.6)+WC[1]*wt*.6;B=B*(1-wt*.6)+WC[2]*wt*.6}const cn=(sfbm(gx*2.5,gz*2.5)-.5)*.05;cs[i*3]=Math.max(0,Math.min(1,R+cn));cs[i*3+1]=Math.max(0,Math.min(1,G+cn*.7));cs[i*3+2]=Math.max(0,Math.min(1,B+cn*.5))}self.postMessage({heights:hs,colors:cs,vertexCount:vc},[hs.buffer,cs.buffer])};
`;
		const blob = new Blob([code], { type: 'application/javascript' });
		terrainWorker = new Worker(URL.createObjectURL(blob));
		return terrainWorker;
	}
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
	let moatSystem: MoatSystem | null = null;
	let nightSkySystem: NightSkySystem | null = null;
	let windSystem: WindSystem | null = null;
	let gridOverlay: THREE.Group | null = null;
	let predictionOverlay: THREE.Group | null = null;
	let terrainMorphOldHeights: Float32Array | null = null;
	let terrainMorphNewHeights: Float32Array | null = null;
	let terrainMorphOldNormals: Float32Array | null = null;
	let terrainMorphNewNormals: Float32Array | null = null;
	let predictionAnalysis: { ground_truth: number[][][]; prediction: number[][][] } | null = null;
	let predictionLoading = false;
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

	// --- Terrain-only height morph state ---
	let terrainMorphing = false;
	let terrainMorphAlpha = 0;
	let terrainMorphStartHeights: Float32Array | null = null;
	let terrainMorphTargetHeights: Float32Array | null = null;
	let terrainMorphStartColors: Float32Array | null = null;
	let terrainMorphTargetColors: Float32Array | null = null;
	let pendingTerrainSystem: TerrainSystem | null = null;
	let pendingMorphGrid: number[][] | null = null;
	let lastGridRows = 0;
	let lastGridCols = 0;
	const TERRAIN_MORPH_DURATION = 0.8;

	// --- Scenery sink/rise state ---
	let sceneryGroup: THREE.Group | null = null;
	let sceneryPhase: 'sinking' | 'rising' | null = null;
	let sceneryAlpha = 0;
	let pendingGrid: number[][] | null = null;
	let pendingSettlements: typeof settlements | null = null;
	const SCENERY_SINK_DURATION = 0.8; // match TERRAIN_MORPH_DURATION to avoid force-snap
	const SCENERY_RISE_DURATION = 0.6;
	const SCENERY_SINK_DEPTH = 4.0;

	// --- Sky transition state ---
	let skyFrozen = false;
	let skyFrozenTime = 0;
	let skyTransitionActive = false;
	let skyTransitionAlpha = 0;
	let skyTransitionFrom = 0;
	let skyTransitionTo = 0;
	const SKY_TRANSITION_DURATION = 2.5;

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

	/** Cleanup grid-dependent systems only (terrain, creatures, scatter, weather).
	 *  Water/moat are kept alive — they only depend on grid dimensions, not content. */
	function cleanupGridSystems() {
		if (terrainSystem) { terrainSystem.dispose(); terrainSystem = null; }
		if (creatureSystem) { creatureSystem.dispose(); creatureSystem = null; }
		if (scatter) { scatter.dispose(); scatter = null; }
		if (weatherSystem) { weatherSystem.dispose(); weatherSystem = null; }
		blockyTerrainGroup = null;
		settlementLights = [];
		fireParticles = [];
		settlementPositions = [];
	}

	/** Set opacity on all meshes in a group for cross-fade transitions.
	 *  Disables depthWrite when transparent so overlapping old/new terrain
	 *  both render instead of occluding each other via depth buffer. */
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
					m.depthWrite = fullyOpaque;
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
			terrainSystem = createTerrain(grid, { season });
			scene.add(terrainSystem.mesh);
		}

		// Camera: wide FOV, close near plane for immersion
		camera.fov = 115;
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

		// Higher quality shadows for ground-level
		if (sunLight) {
			const shadowRes = flythroughActive ? 1024 : 4096;
			sunLight.shadow.mapSize.width = shadowRes;
			sunLight.shadow.mapSize.height = shadowRes;
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
		if (e.code === 'KeyG') {
			if (gridOverlay) gridOverlay.visible = !gridOverlay.visible;
		}
		if (e.code === 'KeyP') {
			togglePredictionOverlay();
		}
		if (e.code === 'KeyT') {
			if (terrainSystem) terrainSystem.mesh.visible = !terrainSystem.mesh.visible;
		}
	}

	async function togglePredictionOverlay() {
		if (!scene || !grid?.length) return;

		// If overlay exists, toggle visibility
		if (predictionOverlay) {
			predictionOverlay.visible = !predictionOverlay.visible;
			return;
		}

		// Fetch analysis data if not loaded — try local calibration first, then API
		if (!predictionAnalysis && !predictionLoading) {
			predictionLoading = true;
			const si = seedIndex ?? 0;
			let loaded = false;

			// Try local calibration data (by round number)
			if (roundNumber) {
				try {
					const localUrl = `http://localhost:7091/api/local/analysis/${roundNumber}/${si}`;
						const resp = await fetch(localUrl);
					if (resp.ok) {
						predictionAnalysis = await resp.json();
						loaded = true;
						}
				} catch { /* fall through to API */ }
			}

			// Fall back to external API
			if (!loaded && roundId) {
				try {
					const apiUrl = `http://localhost:7091/api/rounds/${roundId}/seeds/${si}/analysis`;
						const resp = await fetch(apiUrl);
					if (resp.ok) {
						predictionAnalysis = await resp.json();
						loaded = true;
						} else {
						}
				} catch (e) { /* fetch failed */ }
			}

				predictionLoading = false;
		}

		if (!predictionAnalysis?.prediction || !predictionAnalysis?.ground_truth) {
				return;
		}

		// Build the overlay
		predictionOverlay = buildPredictionOverlay(predictionAnalysis, grid);
		scene.add(predictionOverlay);
		}

	function argmax(probs: number[]): number {
		let best = 0;
		for (let i = 1; i < probs.length; i++) {
			if (probs[i] > probs[best]) best = i;
		}
		return best;
	}

	function buildPredictionOverlay(
		analysis: { ground_truth: number[][][]; prediction: number[][][] },
		g: number[][]
	): THREE.Group {
		const rows = g.length, cols = g[0].length;
		const ox = -cols / 2 + 0.5, oz = -rows / 2 + 0.5;
		const cellCount = rows * cols;

		const positions = new Float32Array(cellCount * 4 * 3); // 4 verts per cell
		const colors = new Float32Array(cellCount * 4 * 4);    // RGBA
		const indices: number[] = [];

		const GREEN = [0.1, 0.85, 0.2, 0.5];
		const RED   = [0.9, 0.15, 0.1, 0.5];

		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const idx = r * cols + c;
				const wx = c + ox, wz = r + oz;
				const h = gY(wx, wz) + 0.12;

				// Compare prediction vs ground truth
				const predClass = argmax(analysis.prediction[r][c]);
				const trueClass = argmax(analysis.ground_truth[r][c]);
				const correct = predClass === trueClass;
				const col = correct ? GREEN : RED;

				// 4 vertices for quad
				const vi = idx * 4;
				const pi = vi * 3, ci = vi * 4;
				const half = 0.48;

				positions[pi]     = wx - half; positions[pi + 1] = h; positions[pi + 2]  = wz - half;
				positions[pi + 3] = wx + half; positions[pi + 4] = h; positions[pi + 5]  = wz - half;
				positions[pi + 6] = wx + half; positions[pi + 7] = h; positions[pi + 8]  = wz + half;
				positions[pi + 9] = wx - half; positions[pi + 10] = h; positions[pi + 11] = wz + half;

				for (let v = 0; v < 4; v++) {
					colors[ci + v * 4]     = col[0];
					colors[ci + v * 4 + 1] = col[1];
					colors[ci + v * 4 + 2] = col[2];
					colors[ci + v * 4 + 3] = col[3];
				}

				indices.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);
			}
		}

		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
		geo.setAttribute('color', new THREE.BufferAttribute(colors, 4));
		geo.setIndex(indices);
		geo.computeVertexNormals();

		const mat = new THREE.ShaderMaterial({
			vertexShader: `
				attribute vec4 color;
				varying vec4 vColor;
				void main() {
					vColor = color;
					gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
				}
			`,
			fragmentShader: `
				varying vec4 vColor;
				void main() {
					gl_FragColor = vColor;
				}
			`,
			transparent: true,
			depthWrite: false,
			side: THREE.DoubleSide,
		});

		const mesh = new THREE.Mesh(geo, mat);
		mesh.renderOrder = 5;

		const group = new THREE.Group();
		group.add(mesh);
		return group;
	}

	async function startFlythrough() {
		if (!scene || !camera || !grid?.length || flythroughLoading) return;

		flythroughLoading = true;
		flythroughLoadProgress = 0;
		await nextFrame(); // let loading overlay paint

		// Step 1: Terrain
		flythroughLoadProgress = 0.1;
		if (!terrainSystem) {
			terrainSystem = createTerrain(grid, { season });
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

	}

	function stopFlythrough() {
		if (flythrough) {
			flythrough.stop();
		}
		flythroughActive = false;
		onFlythroughChange?.(false);
		// Exit fullscreen
		try { document.exitFullscreen?.(); } catch {}
		if (minimapInterval) { clearInterval(minimapInterval); minimapInterval = undefined; }
		// Return to FP walking mode
		if (fpController) {
			fpController.controls.addEventListener('unlock', onFPUnlock);
			fpController.controls.addEventListener('lock', onFPLock);
		}
	}

	// Grid overlay colors — maximally distinct, high saturation
	const TERRAIN_RGB: Record<number, [number, number, number]> = {
		0:  [0.90, 0.80, 0.35],  // Empty/Sand: bright gold
		1:  [1.00, 0.50, 0.00],  // Settlement: vivid orange
		2:  [0.00, 0.55, 1.00],  // Port: bright cyan-blue
		3:  [0.90, 0.15, 0.15],  // Ruin: vivid red
		4:  [0.05, 0.35, 0.05],  // Forest: dark deep green
		5:  [0.60, 0.55, 0.70],  // Mountain: lavender-gray
		10: [0.05, 0.10, 0.55],  // Ocean: dark navy
		11: [0.70, 0.95, 0.10],  // Plains: neon yellow-green
	};
	const DEFAULT_RGB: [number, number, number] = [0.50, 0.50, 0.50];

	/** Build a 3D grid overlay: floor tiles + 2m vertical walls per cell border,
	 *  colored by terrain type, fading to transparent at the top. */
	function buildGridOverlay(g: number[][], heightFn: (x: number, z: number) => number): THREE.Group {
		const r = g.length, c = g[0].length;
		const wallH = 2.0;
		const group = new THREE.Group();

		// Debug: log terrain code distribution
		const codeCounts = new Map<number, number>();
		for (let z = 0; z < r; z++) for (let x = 0; x < c; x++) {
			const code = g[z][x];
			codeCounts.set(code, (codeCounts.get(code) ?? 0) + 1);
		}
	
		// --- Floor: colored transparent tiles ---
		const floorPositions: number[] = [];
		const floorColors: number[] = [];
		const floorAlphas: number[] = [];
		const floorIndices: number[] = [];

		for (let z = 0; z < r; z++) {
			for (let x = 0; x < c; x++) {
				const code = g[z][x];
				const rgb = TERRAIN_RGB[code] ?? DEFAULT_RGB;
				const wx = x - c / 2 + 0.5;
				const wz = z - r / 2 + 0.5;
				const h = heightFn(wx, wz) + 0.06;
				const hs = 0.48; // half cell size (slight gap)
				const vi = floorPositions.length / 3;

				// Water cells: very transparent to blend with actual water below
				const isWater = code === 10 || code === 2;
				const floorAlpha = isWater ? 0.12 : 0.45;

				// 4 corners of the cell
				floorPositions.push(wx - hs, h, wz - hs);
				floorPositions.push(wx + hs, h, wz - hs);
				floorPositions.push(wx + hs, h, wz + hs);
				floorPositions.push(wx - hs, h, wz + hs);
				for (let k = 0; k < 4; k++) {
					floorColors.push(rgb[0], rgb[1], rgb[2]);
					floorAlphas.push(floorAlpha);
				}
				floorIndices.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);
			}
		}

		const floorGeo = new THREE.BufferGeometry();
		floorGeo.setAttribute('position', new THREE.Float32BufferAttribute(floorPositions, 3));
		floorGeo.setAttribute('color', new THREE.Float32BufferAttribute(floorColors, 3));
		floorGeo.setAttribute('aAlpha', new THREE.Float32BufferAttribute(floorAlphas, 1));
		floorGeo.setIndex(floorIndices);

		// --- Flood-fill cluster sizes per cell (connected same-type neighbors) ---
		const clusterSize = new Int32Array(c * r);
		const visited = new Uint8Array(c * r);
		for (let z = 0; z < r; z++) {
			for (let x = 0; x < c; x++) {
				if (visited[z * c + x]) continue;
				const code = g[z][x];
				// BFS flood fill
				const queue: [number, number][] = [[x, z]];
				const cells: number[] = [];
				visited[z * c + x] = 1;
				while (queue.length > 0) {
					const [cx, cz] = queue.pop()!;
					cells.push(cz * c + cx);
					for (const [dx, dz] of [[1,0],[-1,0],[0,1],[0,-1]] as const) {
						const nx = cx + dx, nz = cz + dz;
						if (nx >= 0 && nx < c && nz >= 0 && nz < r && !visited[nz * c + nx] && g[nz][nx] === code) {
							visited[nz * c + nx] = 1;
							queue.push([nx, nz]);
						}
					}
				}
				for (const idx of cells) clusterSize[idx] = cells.length;
			}
		}

		// --- Walls: vertical quads at cell borders, height scales with cluster size ---
		const wallPositions: number[] = [];
		const wallColors: number[] = [];
		const wallAlphas: number[] = [];
		const wallIndices: number[] = [];

		function addWall(x1: number, z1: number, x2: number, z2: number, rgb: [number, number, number], h: number) {
			const h1 = heightFn((x1 + x2) / 2, (z1 + z2) / 2) + 0.06;
			const vi = wallPositions.length / 3;
			wallPositions.push(x1, h1, z1);
			wallPositions.push(x2, h1, z2);
			wallPositions.push(x2, h1 + h, z2);
			wallPositions.push(x1, h1 + h, z1);

			for (let k = 0; k < 4; k++) wallColors.push(rgb[0], rgb[1], rgb[2]);
			wallAlphas.push(0.5, 0.5, 0.0, 0.0);
			wallIndices.push(vi, vi + 1, vi + 2, vi, vi + 2, vi + 3);
		}

		const PLAINS_CODE = 11;
		for (let z = 0; z <= r; z++) {
			for (let x = 0; x <= c; x++) {
				const wx = x - c / 2;
				const wz = z - r / 2;
				const ci = Math.min(z, r - 1) * c + Math.min(x, c - 1);
				const cellCode = g[Math.min(z, r - 1)]?.[Math.min(x, c - 1)] ?? 0;
				const rgb = TERRAIN_RGB[cellCode] ?? DEFAULT_RGB;

				// Wall height: base 1.0, scales up with cluster size (except plains)
				const cs = clusterSize[ci] ?? 1;
				const isWaterCell = cellCode === 10 || cellCode === 2;
				const cellWallH = cellCode === PLAINS_CODE
					? wallH * 0.5
					: isWaterCell
						? wallH * 0.3
						: wallH * (0.5 + Math.min(cs, 20) / 20 * 1.5);

				if (x < c) addWall(wx, wz, wx + 1, wz, rgb, cellWallH);
				if (z < r) addWall(wx, wz, wx, wz + 1, rgb, cellWallH);
			}
		}

		const wallGeo = new THREE.BufferGeometry();
		wallGeo.setAttribute('position', new THREE.Float32BufferAttribute(wallPositions, 3));
		wallGeo.setAttribute('color', new THREE.Float32BufferAttribute(wallColors, 3));
		wallGeo.setAttribute('aAlpha', new THREE.Float32BufferAttribute(wallAlphas, 1));
		wallGeo.setIndex(wallIndices);

		// Shared shader material: vertex colors + per-vertex alpha
		const gridMat = new THREE.ShaderMaterial({
			vertexShader: /* glsl */ `
				attribute float aAlpha;
				varying vec3 vColor;
				varying float vAlpha;
				void main() {
					vColor = color;
					vAlpha = aAlpha;
					gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
				}
			`,
			fragmentShader: /* glsl */ `
				varying vec3 vColor;
				varying float vAlpha;
				void main() {
					gl_FragColor = vec4(vColor, vAlpha);
				}
			`,
			transparent: true,
			depthWrite: false,
			side: THREE.DoubleSide,
			vertexColors: true,
		});

		const floorMesh = new THREE.Mesh(floorGeo, gridMat);
		floorMesh.renderOrder = 10;
		group.add(floorMesh);

		const wallMesh = new THREE.Mesh(wallGeo, gridMat);
		wallMesh.renderOrder = 10;
		group.add(wallMesh);

		group.visible = false; // G to toggle
		return group;
	}

	// --- Floating island cone with earthy underside ---
	function buildIslandSlab(cols: number, rows: number) {
		const rng = mulberry32(12345);
		// Diagonal radius covers rectangular terrain corners
		const topRadius = Math.sqrt(cols * cols + rows * rows) / 2 + 4;
		const bottomRadius = 2.0;
		const coneHeight = 26.0;
		const radialSegs = 56;
		const heightSegs = 24;

		// Inverted cone: wide at top (terrain level), narrows to a point below
		// openEnded=true so the top cap doesn't hide terrain lakes/ocean tiles
		const geo = new THREE.CylinderGeometry(topRadius, bottomRadius, coneHeight, radialSegs, heightSegs, true);

		// Perturb vertices: heavy noise for organic mud/roots/carved stone look
		const pos = geo.attributes.position;
		for (let i = 0; i < pos.count; i++) {
			const y = pos.getY(i);
			const normalizedY = (y + coneHeight / 2) / coneHeight; // 0 at bottom, 1 at top
			const x = pos.getX(i);
			const z = pos.getZ(i);

			// Heavy noise for carved stone effect, more in the middle
			const noiseScale = Math.sin(normalizedY * Math.PI) * 1.5;
			// Vertical ridges (root-like grooves)
			const angle = Math.atan2(z, x);
			const ridgeNoise = Math.sin(angle * 8 + normalizedY * 12) * 0.4 * (1 - normalizedY);
			const jitterX = (rng() - 0.5) * noiseScale + Math.cos(angle) * ridgeNoise;
			const jitterZ = (rng() - 0.5) * noiseScale + Math.sin(angle) * ridgeNoise;
			const jitterY = (rng() - 0.5) * 0.3;

			pos.setX(i, pos.getX(i) + jitterX);
			pos.setZ(i, pos.getZ(i) + jitterZ);
			if (normalizedY < 0.95) {
				pos.setY(i, pos.getY(i) + jitterY);
			}
		}
		pos.needsUpdate = true;
		geo.computeVertexNormals();

		// Vertex colors: terrain-matched rim → wet mud → dark earth → gray stone
		const colors = new Float32Array(pos.count * 3);
		const mudCol   = [0.40, 0.30, 0.18]; // wet dark mud
		const rootCol  = [0.30, 0.22, 0.12]; // dark root/earth
		const stoneCol = [0.28, 0.26, 0.24]; // carved stone
		const ox = -cols / 2;
		const oz = -rows / 2;

		for (let i = 0; i < pos.count; i++) {
			const vx = pos.getX(i);
			const vz = pos.getZ(i);
			const y = pos.getY(i);
			const t = (y + coneHeight / 2) / coneHeight;

			// Sample grid terrain color at top-rim vertex position
			const gx = Math.floor(vx - ox);
			const gz = Math.floor(vz - oz);
			let rimCol = DEFAULT_RGB;
			if (gx >= 0 && gx < cols && gz >= 0 && gz < rows) {
				const code = grid[gz][gx];
				rimCol = TERRAIN_RGB[code] ?? DEFAULT_RGB;
			}

			let r: number, g: number, b: number;
			if (t > 0.90) {
				const blend = (t - 0.90) / 0.10;
				r = mudCol[0] + (rimCol[0] - mudCol[0]) * blend;
				g = mudCol[1] + (rimCol[1] - mudCol[1]) * blend;
				b = mudCol[2] + (rimCol[2] - mudCol[2]) * blend;
			} else if (t > 0.55) {
				const blend = (t - 0.55) / 0.35;
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
			// Slight noise for mottled look
			const noise = (rng() - 0.5) * 0.08;
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
		// Position cone so its top overlaps terrain base — prevents gap/shadow strip
		mesh.position.set(0, -coneHeight / 2 + 0.20, 0);
		mesh.receiveShadow = false;
		mesh.castShadow = false;
		scene.add(mesh);

		// --- Opaque moat floor (prevents seeing inside the open cone) ---
		const moatW = 2.5;
		const mfOuterW = cols / 2 + moatW, mfOuterH = rows / 2 + moatW;
		const mfInnerW = cols / 2 - 1.5, mfInnerH = rows / 2 - 1.5;
		const cr = 2.0; // corner radius matching moat water shader
		function rrect(t: THREE.Shape | THREE.Path, hw: number, hh: number, r: number) {
			t.moveTo(-hw + r, -hh);
			t.lineTo(hw - r, -hh);
			t.absarc(hw - r, -hh + r, r, -Math.PI / 2, 0, false);
			t.lineTo(hw, hh - r);
			t.absarc(hw - r, hh - r, r, 0, Math.PI / 2, false);
			t.lineTo(-hw + r, hh);
			t.absarc(-hw + r, hh - r, r, Math.PI / 2, Math.PI, false);
			t.lineTo(-hw, -hh + r);
			t.absarc(-hw + r, -hh + r, r, Math.PI, Math.PI * 1.5, false);
		}
		const mfShape = new THREE.Shape();
		rrect(mfShape, mfOuterW, mfOuterH, cr);
		mfShape.closePath();
		const mfHole = new THREE.Path();
		rrect(mfHole, mfInnerW, mfInnerH, cr * 0.5);
		mfHole.closePath();
		mfShape.holes.push(mfHole);
		const mfGeo = new THREE.ShapeGeometry(mfShape);
		mfGeo.rotateX(-Math.PI / 2);
		const mfMat = new THREE.MeshStandardMaterial({ color: 0x2a5a3a, roughness: 0.85, metalness: 0.02 });
		const mfMesh = new THREE.Mesh(mfGeo, mfMat);
		mfMesh.position.y = -0.15; // below water level so it doesn't show through
		mfMesh.receiveShadow = false;
		scene.add(mfMesh);

		// --- Rim ground (between moat and cone edge) with terrain-matching colors ---
		const rimInnerW = mfOuterW, rimInnerH = mfOuterH;
		const rimCircleSegs = 64;
		const rimShape = new THREE.Shape();
		for (let i = 0; i < rimCircleSegs; i++) {
			const a = (i / rimCircleSegs) * Math.PI * 2;
			const rx = Math.cos(a) * (topRadius - 0.2);
			const rz = Math.sin(a) * (topRadius - 0.2);
			if (i === 0) rimShape.moveTo(rx, rz);
			else rimShape.lineTo(rx, rz);
		}
		rimShape.closePath();
		const rimHole = new THREE.Path();
		rrect(rimHole, rimInnerW, rimInnerH, cr);
		rimHole.closePath();
		rimShape.holes.push(rimHole);
		const rimGeo = new THREE.ShapeGeometry(rimShape);
		rimGeo.rotateX(-Math.PI / 2);
		// Add vertex colors with noise to match terrain palette
		const rimPA = rimGeo.attributes.position;
		const rimCols = new Float32Array(rimPA.count * 3);
		for (let vi = 0; vi < rimPA.count; vi++) {
			const vx = rimPA.getX(vi), vz = rimPA.getZ(vi);
			const n = Math.sin(vx * 2.3 + vz * 1.7) * 0.04 + (rng() - 0.5) * 0.05;
			rimCols[vi * 3]     = Math.max(0, Math.min(1, 0.34 + n));
			rimCols[vi * 3 + 1] = Math.max(0, Math.min(1, 0.56 + n));
			rimCols[vi * 3 + 2] = Math.max(0, Math.min(1, 0.19 + n * 0.7));
		}
		rimGeo.setAttribute('color', new THREE.BufferAttribute(rimCols, 3));
		const rimMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.82, metalness: 0.02 });
		const rimMesh = new THREE.Mesh(rimGeo, rimMat);
		rimMesh.position.y = 0.05;
		rimMesh.receiveShadow = false;
		scene.add(rimMesh);

		// A few hanging stalactites at the bottom
		const stalMat = new THREE.MeshStandardMaterial({ color: 0x5a5248, roughness: 0.9, flatShading: true });
		for (let i = 0; i < 8; i++) {
			const angle = rng() * Math.PI * 2;
			const dist = rng() * bottomRadius * 0.7;
			const sx = Math.cos(angle) * dist;
			const sz = Math.sin(angle) * dist;
			const stalH = 0.4 + rng() * 1.0;
			const stalR = 0.10 + rng() * 0.18;
			const stalGeo = new THREE.ConeGeometry(stalR, stalH, 5);
			const stal = new THREE.Mesh(stalGeo, stalMat);
			stal.position.set(sx, -coneHeight + 0.20 - stalH / 2, sz);
			stal.rotation.x = Math.PI;
			stal.castShadow = true;
			scene.add(stal);
		}
	}

	// --- Rim scenery: trees, rocks, lakes, bushes outside play grid ---
	function buildRimScenery(cols: number, rows: number) {
		const rng = mulberry32(54321);
		const moatWidth = 2.5;
		const halfW = cols / 2, halfH = rows / 2;
		const outerW = halfW + moatWidth, outerH = halfH + moatWidth;
		const topRadius = Math.sqrt(cols * cols + rows * rows) / 2 + 4;
		const groundY = 0.06;
		const riverHW = 1.0;

		function onRim(x: number, z: number): boolean {
			if (x * x + z * z > (topRadius - 1) * (topRadius - 1)) return false;
			if (Math.abs(x) < outerW + 0.3 && Math.abs(z) < outerH + 0.3) return false;
			if (Math.abs(x) < riverHW && Math.abs(z) > outerH) return false;
			if (Math.abs(z) < riverHW && Math.abs(x) > outerW) return false;
			return true;
		}

		// Gather random rim positions
		const pts: { x: number; z: number }[] = [];
		for (let i = 0; i < 800 && pts.length < 120; i++) {
			const a = rng() * Math.PI * 2;
			const r = Math.sqrt(rng()) * topRadius;
			const x = Math.cos(a) * r, z = Math.sin(a) * r;
			if (onRim(x, z)) pts.push({ x, z });
		}

		const dummy = new THREE.Object3D();

		// --- Rocks (first 30 positions) ---
		const nrocks = Math.min(30, pts.length);
		const rockG = new THREE.DodecahedronGeometry(0.15, 1);
		const rockM = new THREE.MeshStandardMaterial({ color: 0x7a7a72, roughness: 0.95, flatShading: true });
		const rockI = new THREE.InstancedMesh(rockG, rockM, Math.max(1, nrocks));
		rockI.castShadow = true;

		for (let i = 0; i < nrocks; i++) {
			const p = pts[i];
			if (!p) break;
			const s = 0.5 + rng() * 1.5;
			dummy.position.set(p.x, groundY + 0.05 * s, p.z);
			dummy.scale.set(s * (0.7 + rng() * 0.6), s * (0.5 + rng() * 0.5), s * (0.7 + rng() * 0.6));
			dummy.rotation.set(rng() * 0.4, rng() * 6.28, rng() * 0.4);
			dummy.updateMatrix();
			rockI.setMatrixAt(i, dummy.matrix);
		}
		rockI.instanceMatrix.needsUpdate = true;
		rockI.count = nrocks;
		scene.add(rockI);

		// --- Small lakes (circular water pools) ---
		const lakePts: { x: number; z: number }[] = [];
		for (let i = 0; i < 300 && lakePts.length < 4; i++) {
			const a = rng() * Math.PI * 2;
			const minR = Math.max(outerW, outerH) + 1;
			const r = minR + rng() * (topRadius - minR - 2);
			const x = Math.cos(a) * r, z = Math.sin(a) * r;
			if (!onRim(x, z)) continue;
			let ok = true;
			for (const lp of lakePts) if (Math.hypot(x - lp.x, z - lp.z) < 4) { ok = false; break; }
			if (ok) lakePts.push({ x, z });
		}
		for (const lp of lakePts) {
			const lr = 0.8 + rng() * 1.2;
			const lG = new THREE.CircleGeometry(lr, 16);
			lG.rotateX(-Math.PI / 2);
			const lM = new THREE.MeshStandardMaterial({
				color: 0x1a6e7a, transparent: true, opacity: 0.65,
				roughness: 0.1, metalness: 0.2, depthWrite: false,
			});
			const lMesh = new THREE.Mesh(lG, lM);
			lMesh.position.set(lp.x, groundY + 0.01, lp.z);
			lMesh.renderOrder = 3;
			scene.add(lMesh);
		}

		// --- Bushes (near rocks) ---
		const nbush = Math.min(30, nrocks);
		const bushG = new THREE.SphereGeometry(0.12, 5, 4);
		const bushM = new THREE.MeshStandardMaterial({ color: 0x3a7a28, roughness: 0.9 });
		const bushI = new THREE.InstancedMesh(bushG, bushM, Math.max(1, nbush));

		for (let i = 0; i < nbush; i++) {
			const tp = pts[i % nrocks];
			const off = 0.3 + rng() * 0.8;
			const ba = rng() * Math.PI * 2;
			const s = 0.8 + rng() * 1.2;
			dummy.position.set(tp.x + Math.cos(ba) * off, groundY + 0.06 * s, tp.z + Math.sin(ba) * off);
			dummy.scale.set(s * (0.8 + rng() * 0.4), s * (0.5 + rng() * 0.5), s * (0.8 + rng() * 0.4));
			dummy.rotation.y = rng() * 6.28;
			dummy.updateMatrix();
			bushI.setMatrixAt(i, dummy.matrix);
		}
		bushI.instanceMatrix.needsUpdate = true;
		bushI.count = nbush;
		scene.add(bushI);
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
			const s = 4; // pixels per cell
			for (let z = 0; z < rows; z++) {
				for (let x = 0; x < cols; x++) {
					ctx.fillStyle = MINIMAP_COLORS[g[z][x]] ?? '#666';
					ctx.fillRect(x * s, z * s, s, s);
				}
			}
			// Prediction overlay on minimap
			if (predictionOverlay?.visible && predictionAnalysis?.prediction && predictionAnalysis?.ground_truth) {
				const pa = predictionAnalysis;
				for (let pz = 0; pz < rows; pz++) {
					for (let px = 0; px < cols; px++) {
						if (!pa.prediction[pz]?.[px] || !pa.ground_truth[pz]?.[px]) continue;
						const pred = argmax(pa.prediction[pz][px]);
						const truth = argmax(pa.ground_truth[pz][px]);
						ctx.fillStyle = pred === truth ? 'rgba(25,210,60,0.55)' : 'rgba(220,40,30,0.55)';
						ctx.fillRect(px * s, pz * s, s, s);
					}
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

	function rebuildSceneryForNewGrid() {
		if (!pendingGrid || !terrainSystem || !worldGroup) {
			sceneryPhase = null;
			return;
		}
		// Save old scenery — keep it visible while new one builds
		const oldSceneryGroup = sceneryGroup;
		const oldCreatures = creatureSystem;
		const oldScatter = scatter;
		const oldWeather = weatherSystem;
		sceneryGroup = null;
		creatureSystem = null;
		scatter = null;
		weatherSystem = null;
		activeWaterfalls = [];

		// Move water/moat to worldGroup so they stay visible during rebuild
		if (oldSceneryGroup) {
			if (waterSystem?.mesh?.parent === oldSceneryGroup) {
				oldSceneryGroup.remove(waterSystem.mesh);
				worldGroup.add(waterSystem.mesh);
			}
			if (moatSystem?.group?.parent === oldSceneryGroup) {
				oldSceneryGroup.remove(moatSystem.group);
				worldGroup.add(moatSystem.group);
			}
		}

		// Build new scenery in background — old stays visible
		buildScene(terrainSystem).then(() => {
			if (sceneryGroup) sceneryGroup.position.y = 0;

			// NOW remove old scenery (new one is ready — no blink)
			if (oldSceneryGroup && worldGroup) {
				worldGroup.remove(oldSceneryGroup);
				setTimeout(() => disposeGroup(oldSceneryGroup), 50);
			}
			if (oldCreatures) oldCreatures.dispose();
			if (oldScatter) oldScatter.dispose();
			if (oldWeather) oldWeather.dispose();

			sceneryPhase = null;
			pendingGrid = null;
			pendingSettlements = null;
			if (sunLight) sunLight.shadow.needsUpdate = true;
			lastShadowUpdate = performance.now();
			lightProbeNeedsUpdate = true;
			if (flythroughActive && terrainSystem && flythrough) {
				flythrough.transitionToNewPath(grid, terrainSystem.getHeightAt);
			}
		});
	}

	function morphTerrainTo(newGrid: number[][]) {
		if (!terrainSystem) return;

		// Clean up any pending terrain from a previous interrupted morph
		if (pendingTerrainSystem) { pendingTerrainSystem.dispose(); pendingTerrainSystem = null; }

		// Store start state (fast — just copying existing arrays)
		const oldPos = terrainSystem.mesh.geometry.attributes.position;
		const oldCol = terrainSystem.mesh.geometry.attributes.color as THREE.BufferAttribute;
		terrainMorphStartHeights = new Float32Array(oldPos.count);
		for (let i = 0; i < oldPos.count; i++) terrainMorphStartHeights[i] = oldPos.getY(i);
		terrainMorphStartColors = new Float32Array(oldCol.array as Float32Array);

		// Keep scenery visible — NO sinking. Store pending grid for rebuild after morph.
		pendingGrid = newGrid;
		pendingSettlements = settlements;

		// Compute morph targets in Web Worker (off main thread — zero stutter)
		const worker = getTerrainWorker();
		worker.onmessage = (e: MessageEvent<{ heights: Float32Array; colors: Float32Array }>) => {
			terrainMorphTargetHeights = e.data.heights;
			terrainMorphTargetColors = e.data.colors;
			terrainMorphing = true;
			terrainMorphAlpha = 0;
			pendingMorphGrid = newGrid;
		};
		worker.onerror = () => {
			const morphData = computeTerrainMorphData(newGrid, season);
			terrainMorphTargetHeights = morphData.heights;
			terrainMorphTargetColors = morphData.colors;
			terrainMorphing = true;
			terrainMorphAlpha = 0;
			pendingMorphGrid = newGrid;
		};
		const plainGrid = newGrid.map(row => Array.from(row));
		worker.postMessage({ grid: plainGrid, season });
	}

	async function buildScene(reuseTerrainSystem?: TerrainSystem) {
		if (!grid?.length || !container) return;
		displayTime = timeOfDay;
		const gen = ++buildGeneration; // track this build — abort if superseded
		const isFirstBuild = !hemiLight; // true on very first build only
		// Animate terrain morph whenever there's an existing world to transition from
		const shouldTransition = !reuseTerrainSystem && worldGroup !== null;

		if (reuseTerrainSystem) {
			// Keep terrain mesh — only rebuild scenery objects
			terrainSystem = reuseTerrainSystem;
			// Clean non-terrain systems only
			if (creatureSystem) { creatureSystem.dispose(); creatureSystem = null; }
			if (scatter) { scatter.dispose(); scatter = null; }
			if (weatherSystem) { weatherSystem.dispose(); weatherSystem = null; }
		} else if (!shouldTransition) {
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
			if (nightSkySystem) keep.add(nightSkySystem.group);
			if (cloudSystem) for (const s of cloudSystem.sprites) keep.add(s);
			if (wildlifeSystem) keep.add(wildlifeSystem.group);
			if (fpSky) keep.add(fpSky);
			if (predictionOverlay) keep.add(predictionOverlay);
			if (gridOverlay) keep.add(gridOverlay);

			for (let i = scene.children.length - 1; i >= 0; i--) {
				const child = scene.children[i];
				if (!keep.has(child)) {
					scene.remove(child);
					if (child instanceof THREE.Mesh) { child.geometry.dispose(); }
				}
			}
		} else {
			// Transition: preserve old world for cross-fade morph
			// Clean up any in-progress transition before starting a new one
			if (prevWorldGroup) {
				disposeGroup(prevWorldGroup);
				scene.remove(prevWorldGroup);
			}
			if (prevTerrainSystem) { prevTerrainSystem.dispose(); prevTerrainSystem = null; }
			if (prevWaterSystem) { prevWaterSystem.dispose(); prevWaterSystem = null; }
			terrainMorphOldHeights = null;
			terrainMorphNewHeights = null;
			terrainMorphOldNormals = null;
			terrainMorphNewNormals = null;

			// Move current world to prev for cross-fade
			prevWorldGroup = worldGroup;
			prevTerrainSystem = terrainSystem;
			prevWaterSystem = waterSystem;
			// Null out so new build gets fresh systems
			terrainSystem = null;
			waterSystem = null;
			weatherSystem = null;
			moatSystem = null;
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

			// Night sky (stars + aurora) — persistent, added to real scene
			nightSkySystem = createNightSky();
			_realSceneAdd(nightSkySystem.group);

			scene.add = (...args: THREE.Object3D[]) => { newGroup.add(...args); return scene; };
		}

		applyDayNight(timeOfDay);

		// === Square island slab with earth layers ===
		buildIslandSlab(cols, rows);

		// Find clusters
		const clusters = findClusters(grid);

		// Compute road bezier paths for terrain baking
		const settlementClusters = clusters.filter(c => c.terrainType === TerrainCode.SETTLEMENT);
		const roadPaths = computeRoadPaths(settlementClusters, offsetX, offsetZ);
	
		// Build heightmap terrain — skip if reusing existing (morph transition)
		if (!reuseTerrainSystem) {
			terrainSystem = createTerrain(grid, { roads: roadPaths, season });
			scene.add(terrainSystem.mesh);
		}
		lastGridRows = grid.length;
		lastGridCols = grid[0].length;

		// 3D grid overlay — added to worldGroup directly (persists across scenery rebuilds)
		const wasGridVisible = gridOverlay?.visible ?? false;
		if (gridOverlay?.parent) gridOverlay.parent.remove(gridOverlay);
		gridOverlay = buildGridOverlay(grid, terrainSystem!.getHeightAt);
		gridOverlay.visible = wasGridVisible || showGrid;
		// Add to newGroup (goes to worldGroup), NOT sceneryGrp
		scene.add(gridOverlay);

		// All non-terrain objects go into sceneryGroup
		const sceneryGrp = new THREE.Group();
		scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };

		// Water surface + caustic floor — reuse if dimensions unchanged
		if (!waterSystem) {
			waterSystem = createWater(cols, rows, grid);
		}
		scene.add(waterSystem.mesh);

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }
		scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };

		// Moat water ring + river-waterfalls — reuse if dimensions unchanged
		if (!moatSystem) {
			moatSystem = createMoatAndWaterfalls(cols, rows);
		}
		scene.add(moatSystem.group);

		// Rim scenery (trees, rocks, lakes, bushes outside play grid)
		buildRimScenery(cols, rows);

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }
		// Re-assert redirect after potential concurrent build cancellations
		scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };

		// Add 3D cluster objects
		const dummy = new THREE.Object3D();

		// Process clusters in batches with frame yields to avoid stutter
		const forestClusters = clusters.filter(c => c.terrainType === TerrainCode.FOREST);
		// settlementClusters already computed above for road paths
		const portClusters2 = clusters.filter(c => c.terrainType === TerrainCode.PORT);
		const ruinClusters = clusters.filter(c => c.terrainType === TerrainCode.RUIN);

		// Forests — procedural low-poly trees (instanced)
		// Hash grid content so different maps get different tree styles
		let gridHash = 0;
		for (let y = 0; y < Math.min(5, rows); y++)
			for (let x = 0; x < Math.min(5, cols); x++)
				gridHash = (gridHash * 31 + (grid[y][x] ?? 0)) | 0;
		const mapTreeRng = mulberry32(gridHash ^ 999);
		const mapIsPine = mapTreeRng() > 0.5;
		const treeColors = getTreeColors(season, mapIsPine);

		// Count total tree slots (2 per cell for small clusters, 1 for large)
		let totalTreeSlots = 0;
		for (const c of forestClusters) {
			const perCell = c.cells.length <= 8 ? 2 : 1;
			totalTreeSlots += c.cells.length * perCell;
		}
		totalTreeSlots = Math.max(totalTreeSlots, 1);

		// Trunk instances
		const trunkGeo = new THREE.CylinderGeometry(0.04, 0.08, 1, 4);
		const trunkMat = new THREE.MeshStandardMaterial({ color: treeColors.trunk, roughness: 0.9, flatShading: true });
		const trunkInst = new THREE.InstancedMesh(trunkGeo, trunkMat, totalTreeSlots);
		trunkInst.castShadow = true;

		// Canopy instances (3 layers for pine, 1 for deciduous but we allocate 3 either way)
		const canopyGeos = mapIsPine
			? [new THREE.ConeGeometry(0.35, 0.5, 5), new THREE.ConeGeometry(0.28, 0.45, 5), new THREE.ConeGeometry(0.2, 0.4, 5)]
			: [new THREE.DodecahedronGeometry(0.3, 0), new THREE.DodecahedronGeometry(0.3, 0), new THREE.DodecahedronGeometry(0.3, 0)];
		const canopyInsts = canopyGeos.map((geo, i) => {
			const color = treeColors.canopy[i % treeColors.canopy.length];
			const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.85, flatShading: true });
			const inst = new THREE.InstancedMesh(geo, mat, totalTreeSlots);
			inst.castShadow = true;
			return inst;
		});

		// Apply wind sway to canopy materials
		if (windSystem) {
			for (const ci of canopyInsts) {
				applyWindSway(ci.material as THREE.MeshStandardMaterial, windSystem.uniforms);
			}
		}

		// Winter: add snow caps on canopy
		if (season === 'winter') {
			trunkMat.color.set(0x4a3a2a);
			if (!mapIsPine) {
				// Bare deciduous — very few leaves, mostly trunk
				for (const ci of canopyInsts) {
					(ci.material as THREE.MeshStandardMaterial).color.set(0x8a7a6a);
					(ci.material as THREE.MeshStandardMaterial).opacity = 0.4;
					(ci.material as THREE.MeshStandardMaterial).transparent = true;
				}
			}
		}

		let treeIdx = 0;
		for (const cluster of forestClusters) {
			treeIdx = addForest(cluster, offsetX, offsetZ, mapIsPine, trunkInst, canopyInsts, treeIdx, dummy);
		}
		trunkInst.count = treeIdx;
		trunkInst.instanceMatrix.needsUpdate = true;
		scene.add(trunkInst);
		for (const ci of canopyInsts) {
			ci.count = treeIdx;
			ci.instanceMatrix.needsUpdate = true;
			scene.add(ci);
		}

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }
		scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };

		// Settlements (GLB models — moderate cost)
		for (const cluster of settlementClusters) {
			addSettlement(cluster, offsetX, offsetZ, settlementClusters);
		}

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }
		scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };

		// Ports + ruins (lighter)
		for (const cluster of portClusters2) addPort(cluster, offsetX, offsetZ);
		for (const cluster of ruinClusters) addRuin(cluster, offsetX, offsetZ);

		// === Ambient systems (first build only — grid-independent, add to real scene) ===
		if (isFirstBuild) {
			scene.add = _realSceneAdd;
			cloudSystem = createCloudSystem(scene);
			scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };
		}

		await nextFrame();
		if (gen !== buildGeneration) { scene.add = _realSceneAdd; return; }
		scene.add = (...args: THREE.Object3D[]) => { sceneryGrp.add(...args); return scene; };

		// Roads are baked into terrain — no separate mesh needed

		// === Living world: creatures walking between settlements ===
		if (terrainSystem && settlementClusters.length >= 2) {
			createCreatures(scene, grid, terrainSystem.getHeightAt, settlementClusters).then(cs => {
				if (gen !== buildGeneration) { cs.dispose(); return; }
				creatureSystem = cs;
				sceneryGrp.add(cs.group);
			}).catch(() => {});
		}

		// === Environmental scatter (rocks, vegetation, dead trees) ===
		if (terrainSystem) {
			createScatter(grid, terrainSystem.getHeightAt, 800, windSystem?.uniforms).then(s => {
				if (gen !== buildGeneration) { s.dispose(); return; }
				scatter = s;
				sceneryGrp.add(s.group);
			}).catch(() => {});
		}

		// === Waterfalls: mountain cells adjacent to water ===
		const mountainClusters = clusters.filter(c => c.terrainType === TerrainCode.MOUNTAIN);
		for (const mc of mountainClusters) {
			addMountainWaterfalls(mc, offsetX, offsetZ);
		}

		// === Weather: rain, lightning, fog (uses mountain positions) ===
		const mountainPositions = mountainClusters.map(c => ({
			x: c.centerX + offsetX + 0.5,
			z: c.centerY + offsetZ + 0.5,
			radius: Math.sqrt(c.size) * 0.6
		}));
		const portClusters = clusters.filter(c => c.terrainType === TerrainCode.PORT);
		const lakePositions = portClusters.map(c => ({
			x: c.centerX + offsetX + 0.5,
			z: c.centerY + offsetZ + 0.5
		}));
		{
			const mapRadius = Math.max(cols, rows) / 2 + 2;
			weatherSystem = createWeatherSystem(scene, mountainPositions, lakePositions, mapRadius);
		}

		// === Wildlife: birds — recreate each build to ensure visibility ===
		if (wildlifeSystem) { wildlifeSystem.dispose(); wildlifeSystem = null; }
		{
			const savedAdd = scene.add;
			scene.add = _realSceneAdd;
			wildlifeSystem = createWildlifeSystem(scene);
			scene.add = savedAdd;
		}

		// Add scenery sub-group to world group
		sceneryGroup = sceneryGrp;

		// Restore scene.add
		scene.add = _realSceneAdd;

		if (reuseTerrainSystem) {
			// Scenery-only rebuild: terrain mesh stays, add new scenery to worldGroup
			newGroup.add(sceneryGrp);
			while (newGroup.children.length > 0) {
				worldGroup!.add(newGroup.children[0]);
			}
		} else {
			// Full build: add sceneryGroup to newGroup, add newGroup to scene
			newGroup.add(sceneryGrp);

			if (shouldTransition) {
				// Cross-fade: keep old world, fade out while new fades in
				setGroupOpacity(newGroup, 0);
				_realSceneAdd(newGroup);
				transitioning = true;
				transitionAlpha = 0;

				// Store old terrain heights + normals for morphing
				if (prevTerrainSystem) {
					const oldPos = prevTerrainSystem.mesh.geometry.attributes.position;
					terrainMorphOldHeights = new Float32Array(oldPos.count);
					for (let i = 0; i < oldPos.count; i++) terrainMorphOldHeights[i] = oldPos.getY(i);
					const oldNorm = prevTerrainSystem.mesh.geometry.attributes.normal;
					terrainMorphOldNormals = new Float32Array(oldNorm.count * 3);
					for (let i = 0; i < oldNorm.count; i++) {
						terrainMorphOldNormals[i * 3] = oldNorm.getX(i);
						terrainMorphOldNormals[i * 3 + 1] = oldNorm.getY(i);
						terrainMorphOldNormals[i * 3 + 2] = oldNorm.getZ(i);
					}
				}
				// Store new terrain heights + normals (start at flat, morph to actual)
				if (terrainSystem) {
					const newPos = terrainSystem.mesh.geometry.attributes.position;
					terrainMorphNewHeights = new Float32Array(newPos.count);
					for (let i = 0; i < newPos.count; i++) terrainMorphNewHeights[i] = newPos.getY(i);
					const newNorm = terrainSystem.mesh.geometry.attributes.normal;
					terrainMorphNewNormals = new Float32Array(newNorm.count * 3);
					for (let i = 0; i < newNorm.count; i++) {
						terrainMorphNewNormals[i * 3] = newNorm.getX(i);
						terrainMorphNewNormals[i * 3 + 1] = newNorm.getY(i);
						terrainMorphNewNormals[i * 3 + 2] = newNorm.getZ(i);
					}
					for (let i = 0; i < newPos.count; i++) newPos.setY(i, 0.0);
					newPos.needsUpdate = true;
				}
			} else {
				_realSceneAdd(newGroup);
			}
			worldGroup = newGroup;
		}

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

	/** Compute bezier road paths between settlements (MST). Returns path data for terrain baking. */
	function computeRoadPaths(settlements: Cluster[], ox: number, oz: number): import('./terrain').RoadPath[] {
		if (settlements.length < 2) return [];

		// MST edges
		const edges: { from: number; to: number; dist: number }[] = [];
		for (let i = 0; i < settlements.length; i++) {
			for (let j = i + 1; j < settlements.length; j++) {
				const dx = settlements[i].centerX - settlements[j].centerX;
				const dz = settlements[i].centerY - settlements[j].centerY;
				edges.push({ from: i, to: j, dist: Math.sqrt(dx * dx + dz * dz) });
			}
		}
		edges.sort((a, b) => a.dist - b.dist);
		const parent = settlements.map((_, i) => i);
		function find(x: number): number { return parent[x] === x ? x : (parent[x] = find(parent[x])); }
		const roadEdges: { from: number; to: number }[] = [];
		for (const e of edges) {
			const pf = find(e.from), pt = find(e.to);
			if (pf !== pt) { parent[pf] = pt; roadEdges.push(e); if (roadEdges.length >= settlements.length - 1) break; }
		}

		// Cubic bezier helper
		const bez = (a: number, b: number, c: number, d: number, t: number) => {
			const u = 1 - t;
			return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d;
		};

		const paths: import('./terrain').RoadPath[] = [];
		let seed = 42;
		const rng = () => { seed = (seed * 1664525 + 1013904223) & 0x7fffffff; return seed / 0x7fffffff; };

		for (const edge of roadEdges) {
			const a = settlements[edge.from], b = settlements[edge.to];
			const ax = a.centerX + ox + 0.5, az = a.centerY + oz + 0.5;
			const bx = b.centerX + ox + 0.5, bz = b.centerY + oz + 0.5;
			const dx = bx - ax, dz = bz - az;
			const len = Math.sqrt(dx * dx + dz * dz);
			const perpX = -dz / len, perpZ = dx / len;

			// Two bezier control points with random perpendicular offset
			const c1x = ax + dx * 0.33 + perpX * (rng() - 0.5) * len * 0.3;
			const c1z = az + dz * 0.33 + perpZ * (rng() - 0.5) * len * 0.3;
			const c2x = ax + dx * 0.66 + perpX * (rng() - 0.5) * len * 0.3;
			const c2z = az + dz * 0.66 + perpZ * (rng() - 0.5) * len * 0.3;

			// Sample bezier curve
			const segs = Math.max(8, Math.ceil(len * 3));
			const points: [number, number][] = [];
			for (let s = 0; s <= segs; s++) {
				const t = s / segs;
				points.push([bez(ax, c1x, c2x, bx, t), bez(az, c1z, c2z, bz, t)]);
			}

			paths.push({ points, width: 0.75 });
		}

		return paths;
	}

	// Seasonal low-poly tree colors
	function getTreeColors(s: Season, isPine: boolean): { trunk: number; canopy: number[] } {
		const trunk = isPine ? 0x5a4030 : 0x6b4a2a;
		switch (s) {
			case 'spring':
				return { trunk, canopy: isPine
					? [0x4a9e3a, 0x5cb84a, 0x3d8a2e]
					: [0x7dca5c, 0x9ddb7a, 0x5fb840] };
			case 'summer':
				return { trunk, canopy: isPine
					? [0x2d6b1e, 0x3a7a28, 0x1f5a14]
					: [0x4a9030, 0x5da03e, 0x3b8024] };
			case 'autumn':
				return { trunk, canopy: isPine
					? [0x3a6a22, 0x4a7a2e, 0x2e5a18]
					: [0xc86420, 0xd4882c, 0xb84818, 0xd4a030] };
			case 'winter':
				return { trunk, canopy: isPine
					? [0x2a5a1a, 0x325e22, 0x1e4e14]
					: [0x6a5a4a, 0x7a6a5a, 0x5a4a3a] };
		}
	}

	function addForest(
		cluster: Cluster, ox: number, oz: number, isPine: boolean,
		trunkInst: THREE.InstancedMesh, canopyInsts: THREE.InstancedMesh[],
		startIdx: number, dummy: THREE.Object3D
	): number {
		let idx = startIdx;
		// Dense clumps: 2 trees per cell for small clusters, 1 for large
		const treesPerCell = cluster.cells.length <= 8 ? 2 : 1;

		for (let ci = 0; ci < cluster.cells.length; ci++) {
			const cell = cluster.cells[ci];
			for (let ti = 0; ti < treesPerCell; ti++) {
			const rng = mulberry32(cellSeed(cell.x, cell.y, 200 + ti * 77));
			// Clump closer to cell center for dense look
			const spread = 0.15 + rng() * 0.7;
			const px = cell.x + ox + spread;
			const pz = cell.y + oz + 0.15 + rng() * 0.7;
			const groundH = gY(px, pz);
			const h = 0.5 + rng() * 0.9; // bigger trees with good variance
			const w = 0.5 + rng() * 0.7; // wider variance

			// Trunk
			dummy.position.set(px, groundH + h * 0.3, pz);
			dummy.scale.set(w * 0.3, h * 0.6, w * 0.3);
			dummy.rotation.set(0, rng() * Math.PI * 2, 0);
			dummy.updateMatrix();
			trunkInst.setMatrixAt(idx, dummy.matrix);

			if (isPine) {
				// Pine: 3 stacked cones, progressively smaller
				const layers = [
					{ yOff: 0.4, scaleW: 1.0, scaleH: 0.8 },
					{ yOff: 0.6, scaleW: 0.75, scaleH: 0.7 },
					{ yOff: 0.78, scaleW: 0.5, scaleH: 0.6 },
				];
				for (let li = 0; li < layers.length; li++) {
					const l = layers[li];
					dummy.position.set(px, groundH + h * l.yOff, pz);
					dummy.scale.set(w * l.scaleW, h * l.scaleH, w * l.scaleW);
					dummy.updateMatrix();
					canopyInsts[li].setMatrixAt(idx, dummy.matrix);
				}
			} else {
				// Deciduous: single blocky sphere/dodecahedron
				dummy.position.set(px, groundH + h * 0.65, pz);
				const spread = w * (0.8 + rng() * 0.4);
				dummy.scale.set(spread, h * 0.7, spread);
				dummy.updateMatrix();
				for (const ci2 of canopyInsts) {
					ci2.setMatrixAt(idx, dummy.matrix);
				}
			}
			idx++;
			} // end treesPerCell loop
		}
		return idx;
	}

	function addMountain(cluster: Cluster, ox: number, oz: number) {
		const rng = mulberry32(cellSeed(Math.round(cluster.centerX), Math.round(cluster.centerY), 100));
		const cx = cluster.centerX + ox + 0.5;
		const cz = cluster.centerY + oz + 0.5;
		const groundH = gY(cx, cz);

		// GLB mountain models placed on terrain surface
		if (cluster.size >= 7) {
			const m = placeModel('mountains', new THREE.Vector3(cx, groundH, cz), rng() * Math.PI * 2, 0.55);
			if (m) { scene.add(m); }
		} else if (cluster.size >= 4) {
			const m = placeModel('mountainGroup', new THREE.Vector3(cx, groundH, cz), rng() * Math.PI * 2, 0.60);
			if (m) { scene.add(m); }
		} else if (cluster.size >= 2) {
			const m = placeModel('mountain', new THREE.Vector3(cx, groundH, cz), rng() * Math.PI * 2, 0.75);
			if (m) { scene.add(m); }
		}

		// Snow cap on peak
		if (cluster.size >= 2 && groundH > 1.0) {
			const snowR = cluster.size >= 7 ? 0.7 : cluster.size >= 4 ? 0.5 : 0.35;
			const snowH = cluster.size >= 7 ? 0.5 : 0.35;
			const snow = new THREE.Mesh(
				new THREE.ConeGeometry(snowR, snowH, 8),
				new THREE.MeshStandardMaterial({ color: 0xf0ece0, roughness: 0.4 })
			);
			snow.position.set(cx, groundH + snowH * 0.3, cz);
			scene.add(snow);
		}

		// Boulders — use GLB rocks if available
		if (cluster.size >= 3) {
			for (let i = 0; i < Math.min(cluster.size, 8); i++) {
				const cell = cluster.cells[Math.min(i, cluster.cells.length - 1)];
				const bx = cell.x + ox + 0.2 + rng() * 0.6;
				const bz = cell.y + oz + 0.2 + rng() * 0.6;
				const bH = gY(bx, bz);
				const rockModel = placeModel(rng() > 0.5 ? 'rock' : 'rocks', new THREE.Vector3(bx, bH, bz), rng() * Math.PI * 2, 0.20 + rng() * 0.25);
				if (rockModel) { scene.add(rockModel); }
				else {
					const bs = 0.1 + rng() * 0.25;
					const boulder = new THREE.Mesh(
						new THREE.DodecahedronGeometry(bs, 1),
						new THREE.MeshStandardMaterial({ color: 0x8a8878, roughness: 0.9, flatShading: true })
					);
					boulder.position.set(bx, bH + bs * 0.5, bz);
					boulder.rotation.set(rng() * 3, rng() * 3, rng() * 3);
					boulder.castShadow = true;
					scene.add(boulder);
				}
			}
		}
	}

	/** Add particle waterfalls that flow from mountain peak down the ridge into adjacent water */
	function addMountainWaterfalls(cluster: Cluster, ox: number, oz: number) {
		const rows = grid.length, cols = grid[0].length;
		const WATER_CODES = new Set([10, 2]); // ocean, port
		const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
		const added = new Set<string>(); // avoid duplicate waterfalls per edge

		// Find cluster peak (highest point)
		const peakX = cluster.centerX + ox + 0.5;
		const peakZ = cluster.centerY + oz + 0.5;
		const peakH = gY(peakX, peakZ);

		for (const cell of cluster.cells) {
			for (const [ddx, ddz] of dirs) {
				const nx = cell.x + ddx, nz = cell.y + ddz;
				if (nx < 0 || nx >= cols || nz < 0 || nz >= rows) continue;
				if (!WATER_CODES.has(grid[nz][nx])) continue;

				const key = `${Math.min(cell.x, nx)},${Math.min(cell.y, nz)},${ddx},${ddz}`;
				if (added.has(key)) continue;
				added.add(key);

				const edgeX = cell.x + ox + 0.5;
				const edgeZ = cell.y + oz + 0.5;
				const watX = nx + ox + 0.5;
				const watZ = nz + oz + 0.5;
				const edgeH = gY(edgeX, edgeZ);
				const watH = gY(watX, watZ);
				if (edgeH - watH < 0.15) continue;

				// Flow path: peak → edge → water (sampled along the ridge)
				const count = 60;
				const geo = new THREE.BufferGeometry();
				const positions = new Float32Array(count * 3);
				const rng = mulberry32(cellSeed(cell.x, cell.y, 500 + ddx * 10 + ddz));

				for (let i = 0; i < count; i++) {
					const t = rng(); // 0=peak, 1=water
					// Path: lerp from peak → edge (t=0..0.6) then edge → water (t=0.6..1)
					let px: number, pz: number, py: number;
					if (t < 0.6) {
						const s = t / 0.6;
						px = peakX + (edgeX - peakX) * s + (rng() - 0.5) * 0.25;
						pz = peakZ + (edgeZ - peakZ) * s + (rng() - 0.5) * 0.25;
						py = peakH + (edgeH - peakH) * s;
					} else {
						const s = (t - 0.6) / 0.4;
						px = edgeX + (watX - edgeX) * s + (rng() - 0.5) * 0.2;
						pz = edgeZ + (watZ - edgeZ) * s + (rng() - 0.5) * 0.2;
						py = edgeH + (watH - edgeH) * s;
					}
					positions[i * 3] = px;
					positions[i * 3 + 1] = py + 0.05;
					positions[i * 3 + 2] = pz;
				}
				geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
				const mat = new THREE.PointsMaterial({
					color: 0x88ccee, size: 0.06,
					transparent: true, opacity: 0.5, depthWrite: false,
				});
				const points = new THREE.Points(geo, mat);
				scene.add(points);
				activeWaterfalls.push({
					points, topH: peakH, botH: watH,
					midX: (peakX + watX) / 2, midZ: (peakZ + watZ) / 2,
					// Store flow path for animation
					peakX, peakZ, peakH, edgeX, edgeZ, edgeH, watX, watZ, watH
				});

				// Foam splash at water entry
				const foamGeo = new THREE.CircleGeometry(0.25, 6);
				foamGeo.rotateX(-Math.PI / 2);
				const foamMat = new THREE.MeshStandardMaterial({
					color: 0xddeeff, transparent: true, opacity: 0.4,
					depthWrite: false, roughness: 0.1,
				});
				const foam = new THREE.Mesh(foamGeo, foamMat);
				foam.position.set((edgeX + watX) / 2, Math.max(watH, WATER_LEVEL) + 0.02, (edgeZ + watZ) / 2);
				scene.add(foam);
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
			const groundH = gY(px, pz);

			if (isTown) {
				if (isCenter && isLarge) {
					// Town center: market or fortress
					const special = townSpecials[Math.floor(rng() * townSpecials.length)];
					const m = placeModel(special, new THREE.Vector3(px, groundH, pz), rot, 0.825);
					if (m) scene.add(m);
				} else if (isCenter && isMedium) {
					const m = placeModel('pfMarket', new THREE.Vector3(px, groundH, pz), rot, 0.72);
					if (m) scene.add(m);
				} else {
					// Mix of houses and farms
					const pick = townBuildings[Math.floor(rng() * townBuildings.length)];
					const scale = pick.startsWith('pf') ? 0.57 + rng() * 0.15 : 0.75 + rng() * 0.18;
					const m = placeModel(pick, new THREE.Vector3(px + jx, groundH, pz + jz), rot, scale);
					if (m) scene.add(m);
					else {
						// Procedural fallback
						const wallMat = new THREE.MeshStandardMaterial({ color: 0x8a6a40, roughness: 0.8 });
						const hutS = 0.675 + rng() * 0.225;
						const hut = new THREE.Mesh(new THREE.BoxGeometry(hutS, 0.48, hutS), wallMat);
						hut.position.set(px + jx, groundH + 0.24, pz + jz);
						hut.castShadow = true;
						scene.add(hut);
						const roof = new THREE.Mesh(
							new THREE.ConeGeometry(hutS * 0.8, 0.36, 4),
							new THREE.MeshStandardMaterial({ color: 0x7a6a52, roughness: 0.9 })
						);
						roof.position.copy(hut.position);
						roof.position.y += 0.42;
						roof.rotation.y = Math.PI / 4;
						scene.add(roof);
					}
				}
			} else {
				// Village: mostly huts, small farms, crops
				if (isCenter && isMedium) {
					// Central building for medium village
					const m = placeModel('house', new THREE.Vector3(px, groundH, pz), rot, 0.825);
					if (m) scene.add(m);
				} else {
					const pick = villageBuildings[Math.floor(rng() * villageBuildings.length)];
					const scale = pick.startsWith('pf') ? 0.54 + rng() * 0.12 : 0.42 + rng() * 0.15;
					const m = placeModel(pick, new THREE.Vector3(px + jx, groundH, pz + jz), rot, scale);
					if (m) scene.add(m);
					else {
						const wallMat = new THREE.MeshStandardMaterial({ color: 0x8a6a40, roughness: 0.8 });
						const hutS = 0.60 + rng() * 0.18;
						const hut = new THREE.Mesh(new THREE.BoxGeometry(hutS, 0.42, hutS), wallMat);
						hut.position.set(px + jx, groundH + 0.21, pz + jz);
						hut.castShadow = true;
						scene.add(hut);
						const roof = new THREE.Mesh(
							new THREE.ConeGeometry(hutS * 0.8, 0.24, 4),
							new THREE.MeshStandardMaterial({ color: 0x7a6a52, roughness: 0.9 })
						);
						roof.position.copy(hut.position);
						roof.position.y += 0.285;
						roof.rotation.y = Math.PI / 4;
						scene.add(roof);
					}
				}
			}

			// Fire lights disabled for performance

			if (i === 0) {
				settlementPositions.push(new THREE.Vector3(px, groundH + 0.1, pz));
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
				const wH = gY(wx, wz);
				const wm = placeModel('woodenWall', new THREE.Vector3(wx, wH, wz), angle + Math.PI / 2, 0.33);
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
				const wH = gY(wx, wz);
				const wm = placeModel('woodenWall', new THREE.Vector3(wx, wH, wz), angle + Math.PI / 2, 0.27);
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
				const wH = gY(wx, wz);
				const wm = placeModel('woodenWall', new THREE.Vector3(wx, wH, wz), angle + Math.PI / 2, 0.30);
				if (wm) { scene.add(wm); }
				else {
					const fenceMat = new THREE.MeshStandardMaterial({ color: 0x6a5030, roughness: 0.9 });
					const postH = 0.33;
					const post = new THREE.Mesh(new THREE.BoxGeometry(0.06, postH, 0.06), fenceMat);
					post.position.set(wx, wH + postH / 2, wz);
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
			const dm = placeModel('dock', new THREE.Vector3(px, 0.05, pz), rng() * Math.PI * 2, undefined);
			if (dm) { dm.scale.multiplyScalar(1.5); scene.add(dm); }
			else {
				const dockMat = new THREE.MeshStandardMaterial({ color: 0x6a5038, roughness: 0.9 });
				const dock = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.06, 0.6), dockMat);
				dock.position.set(px, 0.03, pz);
				scene.add(dock);
			}
		}

		// Port building / ship
		if (cluster.size >= 2) {
			const cx = cluster.centerX + ox + 0.5;
			const cz = cluster.centerY + oz + 0.5;
			const pm = placeModel('port', new THREE.Vector3(cx, 0.05, cz + 0.8), rng() * Math.PI * 2, 0.375);
			if (pm) { scene.add(pm); }
			else {
				// Procedural ship fallback
				const hullMat = new THREE.MeshStandardMaterial({ color: 0x5a3a20, roughness: 0.85 });
				const hull = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.18, 0.33), hullMat);
				hull.position.set(cx, -0.02, cz + 1.2);
				hull.castShadow = true;
				scene.add(hull);
				const mast = new THREE.Mesh(
					new THREE.CylinderGeometry(0.03, 0.03, 0.75, 4),
					new THREE.MeshStandardMaterial({ color: 0x6a5038 })
				);
				mast.position.set(cx, 0.375, cz + 1.2);
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
				const h = 0.3 + rng() * 0.525;
				const w = 0.09 + rng() * 0.09;
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
				const rs = 0.06 + rng() * 0.09;
				const rubbleGeo = new THREE.DodecahedronGeometry(rs, 0);
				const rubble = new THREE.Mesh(rubbleGeo, rubbleMat);
				const rx = cell.x + ox + 0.2 + rng() * 0.6;
				const rz = cell.y + oz + 0.2 + rng() * 0.6;
				rubble.position.set(rx, gY(rx, rz) + rs, rz);
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

			// Throttle shadow re-renders — orbit=5s, FP/flythrough=3s
			const now = performance.now();
			const shadowInterval = fpMode || flythroughActive ? 3000 : 5000;
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

		// Smooth time transitions
		if (skyFrozen) {
			// During scene transition: hold sky at frozen time
			displayTime = skyFrozenTime;
		} else if (skyTransitionActive) {
			// After scene transition: smooth cubic ease-in-out to new time
			skyTransitionAlpha = Math.min(1.0, skyTransitionAlpha + dt / SKY_TRANSITION_DURATION);
			const t = skyTransitionAlpha;
			const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
			displayTime = skyTransitionFrom + (skyTransitionTo - skyTransitionFrom) * eased;
			if (skyTransitionAlpha >= 1.0) {
				skyTransitionActive = false;
				displayTime = skyTransitionTo;
			}
		} else {
			// Normal: lerp toward target
			const timeDiff = timeOfDay - displayTime;
			const timeLerpSpeed = flythroughActive ? 1.2 : 5;
			if (Math.abs(timeDiff) > 0.005) {
				displayTime += timeDiff * Math.min(1, dt * timeLerpSpeed);
			} else {
				displayTime = timeOfDay;
			}
		}
		const dn = applyDayNight(displayTime);

		// Update celestials every frame (world-space, large orbit radius)
		if (celestialSystem && cachedDN) {
			celestialSystem.update(cachedDN.sunPosition, cachedDN.moonPosition, cachedDN.nightFade);
		}
		if (nightSkySystem && cachedDN) {
			nightSkySystem.update(dt, cachedDN.nightFade);
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

		// Wind drives clouds and vegetation sway (uniforms auto-propagate to shaders)
		if (windSystem) windSystem.update(dt);
		if (cloudSystem) cloudSystem.update(dt, dn.skyColor, windSystem?.direction, windSystem?.strength, windSystem?.stormIntensity);
		if (weatherSystem) weatherSystem.update(dt, displayTime, windSystem?.stormIntensity, season);
		// Animate mountain waterfalls — particles flow along ridge then fall
		for (const wf of activeWaterfalls) {
			const pos = wf.points.geometry.attributes.position as THREE.BufferAttribute;
			const arr = pos.array as Float32Array;
			const speed = dt * 1.2;
			for (let i = 0; i < pos.count; i++) {
				// Move particle toward water along the path
				const px = arr[i * 3], py = arr[i * 3 + 1], pz = arr[i * 3 + 2];
				// Direction: peak → edge → water (use distance to edge as phase)
				const dEdge = Math.sqrt((px - wf.edgeX) ** 2 + (pz - wf.edgeZ) ** 2);
				const dWat = Math.sqrt((px - wf.watX) ** 2 + (pz - wf.watZ) ** 2);
				let tx: number, tz: number, ty: number;
				if (dEdge > 0.3) {
					// Moving toward edge (down the ridge)
					tx = wf.edgeX; tz = wf.edgeZ; ty = wf.edgeH;
				} else {
					// Past edge, falling toward water
					tx = wf.watX; tz = wf.watZ; ty = wf.watH;
				}
				const dx = tx - px, dz = tz - pz, dl = Math.sqrt(dx * dx + dz * dz) || 1;
				arr[i * 3] += (dx / dl) * speed + (Math.random() - 0.5) * 0.01;
				arr[i * 3 + 1] -= speed * (wf.peakH - wf.watH) * 0.4; // gravity along height diff
				arr[i * 3 + 2] += (dz / dl) * speed + (Math.random() - 0.5) * 0.01;

				// Recycle when past water or below water level
				if (arr[i * 3 + 1] < wf.watH - 0.1 || dWat < 0.2) {
					// Respawn near peak with slight randomness
					arr[i * 3]     = wf.peakX + (Math.random() - 0.5) * 0.3;
					arr[i * 3 + 1] = wf.peakH + 0.05;
					arr[i * 3 + 2] = wf.peakZ + (Math.random() - 0.5) * 0.3;
				}
			}
			pos.needsUpdate = true;
		}
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
		if (moatSystem) moatSystem.update(dt, cachedDN?.sunPosition);
		if (wildlifeSystem) wildlifeSystem.update(dt, displayTime);
		if (creatureSystem) creatureSystem.update(dt, displayTime, camera);

		// Cross-fade transition: morph terrain + fade objects
		if (transitioning) {
			transitionAlpha = Math.min(1.0, transitionAlpha + dt / TRANSITION_DURATION);
			const t = transitionAlpha;
			const eased = t * t * (3 - 2 * t); // smoothstep

			// Fade new world in, old world out
			setGroupOpacity(worldGroup, eased);
			if (prevWorldGroup) setGroupOpacity(prevWorldGroup, 1.0 - eased);

			// Morph old terrain heights → flat, lerp normals → up
			if (prevTerrainSystem && terrainMorphOldHeights) {
				const pos = prevTerrainSystem.mesh.geometry.attributes.position;
				for (let i = 0; i < pos.count; i++) {
					pos.setY(i, terrainMorphOldHeights[i] * (1 - eased));
				}
				pos.needsUpdate = true;
				if (terrainMorphOldNormals) {
					const norms = prevTerrainSystem.mesh.geometry.attributes.normal;
					const arr = norms.array as Float32Array;
					const src = terrainMorphOldNormals;
					const inv = 1 - eased;
					for (let i = 0, len = norms.count; i < len; i++) {
						const i3 = i * 3;
						arr[i3]     = src[i3] * inv;
						arr[i3 + 1] = src[i3 + 1] * inv + eased;
						arr[i3 + 2] = src[i3 + 2] * inv;
					}
					(norms as THREE.BufferAttribute).needsUpdate = true;
				}
			}

			// Morph new terrain heights flat → actual, lerp normals up → actual
			if (terrainSystem && terrainMorphNewHeights) {
				const pos = terrainSystem.mesh.geometry.attributes.position;
				for (let i = 0; i < pos.count; i++) {
					pos.setY(i, terrainMorphNewHeights[i] * eased);
				}
				pos.needsUpdate = true;
				if (terrainMorphNewNormals) {
					const norms = terrainSystem.mesh.geometry.attributes.normal;
					const arr = norms.array as Float32Array;
					const src = terrainMorphNewNormals;
					const inv = 1 - eased;
					for (let i = 0, len = norms.count; i < len; i++) {
						const i3 = i * 3;
						arr[i3]     = src[i3] * eased;
						arr[i3 + 1] = inv + src[i3 + 1] * eased;
						arr[i3 + 2] = src[i3 + 2] * eased;
					}
					(norms as THREE.BufferAttribute).needsUpdate = true;
				}
			}

			if (transitionAlpha >= 1.0) {
				transitioning = false;
				if (prevWorldGroup) {
					disposeGroup(prevWorldGroup);
					scene.remove(prevWorldGroup);
					prevWorldGroup = null;
				}
				if (prevTerrainSystem) { prevTerrainSystem.dispose(); prevTerrainSystem = null; }
				if (prevWaterSystem) { prevWaterSystem.dispose(); prevWaterSystem = null; }
				terrainMorphOldHeights = null;
				terrainMorphNewHeights = null;
				terrainMorphOldNormals = null;
				terrainMorphNewNormals = null;
				setGroupOpacity(worldGroup, 1.0);
			}
		}

		// Terrain-only height morph (no scene rebuild)
		if (terrainMorphing && terrainSystem && terrainMorphStartHeights && terrainMorphTargetHeights) {
			terrainMorphAlpha = Math.min(1.0, terrainMorphAlpha + dt / TERRAIN_MORPH_DURATION);
			const t = terrainMorphAlpha;
			const eased = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

			// Morph heights — direct Float32Array access (no getY/setY overhead)
			const posArr = (terrainSystem.mesh.geometry.attributes.position as THREE.BufferAttribute).array as Float32Array;
			const sh = terrainMorphStartHeights, th = terrainMorphTargetHeights;
			const stride = 3; // x,y,z per vertex
			for (let i = 0, len = sh.length; i < len; i++) {
				posArr[i * stride + 1] = sh[i] + (th[i] - sh[i]) * eased;
			}
			(terrainSystem.mesh.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;

			// Smoothly lerp vertex colors during morph
			if (terrainMorphStartColors && terrainMorphTargetColors) {
				const colAttr = terrainSystem.mesh.geometry.attributes.color as THREE.BufferAttribute;
				const colArr = colAttr.array as Float32Array;
				const sc = terrainMorphStartColors, tc = terrainMorphTargetColors;
				for (let i = 0, len = sc.length; i < len; i++) {
					colArr[i] = sc[i] + (tc[i] - sc[i]) * eased;
				}
				colAttr.needsUpdate = true;
			}

			if (terrainMorphAlpha >= 1.0) {
				terrainMorphing = false;

				// Final snap (ensure exact target colors)
				if (terrainMorphStartColors && terrainMorphTargetColors) {
					const colAttr = terrainSystem.mesh.geometry.attributes.color as THREE.BufferAttribute;
					(colAttr.array as Float32Array).set(terrainMorphTargetColors);
					colAttr.needsUpdate = true;
				}
				terrainSystem.mesh.geometry.computeVertexNormals();

				terrainMorphStartHeights = null;
				terrainMorphTargetHeights = null;
				terrainMorphStartColors = null;
				terrainMorphTargetColors = null;

				// Keep existing terrain mesh — just update getHeightAt for new grid
				pendingMorphGrid = null;
				if (terrainSystem && pendingGrid) {
					updateTerrainHeightFn(terrainSystem, pendingGrid);
					lastGridRows = pendingGrid.length;
					lastGridCols = pendingGrid[0].length;
				}
				// Rebuild scenery (trees, settlements, etc.) on the existing terrain
				if (pendingGrid) {
					rebuildSceneryForNewGrid();
				}
			}
		}

		// Scenery sink/rise animation
		if (sceneryPhase === 'sinking' && sceneryGroup) {
			sceneryAlpha = Math.min(1.0, sceneryAlpha + dt / SCENERY_SINK_DURATION);
			const t = sceneryAlpha;
			const eased = t * t * (3 - 2 * t); // smoothstep
			sceneryGroup.position.y = -SCENERY_SINK_DEPTH * eased;

			if (sceneryAlpha >= 1.0) {
				// Scenery fully sunk — rebuild immediately (don't wait for terrain morph)
				rebuildSceneryForNewGrid();
			}
		}
		if (sceneryPhase === 'rising' && sceneryGroup) {
			sceneryAlpha = Math.min(1.0, sceneryAlpha + dt / SCENERY_RISE_DURATION);
			const t = sceneryAlpha;
			const eased = t * t * (3 - 2 * t); // smoothstep
			sceneryGroup.position.y = -SCENERY_SINK_DEPTH * (1 - eased);

			if (sceneryAlpha >= 1.0) {
				sceneryGroup.position.y = 0;
				sceneryPhase = null;
				pendingGrid = null;
				pendingSettlements = null;
				// Force shadow update after new scenery is in place
				if (sunLight) sunLight.shadow.needsUpdate = true;
				lastShadowUpdate = performance.now();
				lightProbeNeedsUpdate = true;
				// Update flythrough path for new terrain if active
				if (flythroughActive && terrainSystem && flythrough) {
					flythrough.transitionToNewPath(grid, terrainSystem.getHeightAt);
				}
				// Start smooth sky transition now that scene is ready
				if (skyFrozen) {
					skyFrozen = false;
					skyTransitionActive = true;
					skyTransitionAlpha = 0;
					skyTransitionFrom = skyFrozenTime;
					skyTransitionTo = timeOfDay;
				}
			}
		}

		// Settlement fires disabled for performance

		// First-person mode: update FP controller or flythrough

		if (fpMode) {
			// Flythrough overrides FP controls
			if (flythroughActive && flythrough) {
				flythrough.update(camera, dt);
			} else if (fpController) {
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
			// Idle cinema mode — random still shots with handheld sway
			const idleTime = now - lastInteraction;
			if (!freezeCamera && idleTime > 10 && controls) {
				// Initialize cinema mode on first idle frame
				if (!idleCinemaActive) {
					idleCinemaActive = true;
					idleShotTimer = 0;
					idleShotPhase = 'transition';
					idleTransitionProgress = 0;
					idleHandheldTime = 0;
					// Current camera position as "from"
					idleCamFrom = {
						px: camera.position.x, py: camera.position.y, pz: camera.position.z,
						lx: 0, ly: gY(0, 0), lz: 0
					};
					idleCamTo = generateIdleShot();
					idleCamCurrent = { ...idleCamFrom };
				}

				idleShotTimer += dt;
				idleHandheldTime += dt;

				if (idleShotPhase === 'transition') {
					idleTransitionProgress = Math.min(1, idleTransitionProgress + dt / IDLE_TRANSITION_DURATION);
					const t = smoothstep(idleTransitionProgress);
					idleCamCurrent.px = idleCamFrom.px + (idleCamTo.px - idleCamFrom.px) * t;
					idleCamCurrent.py = idleCamFrom.py + (idleCamTo.py - idleCamFrom.py) * t;
					idleCamCurrent.pz = idleCamFrom.pz + (idleCamTo.pz - idleCamFrom.pz) * t;
					idleCamCurrent.lx = idleCamFrom.lx + (idleCamTo.lx - idleCamFrom.lx) * t;
					idleCamCurrent.ly = idleCamFrom.ly + (idleCamTo.ly - idleCamFrom.ly) * t;
					idleCamCurrent.lz = idleCamFrom.lz + (idleCamTo.lz - idleCamFrom.lz) * t;
					if (idleTransitionProgress >= 1) {
						idleShotPhase = 'hold';
						idleShotTimer = 0;
					}
				} else {
					// Hold phase — check if it's time to pick a new shot
					if (idleShotTimer >= IDLE_SHOT_DURATION) {
						idleCamFrom = { ...idleCamTo };
						idleCamTo = generateIdleShot();
						idleShotPhase = 'transition';
						idleTransitionProgress = 0;
						idleShotTimer = 0;
					}
				}

				// Handheld sway — subtle noise-based camera shake
				const swayX = Math.sin(idleHandheldTime * 0.7) * 0.03 + Math.sin(idleHandheldTime * 1.3) * 0.015;
				const swayY = Math.sin(idleHandheldTime * 0.5 + 1) * 0.02 + Math.cos(idleHandheldTime * 1.1) * 0.01;
				const swayZ = Math.cos(idleHandheldTime * 0.6 + 2) * 0.03 + Math.sin(idleHandheldTime * 1.5) * 0.015;

				camera.position.set(
					idleCamCurrent.px + swayX,
					idleCamCurrent.py + swayY,
					idleCamCurrent.pz + swayZ
				);
				camera.lookAt(
					idleCamCurrent.lx + swayX * 0.3,
					idleCamCurrent.ly + swayY * 0.2,
					idleCamCurrent.lz + swayZ * 0.3
				);
			} else {
				if (idleCinemaActive) {
					idleCinemaActive = false;
					autoRotateAngle = Math.atan2(camera.position.z, camera.position.x);
				}
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

		// Global wind system
		windSystem = createWind();

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
			idleCinemaActive = false;
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
		if (pendingTerrainSystem) { pendingTerrainSystem.dispose(); pendingTerrainSystem = null; }
		if (prevWaterSystem) { prevWaterSystem.dispose(); prevWaterSystem = null; }
		if (waterSystem) { waterSystem.dispose(); waterSystem = null; }
		if (moatSystem) { moatSystem.dispose(); moatSystem = null; }
		if (cloudSystem) { cloudSystem.dispose(); cloudSystem = null; }
		if (waterfallSystem) { waterfallSystem.dispose(); waterfallSystem = null; }
		if (celestialSystem) { celestialSystem.dispose(); celestialSystem = null; }
		if (nightSkySystem) { nightSkySystem.dispose(); nightSkySystem = null; }
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
		if (terrainWorker) { terrainWorker.terminate(); terrainWorker = null; }
		if (renderer) renderer.dispose();
		window.removeEventListener('resize', handleResize);
		window.removeEventListener('keydown', onGlobalKeyDown);
	});

	let lastSeason: Season | undefined = undefined;
	$effect(() => {
		grid; settlements; season;
		// Invalidate stale prediction overlay on seed/round/season change
		const wasShowingPrediction = predictionOverlay?.visible ?? false;
		if (predictionOverlay) {
			scene?.remove(predictionOverlay);
			predictionOverlay = null;
			predictionAnalysis = null;
		}
		if (scene && grid?.length) {
			lastSeason = season;
			const canMorph = terrainSystem
				&& grid.length === lastGridRows
				&& grid[0].length === lastGridCols
				&& !transitioning
				&& !terrainMorphing;
			if (canMorph) {
				// Morph handles both grid changes AND season changes (worker uses season)
				morphTerrainTo(grid);
			} else {
				buildScene();
			}
		}
		// Re-fetch prediction overlay if it was visible
		if (wasShowingPrediction) togglePredictionOverlay();
	});

	// Sync overlay visibility with props — read prop first to ensure dependency tracking
	$effect(() => {
		const vis = showGrid;
		if (gridOverlay) gridOverlay.visible = vis;
	});
	$effect(() => {
		const vis = showTerrain;
		if (terrainSystem) terrainSystem.mesh.visible = vis;
	});
	$effect(() => {
		const vis = showPrediction;
		if (vis && !predictionOverlay) {
			togglePredictionOverlay();
		} else if (!vis && predictionOverlay) {
			predictionOverlay.visible = false;
		} else if (vis && predictionOverlay) {
			predictionOverlay.visible = true;
		}
	});
</script>

<div class="relative w-full h-full">
	<div bind:this={container} class="w-full h-full rounded-lg overflow-hidden border border-cyber-border"></div>

	<!-- Performance stats overlay (hidden during flythrough) -->
	{#if fpsDisplay && !flythroughActive}
		<div class="absolute bottom-2 left-2 z-10 px-2 py-1 text-[10px] font-mono text-cyber-muted bg-cyber-bg/70 rounded backdrop-blur-sm pointer-events-none">
			{fpsDisplay} · {drawCallsDisplay}
		</div>
	{/if}

	<!-- Minimap (all modes) -->
	{#if grid && !flythroughLoading}
		{@const terrainColors = { 0: '#2a2a3a', 1: '#d4a843', 2: '#4fc3f7', 3: '#e53935', 4: '#2e7d32', 5: '#8a8a8a', 10: '#1565c0', 11: '#558b2f' } as Record<number, string>}
		{@const terrainNames = new Map([[0,'Sand'],[1,'Town'],[2,'Port'],[3,'Ruin'],[4,'Forest'],[5,'Mountain'],[10,'Ocean'],[11,'Plains']])}
		{@const total = grid.length * grid[0].length}
		{@const terrainCounts = (() => {
			const c: [number, number][] = [];
			const m = new Map<number, number>();
			for (const row of grid) for (const cell of row) m.set(cell, (m.get(cell) || 0) + 1);
			m.forEach((v, k) => c.push([k, v]));
			return c.sort((a, b) => b[1] - a[1]);
		})()}
		<div class="absolute bottom-4 right-4 z-10 pointer-events-none flex gap-3 items-end" style="animation: hudFadeIn 1s ease-out forwards; opacity: {flythroughActive ? 0.6 : 0.8};">
			<div class="text-[9px] font-mono leading-relaxed space-y-0.5 px-2 py-1.5 rounded" style="background: rgba(0,0,0,0.45); backdrop-filter: blur(4px);">
				{#each terrainCounts as [code, count]}
					<div class="flex items-center gap-1.5">
						<span class="text-white/70 w-[52px] inline-block">{terrainNames.get(code) ?? '?'}</span>
						<span class="text-white/50 w-[24px] inline-block text-right">{(count / total * 100).toFixed(0)}%</span>
						<span class="w-2 h-2 rounded-sm shrink-0" style="background: {terrainColors[code] ?? '#333'}"></span>
					</div>
				{/each}
			</div>
			<canvas
				class="rounded border border-white/20"
				width={grid[0].length * 4}
				height={grid.length * 4}
				style="width: {Math.min(140, grid[0].length * 4)}px; height: {Math.min(140, grid.length * 4)}px; image-rendering: pixelated;"
				use:drawMinimap={grid}
			></canvas>
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
		<!-- Crosshair (hidden during flythrough) -->
		{#if !flythroughActive}
		<div class="absolute inset-0 pointer-events-none flex items-center justify-center z-10">
			<div class="w-5 h-5 relative opacity-40">
				<div class="absolute top-1/2 left-0 w-full h-px bg-white -translate-y-px"></div>
				<div class="absolute left-1/2 top-0 h-full w-px bg-white -translate-x-px"></div>
			</div>
		</div>
		{/if}

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
			{@const h = Math.floor(displayTime)}
			{@const m = Math.floor((displayTime - h) * 60)}
			{@const timeStr = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`}
			{@const timeLabel = displayTime < 6 ? 'Night' : displayTime < 8 ? 'Dawn' : displayTime < 12 ? 'Morning' : displayTime < 14 ? 'Midday' : displayTime < 17 ? 'Afternoon' : displayTime < 19 ? 'Dusk' : 'Night'}
			{@const weatherLabel = windSystem?.weather === 'storm' ? 'Storm' : windSystem?.weather === 'rain' ? 'Rain' : windSystem?.weather === 'cloudy' ? 'Cloudy' : 'Clear'}
			<div class="absolute inset-0 pointer-events-none z-20 flex flex-col items-center"
				style="animation: hudFadeIn 2s ease-out forwards">
				<div class="mt-6 text-center flex flex-col items-center">
				<div class="inline-block px-6 py-2.5 rounded-lg" style="background: rgba(10, 15, 40, 0.50); backdrop-filter: blur(6px);">
					<h1 class="text-2xl font-thin tracking-[0.35em] uppercase"
						style="color: rgba(255,255,255,0.85);
							text-shadow: 0 0 30px rgba(255,255,255,0.3), 0 0 60px rgba(135,206,235,0.2);">
						ASTAR ISLAND
					</h1>
					<div class="mt-1.5 flex items-center justify-center gap-2 text-[10px] font-mono tracking-wider uppercase"
						style="color: rgba(255,255,255,0.50); text-shadow: 0 0 15px rgba(255,255,255,0.2);">
						{#if roundLabel}<span>{roundLabel}</span><span class="text-white/25">&middot;</span>{/if}
						{#if seedLabel}<span>{seedLabel}</span><span class="text-white/25">&middot;</span>{/if}
						<span>{timeStr} {timeLabel}</span>
						<span class="text-white/25">&middot;</span>
						<span>{weatherLabel}</span>
					</div>
				</div>
				</div>
			</div>

			<!-- Exit hint -->
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
