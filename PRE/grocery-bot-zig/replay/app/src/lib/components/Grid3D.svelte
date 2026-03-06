<script>
	import { onMount } from 'svelte';
	import * as THREE from 'three';
	import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

	let {
		width,
		height,
		wallSet,
		shelfSet,
		itemMap,
		dropOff,
		dropOffZones = null,
		spawn,
		bots = [],
		botColors = [],
		selectedBot = null,
		onSelectBot = () => {},
		activeTypes = new Set(),
		previewTypes = new Set(),
		difficulty = null,
	} = $props();

	const nightmare = $derived(difficulty === 'nightmare');

	let container;
	let renderer, scene, camera, controls;
	let botMeshes = new Map();
	let botLabelSprites = new Map();
	let invDots = [];
	let activeGlows = [];
	let previewGlows = [];
	let dropoffGlows = [];
	let clock;
	let animId;
	let mounted = false;

	// Colors
	const FLOOR_COLOR = 0x0a0e17;
	const WALL_COLOR = 0x2d333b;
	const WALL_EDGE_COLOR = 0x22272e;
	const SHELF_COLOR = 0x0d2818;
	const DROPOFF_COLOR = 0x39d353;
	const SPAWN_COLOR = 0x58a6ff;

	const NM_FLOOR_COLOR = 0x1a1818;
	const NM_WALL_COLOR = 0x351515;
	const NM_WALL_EDGE_COLOR = 0x2a1010;
	const NM_SHELF_COLOR = 0x1a0a0a;
	const NM_DROPOFF_COLOR = 0x880000;
	const NM_SPAWN_COLOR = 0xaa0033;

	const ITEM_CLR = {
		milk: 0xdfe6e9, bread: 0xffeaa7, eggs: 0xfab1a0, butter: 0xfdcb6e,
		cheese: 0xf39c12, pasta: 0xe17055, rice: 0xdfe6e9, juice: 0x8aa8b8,
		yogurt: 0xa29bfe, cereal: 0xe67e22, flour: 0xb2bec3, sugar: 0xdfe6e9,
		coffee: 0x6d4c2a, tea: 0x4dbd6a, oil: 0xfdcb6e, salt: 0xb2bec3,
		cream: 0xdfe6e9, oats: 0xd4a76a,
		apples: 0x33ff00, bananas: 0x556655, carrots: 0xaa0000, lettuce: 0xaa6677,
		onions: 0x665544, peppers: 0xff6600, tomatoes: 0xddaa00,
	};

	// Build dropoff set
	const dropOffList = $derived(dropOffZones && dropOffZones.length > 0 ? dropOffZones : (dropOff ? [dropOff] : []));
	const dropOffSet = $derived(new Set(dropOffList.map(d => `${d[0]},${d[1]}`)));

	function getItemColor(type) {
		return ITEM_CLR[type] || 0xaaaaaa;
	}

	function parseBotColor(cssColor) {
		return parseInt(cssColor.replace('#', ''), 16);
	}

	function buildScene() {
		if (!scene) return;

		// Clear previous
		while (scene.children.length > 0) scene.remove(scene.children[0]);
		botMeshes.clear();
		botLabelSprites.clear();
		invDots = [];
		activeGlows = [];
		previewGlows = [];
		dropoffGlows = [];

		const nm = nightmare;
		const floorClr = nm ? NM_FLOOR_COLOR : FLOOR_COLOR;
		const wallClr = nm ? NM_WALL_COLOR : WALL_COLOR;
		const wallEdgeClr = nm ? NM_WALL_EDGE_COLOR : WALL_EDGE_COLOR;
		const shelfClr = nm ? NM_SHELF_COLOR : SHELF_COLOR;
		const dropClr = nm ? NM_DROPOFF_COLOR : DROPOFF_COLOR;
		const spawnClr = nm ? NM_SPAWN_COLOR : SPAWN_COLOR;

		// Ambient + directional light
		const ambient = new THREE.AmbientLight(nm ? 0x331111 : 0x334455, 0.6);
		scene.add(ambient);
		const dirLight = new THREE.DirectionalLight(nm ? 0xff4444 : 0xffffff, 0.8);
		dirLight.position.set(width * 0.5, 15, -height * 0.3);
		dirLight.castShadow = true;
		dirLight.shadow.mapSize.set(2048, 2048);
		dirLight.shadow.camera.left = -width;
		dirLight.shadow.camera.right = width;
		dirLight.shadow.camera.top = height;
		dirLight.shadow.camera.bottom = -height;
		scene.add(dirLight);

		// Subtle point light for glow feel
		const pointLight = new THREE.PointLight(nm ? 0xff2200 : 0x39d353, 0.3, 50);
		pointLight.position.set(width / 2, 8, height / 2);
		scene.add(pointLight);

		// Floor plane
		const floorGeo = new THREE.PlaneGeometry(width, height);
		const floorMat = new THREE.MeshStandardMaterial({
			color: floorClr,
			roughness: 0.9,
			metalness: 0.1,
		});
		const floor = new THREE.Mesh(floorGeo, floorMat);
		floor.rotation.x = -Math.PI / 2;
		floor.position.set(width / 2 - 0.5, -0.01, height / 2 - 0.5);
		floor.receiveShadow = true;
		scene.add(floor);

		// Grid lines on floor
		const gridHelper = new THREE.GridHelper(Math.max(width, height), Math.max(width, height), nm ? 0x2a2626 : 0x161b22, nm ? 0x221e1e : 0x0d1117);
		gridHelper.position.set(width / 2 - 0.5, 0.005, height / 2 - 0.5);
		scene.add(gridHelper);

		// Shared geometries
		const wallGeo = new THREE.BoxGeometry(0.95, 0.8, 0.95);
		const shelfGeo = new THREE.BoxGeometry(0.92, 0.35, 0.92);
		const itemGeo = new THREE.SphereGeometry(0.12, 8, 8);

		// Instanced walls
		const wallPositions = [];
		const edgeWallPositions = [];
		for (const key of wallSet) {
			const [x, y] = key.split(',').map(Number);
			const isEdge = (x === 0 || y === 0 || x === width - 1 || y === height - 1);
			if (isEdge) edgeWallPositions.push([x, y]);
			else wallPositions.push([x, y]);
		}

		// Edge walls
		if (edgeWallPositions.length > 0) {
			const edgeMat = new THREE.MeshStandardMaterial({ color: wallEdgeClr, roughness: 0.8, metalness: 0.2 });
			const edgeInst = new THREE.InstancedMesh(wallGeo, edgeMat, edgeWallPositions.length);
			const dummy = new THREE.Object3D();
			edgeWallPositions.forEach(([x, y], i) => {
				dummy.position.set(x, 0.4, y);
				dummy.updateMatrix();
				edgeInst.setMatrixAt(i, dummy.matrix);
			});
			edgeInst.castShadow = true;
			edgeInst.receiveShadow = true;
			scene.add(edgeInst);
		}

		// Interior walls
		if (wallPositions.length > 0) {
			const wallMat = new THREE.MeshStandardMaterial({ color: wallClr, roughness: 0.7, metalness: 0.2 });
			const wallInst = new THREE.InstancedMesh(wallGeo, wallMat, wallPositions.length);
			const dummy = new THREE.Object3D();
			wallPositions.forEach(([x, y], i) => {
				dummy.position.set(x, 0.4, y);
				dummy.updateMatrix();
				wallInst.setMatrixAt(i, dummy.matrix);
			});
			wallInst.castShadow = true;
			wallInst.receiveShadow = true;
			scene.add(wallInst);
		}

		// Shelves
		const shelfPositions = [];
		for (const key of shelfSet) {
			const [x, y] = key.split(',').map(Number);
			shelfPositions.push([x, y]);
		}
		if (shelfPositions.length > 0) {
			const shelfMat = new THREE.MeshStandardMaterial({
				color: shelfClr,
				roughness: 0.6,
				metalness: 0.3,
				emissive: nm ? 0x110505 : 0x001a08,
				emissiveIntensity: 0.2,
			});
			const shelfInst = new THREE.InstancedMesh(shelfGeo, shelfMat, shelfPositions.length);
			const dummy = new THREE.Object3D();
			shelfPositions.forEach(([x, y], i) => {
				dummy.position.set(x, 0.175, y);
				dummy.updateMatrix();
				shelfInst.setMatrixAt(i, dummy.matrix);
			});
			shelfInst.castShadow = true;
			shelfInst.receiveShadow = true;
			scene.add(shelfInst);
		}

		// Items on shelves (small spheres)
		for (const [key, items] of itemMap) {
			const [x, y] = key.split(',').map(Number);
			const itype = items[0].type;
			const clr = getItemColor(itype);
			const isActive = activeTypes.has(itype);
			const isPreview = !isActive && previewTypes.has(itype);

			const itemMat = new THREE.MeshStandardMaterial({
				color: clr,
				emissive: clr,
				emissiveIntensity: isActive ? 0.6 : isPreview ? 0.3 : 0.1,
				roughness: 0.4,
				metalness: 0.5,
			});
			const sphere = new THREE.Mesh(itemGeo, itemMat);
			sphere.position.set(x, 0.5, y);
			sphere.castShadow = true;
			scene.add(sphere);

			if (isActive) {
				// Glowing ring for active items
				const ringGeo = new THREE.RingGeometry(0.3, 0.42, 16);
				const ringMat = new THREE.MeshBasicMaterial({
					color: nm ? 0xff1493 : 0xfacc15,
					transparent: true,
					opacity: 0.6,
					side: THREE.DoubleSide,
				});
				const ring = new THREE.Mesh(ringGeo, ringMat);
				ring.rotation.x = -Math.PI / 2;
				ring.position.set(x, 0.01, y);
				scene.add(ring);
				activeGlows.push({ ring, mat: ringMat, x, y });
			} else if (isPreview) {
				const ringGeo = new THREE.RingGeometry(0.25, 0.35, 16);
				const ringMat = new THREE.MeshBasicMaterial({
					color: nm ? 0xff69b4 : 0xf472b6,
					transparent: true,
					opacity: 0.3,
					side: THREE.DoubleSide,
				});
				const ring = new THREE.Mesh(ringGeo, ringMat);
				ring.rotation.x = -Math.PI / 2;
				ring.position.set(x, 0.01, y);
				scene.add(ring);
				previewGlows.push({ ring, mat: ringMat });
			}
		}

		// Drop-off zones
		for (const [dx, dy] of dropOffList) {
			const dropGeo = new THREE.BoxGeometry(0.95, 0.05, 0.95);
			const dropMat = new THREE.MeshStandardMaterial({
				color: dropClr,
				emissive: dropClr,
				emissiveIntensity: 0.5,
				transparent: true,
				opacity: 0.8,
				roughness: 0.3,
				metalness: 0.7,
			});
			const dropMesh = new THREE.Mesh(dropGeo, dropMat);
			dropMesh.position.set(dx, 0.03, dy);
			scene.add(dropMesh);

			// Vertical beam
			const beamGeo = new THREE.CylinderGeometry(0.02, 0.3, 3, 8, 1, true);
			const beamMat = new THREE.MeshBasicMaterial({
				color: dropClr,
				transparent: true,
				opacity: 0.15,
				side: THREE.DoubleSide,
			});
			const beam = new THREE.Mesh(beamGeo, beamMat);
			beam.position.set(dx, 1.5, dy);
			scene.add(beam);
			dropoffGlows.push({ mesh: dropMesh, beam, mat: dropMat, beamMat });
		}

		// Spawn
		const spawnGeo = new THREE.BoxGeometry(0.95, 0.05, 0.95);
		const spawnMat = new THREE.MeshStandardMaterial({
			color: spawnClr,
			emissive: spawnClr,
			emissiveIntensity: 0.4,
			transparent: true,
			opacity: 0.7,
		});
		const spawnMesh = new THREE.Mesh(spawnGeo, spawnMat);
		spawnMesh.position.set(spawn[0], 0.03, spawn[1]);
		scene.add(spawnMesh);

		// Fog for atmosphere
		scene.fog = new THREE.FogExp2(nm ? 0x0a0505 : 0x010409, nm ? 0.04 : 0.03);

		updateBots();
	}

	function updateBots() {
		if (!scene) return;

		// Remove old bot meshes
		for (const [, mesh] of botMeshes) scene.remove(mesh);
		for (const [, sprite] of botLabelSprites) scene.remove(sprite);
		for (const dot of invDots) scene.remove(dot);
		botMeshes.clear();
		botLabelSprites.clear();
		invDots = [];

		const bodyGeo = new THREE.CapsuleGeometry(0.2, 0.4, 8, 12);
		const headGeo = new THREE.SphereGeometry(0.15, 12, 12);
		const eyeGeo = new THREE.SphereGeometry(0.04, 6, 6);
		const dotGeo = new THREE.SphereGeometry(0.06, 6, 6);

		for (const bot of bots) {
			const [bx, by] = bot.position;
			const cssColor = botColors[bot.id % botColors.length];
			const clr = parseBotColor(cssColor);
			const isSelected = selectedBot === bot.id;

			const group = new THREE.Group();

			// Body (capsule)
			const bodyMat = new THREE.MeshStandardMaterial({
				color: 0x161b22,
				emissive: clr,
				emissiveIntensity: isSelected ? 0.5 : 0.2,
				roughness: 0.4,
				metalness: 0.6,
			});
			const body = new THREE.Mesh(bodyGeo, bodyMat);
			body.position.y = 0.45;
			body.castShadow = true;
			group.add(body);

			// Head
			const headMat = new THREE.MeshStandardMaterial({
				color: clr,
				emissive: clr,
				emissiveIntensity: isSelected ? 0.8 : 0.3,
				roughness: 0.3,
				metalness: 0.7,
			});
			const head = new THREE.Mesh(headGeo, headMat);
			head.position.y = 0.85;
			head.castShadow = true;
			group.add(head);

			// Eyes
			const eyeMat = new THREE.MeshBasicMaterial({ color: nightmare ? 0xff0000 : 0xff0055 });
			const eyeL = new THREE.Mesh(eyeGeo, eyeMat);
			eyeL.position.set(-0.07, 0.87, -0.12);
			group.add(eyeL);
			const eyeR = new THREE.Mesh(eyeGeo, eyeMat);
			eyeR.position.set(0.07, 0.87, -0.12);
			group.add(eyeR);

			// Selection ring
			if (isSelected) {
				const selRingGeo = new THREE.RingGeometry(0.35, 0.42, 24);
				const selRingMat = new THREE.MeshBasicMaterial({
					color: clr,
					transparent: true,
					opacity: 0.7,
					side: THREE.DoubleSide,
				});
				const selRing = new THREE.Mesh(selRingGeo, selRingMat);
				selRing.rotation.x = -Math.PI / 2;
				selRing.position.y = 0.02;
				group.add(selRing);
			}

			group.position.set(bx, 0, by);
			scene.add(group);
			botMeshes.set(bot.id, group);

			// Inventory dots floating above
			bot.inventory.forEach((item, i) => {
				const dotMat = new THREE.MeshStandardMaterial({
					color: getItemColor(item),
					emissive: getItemColor(item),
					emissiveIntensity: 0.4,
				});
				const dot = new THREE.Mesh(dotGeo, dotMat);
				const offset = (i - (bot.inventory.length - 1) / 2) * 0.18;
				dot.position.set(bx + offset, 1.15, by);
				scene.add(dot);
				invDots.push(dot);
			});
		}
	}

	function animate() {
		if (!mounted) return;
		animId = requestAnimationFrame(animate);

		const t = clock.getElapsedTime();

		// Animate active item glows
		for (const g of activeGlows) {
			const pulse = Math.cos(t * 2.5) * 0.5 + 0.5;
			g.mat.opacity = 0.3 + pulse * 0.5;
			g.ring.scale.setScalar(1 + pulse * 0.15);
		}
		for (const g of previewGlows) {
			const pulse = Math.sin(t * 2.5) * 0.5 + 0.5;
			g.mat.opacity = 0.15 + pulse * 0.3;
		}

		// Animate dropoff beams
		for (const g of dropoffGlows) {
			const pulse = Math.sin(t * 1.5) * 0.5 + 0.5;
			g.mat.emissiveIntensity = 0.3 + pulse * 0.4;
			g.beamMat.opacity = 0.08 + pulse * 0.12;
			g.beam.rotation.y = t * 0.5;
		}

		// Bob bots gently
		for (const [id, group] of botMeshes) {
			group.position.y = Math.sin(t * 2 + id * 1.3) * 0.03;
		}

		// Float inventory dots
		for (let i = 0; i < invDots.length; i++) {
			invDots[i].position.y = 1.15 + Math.sin(t * 3 + i * 0.7) * 0.04;
		}

		controls.update();
		renderer.render(scene, camera);
	}

	function handleResize() {
		if (!container || !renderer || !camera) return;
		const w = container.clientWidth;
		const h = container.clientHeight;
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
		renderer.setSize(w, h);
	}

	onMount(() => {
		mounted = true;
		clock = new THREE.Clock();

		// Renderer
		renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		renderer.setSize(container.clientWidth, container.clientHeight);
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFSoftShadowMap;
		renderer.toneMapping = THREE.ACESFilmicToneMapping;
		renderer.toneMappingExposure = nightmare ? 0.8 : 1.0;
		container.appendChild(renderer.domElement);

		// Scene
		scene = new THREE.Scene();
		scene.background = new THREE.Color(nightmare ? 0x050202 : 0x010409);

		// Camera (isometric-ish perspective)
		const aspect = container.clientWidth / container.clientHeight;
		camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 200);
		const cx = width / 2;
		const cz = height / 2;
		camera.position.set(cx + width * 0.4, Math.max(width, height) * 0.7, cz + height * 0.6);
		camera.lookAt(cx, 0, cz);

		// Controls
		controls = new OrbitControls(camera, renderer.domElement);
		controls.target.set(cx - 0.5, 0, cz - 0.5);
		controls.enableDamping = true;
		controls.dampingFactor = 0.08;
		controls.maxPolarAngle = Math.PI / 2.1;
		controls.minDistance = 3;
		controls.maxDistance = Math.max(width, height) * 2;
		controls.update();

		// Click handler for bot selection
		const raycaster = new THREE.Raycaster();
		const mouse = new THREE.Vector2();
		renderer.domElement.addEventListener('dblclick', (e) => {
			const rect = renderer.domElement.getBoundingClientRect();
			mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
			mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
			raycaster.setFromCamera(mouse, camera);
			// Check bot meshes
			for (const [id, group] of botMeshes) {
				const intersects = raycaster.intersectObjects(group.children, true);
				if (intersects.length > 0) {
					onSelectBot(id);
					return;
				}
			}
		});

		buildScene();
		animate();

		const ro = new ResizeObserver(handleResize);
		ro.observe(container);

		return () => {
			mounted = false;
			if (animId) cancelAnimationFrame(animId);
			ro.disconnect();
			controls.dispose();
			renderer.dispose();
			if (renderer.domElement.parentNode) {
				renderer.domElement.parentNode.removeChild(renderer.domElement);
			}
		};
	});

	// Reactively rebuild when bots/orders change
	$effect(() => {
		// Touch reactive deps
		bots;
		activeTypes;
		previewTypes;
		selectedBot;
		if (scene && mounted) {
			buildScene();
		}
	});
</script>

<div class="grid3d-container" bind:this={container}></div>

<style>
	.grid3d-container {
		width: 100%;
		aspect-ratio: 16 / 10;
		border-radius: 4px;
		overflow: hidden;
		background: #010409;
	}
	.grid3d-container :global(canvas) {
		display: block;
		width: 100% !important;
		height: 100% !important;
	}
</style>
