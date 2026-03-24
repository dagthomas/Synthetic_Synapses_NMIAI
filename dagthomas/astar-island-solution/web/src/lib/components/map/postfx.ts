/**
 * Post-processing pipeline for epic first-person mode.
 * Bloom + vignette + ACES tone mapping + SMAA + depth-of-field hint.
 */
import * as THREE from 'three';
import {
	EffectComposer,
	EffectPass,
	RenderPass,
	BloomEffect,
	BlendFunction,
	VignetteEffect,
	ToneMappingEffect,
	SMAAEffect,
	SMAAPreset,
	ToneMappingMode,
	DepthOfFieldEffect,
	GodRaysEffect,
	ChromaticAberrationEffect,
	NoiseEffect
} from 'postprocessing';

export type ViewMode = 'orbit' | 'fp' | 'flythrough';

export interface PostFXPipeline {
	composer: EffectComposer;
	render(dt: number): void;
	setMode(mode: ViewMode): void;
	updateAtmosphere(timeOfDay: number, mode?: ViewMode): void;
	setSunMesh(mesh: THREE.Mesh | null): void;
	resize(width: number, height: number): void;
	dispose(): void;
}

export function createPostFX(
	renderer: THREE.WebGLRenderer,
	scene: THREE.Scene,
	camera: THREE.PerspectiveCamera
): PostFXPipeline {
	const composer = new EffectComposer(renderer, {
		frameBufferType: THREE.HalfFloatType
	});

	// Render pass
	const renderPass = new RenderPass(scene, camera);
	composer.addPass(renderPass);

	// Bloom — mipmap blur for soft, cinematic glow
	const bloom = new BloomEffect({
		intensity: 0.8,
		luminanceThreshold: 0.6,
		luminanceSmoothing: 0.4,
		mipmapBlur: true,
		radius: 0.6
	});

	// Depth of field — subtle background blur for cinematic feel
	const dof = new DepthOfFieldEffect(camera, {
		focusDistance: 0.02,
		focalLength: 0.05,
		bokehScale: 1.5
	});

	// Vignette — cinematic dark edges
	const vignette = new VignetteEffect({
		darkness: 0.5,
		offset: 0.3
	});

	// ACES Filmic tone mapping
	const toneMapping = new ToneMappingEffect({
		mode: ToneMappingMode.ACES_FILMIC,
		resolution: 256,
		whitePoint: 4.0,
		middleGrey: 0.65,
		minLuminance: 0.01,
		averageLuminance: 1.0,
		adaptationRate: 1.5
	});

	// SMAA anti-aliasing
	const smaa = new SMAAEffect({ preset: SMAAPreset.HIGH });

	// Chromatic aberration — subtle lens distortion at edges
	const chromaticAberration = new ChromaticAberrationEffect({
		offset: new THREE.Vector2(0.0006, 0.0006),
		radialModulation: true,
		modulationOffset: 0.2
	});

	// Film grain — subtle noise for cinematic texture
	const noise = new NoiseEffect({
		premultiply: true
	});
	noise.blendMode.opacity.value = 0.15;

	// God rays — initialized without light source, added when sun mesh is available
	let godRays: GodRaysEffect | null = null;
	let godRaysPass: EffectPass | null = null;

	// Merge effects into minimal passes
	composer.addPass(new EffectPass(camera, bloom, dof, vignette));
	composer.addPass(new EffectPass(camera, chromaticAberration, noise));
	composer.addPass(new EffectPass(camera, toneMapping, smaa));

	let currentMode: ViewMode = 'orbit';
	dof.blendMode.setBlendFunction(BlendFunction.SKIP);
	// Disable cinematic effects in orbit mode
	chromaticAberration.blendMode.setBlendFunction(BlendFunction.SKIP);
	noise.blendMode.setBlendFunction(BlendFunction.SKIP);

	return {
		composer,

		render(dt: number) {
			composer.render(dt);
		},

		setMode(mode: ViewMode) {
			if (mode === currentMode) return;
			currentMode = mode;
			if (mode === 'orbit') {
				dof.blendMode.setBlendFunction(BlendFunction.SKIP);
				chromaticAberration.blendMode.setBlendFunction(BlendFunction.SKIP);
				noise.blendMode.setBlendFunction(BlendFunction.SKIP);
				if (godRays) godRays.blendMode.setBlendFunction(BlendFunction.SKIP);
			} else {
				dof.blendMode.setBlendFunction(BlendFunction.NORMAL);
				chromaticAberration.blendMode.setBlendFunction(BlendFunction.NORMAL);
				noise.blendMode.setBlendFunction(BlendFunction.NORMAL);
				if (godRays) godRays.blendMode.setBlendFunction(BlendFunction.SCREEN);
			}
		},

		setSunMesh(mesh: THREE.Mesh | null) {
			if (!mesh || godRays) return; // only create once
			godRays = new GodRaysEffect(camera, mesh, {
				density: 0.96,
				decay: 0.93,
				weight: 0.4,
				exposure: 0.55,
				samples: 60,
				clampMax: 1.0
			});
			godRays.blendMode.setBlendFunction(
				currentMode === 'orbit' ? BlendFunction.SKIP : BlendFunction.SCREEN
			);
			godRaysPass = new EffectPass(camera, godRays);
			// Insert before tone mapping pass (second-to-last)
			const passes = composer.passes;
			composer.removePass(passes[passes.length - 1]); // remove toneMapping+smaa
			composer.addPass(godRaysPass);
			composer.addPass(new EffectPass(camera, toneMapping, smaa)); // re-add last
		},

		updateAtmosphere(timeOfDay: number, mode: ViewMode = 'orbit') {
			const isNight = timeOfDay < 6 || timeOfDay > 18;
			const isDawnDusk = (timeOfDay >= 5 && timeOfDay <= 7) || (timeOfDay >= 17 && timeOfDay <= 19);
			const lumMat = bloom.luminanceMaterial;

			// DOF scale per view mode: orbit = disabled, flythrough = moderate, FP = full
			const dofBase = mode === 'fp' ? 1.0 : mode === 'flythrough' ? 0.3 : 0.0;

			if (isDawnDusk) {
				bloom.intensity = 1.5;
				lumMat.threshold = 0.30;
				vignette.darkness = 0.6;
				dof.bokehScale = 4.0 * dofBase;
				if (godRays) godRays.godRaysMaterial.weight = 0.7; // strongest at golden hour
				noise.blendMode.opacity.value = 0.12;
				chromaticAberration.offset.set(0.001, 0.001); // more distortion at dawn/dusk
			} else if (isNight) {
				bloom.intensity = 1.7;
				lumMat.threshold = 0.20;
				vignette.darkness = 0.65;
				dof.bokehScale = 2.5 * dofBase;
				if (godRays) godRays.godRaysMaterial.weight = 0.0; // no sun rays at night
				noise.blendMode.opacity.value = 0.2; // more grain at night (cinematic)
				chromaticAberration.offset.set(0.0004, 0.0004);
			} else {
				bloom.intensity = 0.9;
				lumMat.threshold = 0.45;
				vignette.darkness = 0.45;
				dof.bokehScale = 3.0 * dofBase;
				if (godRays) godRays.godRaysMaterial.weight = 0.35;
				noise.blendMode.opacity.value = 0.1;
				chromaticAberration.offset.set(0.0006, 0.0006);
			}
		},

		resize(width: number, height: number) {
			composer.setSize(width, height);
		},

		dispose() {
			composer.dispose();
		}
	};
}
