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
	DepthOfFieldEffect
} from 'postprocessing';

export type ViewMode = 'orbit' | 'fp' | 'flythrough';

export interface PostFXPipeline {
	composer: EffectComposer;
	render(dt: number): void;
	setMode(mode: ViewMode): void;
	updateAtmosphere(timeOfDay: number, mode?: ViewMode): void;
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

	// Merge effects into minimal passes
	composer.addPass(new EffectPass(camera, bloom, dof, vignette));
	composer.addPass(new EffectPass(camera, toneMapping, smaa));

	let currentMode: ViewMode = 'orbit';
	// Start with DOF disabled in orbit mode (expensive depth pre-pass for barely visible effect)
	dof.blendMode.setBlendFunction(BlendFunction.SKIP);

	return {
		composer,

		render(dt: number) {
			composer.render(dt);
		},

		setMode(mode: ViewMode) {
			if (mode === currentMode) return;
			currentMode = mode;
			if (mode === 'orbit') {
				// Disable DOF in orbit — bokehScale 0.15 is invisible, but depth pre-pass is expensive
				dof.blendMode.setBlendFunction(BlendFunction.SKIP);
			} else {
				// Enable DOF in FP/flythrough
				dof.blendMode.setBlendFunction(BlendFunction.NORMAL);
			}
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
			} else if (isNight) {
				bloom.intensity = 1.7;
				lumMat.threshold = 0.20;
				vignette.darkness = 0.65;
				dof.bokehScale = 2.5 * dofBase;
			} else {
				// Bloom visible in all modes — lower threshold so terrain/buildings glow softly
				bloom.intensity = 0.9;
				lumMat.threshold = 0.45;
				vignette.darkness = 0.45;
				dof.bokehScale = 3.0 * dofBase;
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
