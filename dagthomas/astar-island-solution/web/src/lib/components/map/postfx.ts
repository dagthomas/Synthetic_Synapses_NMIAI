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
	VignetteEffect,
	ToneMappingEffect,
	SMAAEffect,
	SMAAPreset,
	ToneMappingMode,
	DepthOfFieldEffect
} from 'postprocessing';

export interface PostFXPipeline {
	composer: EffectComposer;
	render(dt: number): void;
	updateAtmosphere(timeOfDay: number): void;
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
		bokehScale: 3.0
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

	return {
		composer,

		render(dt: number) {
			composer.render(dt);
		},

		updateAtmosphere(timeOfDay: number) {
			const isNight = timeOfDay < 6 || timeOfDay > 18;
			const isDawnDusk = (timeOfDay >= 5 && timeOfDay <= 7) || (timeOfDay >= 17 && timeOfDay <= 19);
			const lumMat = bloom.luminanceMaterial;

			if (isDawnDusk) {
				// Golden hour: heavy bloom, warm glow everywhere
				bloom.intensity = 1.4;
				lumMat.threshold = 0.35;
				vignette.darkness = 0.6;
				dof.bokehScale = 4.0;
			} else if (isNight) {
				// Night: moody, dark vignette, fires glow hard
				bloom.intensity = 1.1;
				lumMat.threshold = 0.3;
				vignette.darkness = 0.75;
				dof.bokehScale = 2.5;
			} else {
				// Day: clean, subtle effects
				bloom.intensity = 0.7;
				lumMat.threshold = 0.65;
				vignette.darkness = 0.45;
				dof.bokehScale = 3.0;
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
