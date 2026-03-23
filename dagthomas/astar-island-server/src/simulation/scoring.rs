//! Entropy-weighted KL divergence scoring (competition formula).

use super::cell::NUM_CLASSES;

/// Compute entropy-weighted KL divergence score (0-100).
///
/// Port of sim_model.py:compute_score.
/// score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
pub fn compute_score(gt: &[[f64; NUM_CLASSES]], pred: &[[f64; NUM_CLASSES]]) -> f64 {
    assert_eq!(gt.len(), pred.len());

    let mut weight_sum = 0.0;
    let mut weighted_kl_sum = 0.0;

    for i in 0..gt.len() {
        // Compute entropy of ground truth cell
        let mut entropy = 0.0;
        for c in 0..NUM_CLASSES {
            let p = gt[i][c].max(1e-10);
            entropy -= p * p.ln();
        }

        // Skip static cells (near-zero entropy)
        if entropy <= 0.01 {
            continue;
        }

        // Compute KL divergence: KL(gt || pred)
        let mut kl = 0.0;
        for c in 0..NUM_CLASSES {
            let p = gt[i][c].max(1e-10);
            let q = pred[i][c].max(1e-10);
            kl += p * (p / q).ln();
        }

        weight_sum += entropy;
        weighted_kl_sum += entropy * kl;
    }

    if weight_sum <= 0.0 {
        return 100.0;
    }

    let wkl = weighted_kl_sum / weight_sum;
    (100.0 * (-3.0 * wkl).exp()).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_prediction() {
        let gt = vec![[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]; 10];
        let score = compute_score(&gt, &gt);
        assert!((score - 100.0).abs() < 0.01, "Perfect prediction should score 100, got {score}");
    }

    #[test]
    fn test_uniform_prediction() {
        let gt = vec![[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]; 100];
        let uniform = vec![[1.0 / 6.0; NUM_CLASSES]; 100];
        let score = compute_score(&gt, &uniform);
        assert!(score < 50.0, "Uniform prediction should score low, got {score}");
        assert!(score > 0.0, "Uniform prediction should be above 0, got {score}");
    }

    #[test]
    fn test_static_cells_ignored() {
        // All cells are deterministic (entropy ~0) → score should be 100
        let gt = vec![[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 100];
        let pred = vec![[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]; 100];
        let score = compute_score(&gt, &pred);
        assert!((score - 100.0).abs() < 0.01, "Static cells should give perfect score, got {score}");
    }
}
