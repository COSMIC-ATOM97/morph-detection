import numpy as np

def compute_calibrated_decision_scores(spatial_margins: np.ndarray,
                                       freq_margins: np.ndarray,
                                       bona_fide_mask: np.ndarray):
    """
    Standardize linear cosine margin scores against in-domain bona fide baseline statistics
    to eliminate camera sensor noise (PRNU) false-alarm biases.
    
    Args:
        spatial_margins: Margin array S_spatial = cos(theta_morph) - cos(theta_bona)
        freq_margins: Margin array S_freq = cos(theta_morph) - cos(theta_bona)
        bona_fide_mask: Boolean array where True indicates bona fide samples
        
    Returns:
        calibrated_scores: Z-score standardized late decision fusion scores
    """
    bf_spatial = spatial_margins[bona_fide_mask]
    bf_freq = freq_margins[bona_fide_mask]

    mu_s, std_s = float(np.mean(bf_spatial)), float(np.std(bf_spatial) + 1e-7)
    mu_f, std_f = float(np.mean(bf_freq)), float(np.std(bf_freq) + 1e-7)

    # Standardize both streams against genuine baseline
    z_spatial = (spatial_margins - mu_s) / std_s
    z_freq = (freq_margins - mu_f) / std_f

    # Late calibrated decision fusion
    return 0.5 * z_spatial + 0.5 * z_freq
