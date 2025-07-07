import warnings

import numpy as np
import torch


def load_state_dict_flexible(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # If checkpoint is a wrapper (like {"state_dict": ...})
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state_dict = model.state_dict()
    loaded_state_dict = {}

    for name, param in state_dict.items():
        if name not in model_state_dict:
            warnings.warn(f"Skipping '{name}' as it is not in the model.")
            continue

        if model_state_dict[name].shape != param.shape:
            warnings.warn(
                f"Shape mismatch for '{name}': "
                f"model={model_state_dict[name].shape}, "
                f"checkpoint={param.shape}. Skipping."
            )
            continue

        loaded_state_dict[name] = param

    # Load only the matching parameters
    missing_keys, unexpected_keys = model.load_state_dict(
        loaded_state_dict, strict=False
    )

    if missing_keys:
        warnings.warn(f"Missing keys in checkpoint (not loaded): {missing_keys}")
    if unexpected_keys:
        warnings.warn(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")


def kl_divergence_kde(p_kde, q_kde, num_samples=10_000):
    """
    Approximate KL(P || Q) where P and Q are gaussian_kde objects.

    Args:
        p_kde: gaussian_kde object representing distribution P
        q_kde: gaussian_kde object representing distribution Q
        samples: Optional array of sample points to evaluate at (shape: [d, N])
        num_samples: Number of points to sample if samples not provided

    Returns:
        Approximate KL divergence D_KL(P || Q)
    """
    # Sample from P
    samples = p_kde.resample(num_samples)

    # Evaluate log densities
    p_vals = p_kde.evaluate(samples)
    q_vals = q_kde.evaluate(samples)

    # Add small epsilon to avoid log(0) or division by zero
    eps = 1e-10
    p_vals = np.clip(p_vals, eps, None)
    q_vals = np.clip(q_vals, eps, None)

    # Compute KL divergence
    kl_div = np.mean(np.log(p_vals / q_vals))
    return kl_div


def iou_kde(p_kde, q_kde, num_samples=10000):
    """
    Approximate IoU(P, Q) between two gaussian_kde distributions.

    Args:
        p_kde: gaussian_kde object (distribution P)
        q_kde: gaussian_kde object (distribution Q)
        samples: Optional sample locations (shape: [d, N])
        num_samples: Number of samples to draw if samples is None

    Returns:
        Approximate IoU value
    """
    # Draw from both distributions and merge
    samples_p = p_kde.resample(num_samples // 2)
    samples_q = q_kde.resample(num_samples // 2)
    samples = np.hstack([samples_p, samples_q])  # shape: [d, N]

    # Evaluate densities at sample points
    p_vals = p_kde.evaluate(samples)
    q_vals = q_kde.evaluate(samples)

    # Avoid numerical issues
    eps = 1e-10
    p_vals = np.clip(p_vals, eps, None)
    q_vals = np.clip(q_vals, eps, None)

    # Compute pointwise min and max
    intersection = np.minimum(p_vals, q_vals)
    union = np.maximum(p_vals, q_vals)

    # Approximate IoU
    iou = np.sum(intersection) / np.sum(union)
    return iou
