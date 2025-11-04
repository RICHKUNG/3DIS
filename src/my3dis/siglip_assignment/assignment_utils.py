"""
Utility functions for SigLIP feature assignment and open-vocabulary inference.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def aggregate_features(
    features_list: List[torch.Tensor],
    method: str = 'mean',
) -> torch.Tensor:
    """
    Aggregate features from multiple views.

    Args:
        features_list: List of feature tensors [D]
        method: Aggregation method ('mean', 'max', 'weighted_mean')

    Returns:
        Aggregated feature [D]
    """
    if len(features_list) == 0:
        raise ValueError("Empty feature list")

    if len(features_list) == 1:
        return features_list[0]

    features_stack = torch.stack(features_list)

    if method == 'mean':
        return features_stack.mean(dim=0)
    elif method == 'max':
        return features_stack.max(dim=0)[0]
    elif method == 'weighted_mean':
        # Weight by L2 norm (stronger features get more weight)
        weights = features_stack.norm(dim=1)
        weights = weights / weights.sum()
        return (features_stack * weights[:, None]).sum(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def assign_labels(
    features: torch.Tensor,
    text_features: torch.Tensor,
    scale_factor: float = 300.0,
    duplicate: bool = True,
    topk: int = 600,
) -> Dict[str, np.ndarray]:
    """
    Assign labels to proposals using open-vocabulary inference.

    Based on the implementation in utils_new/utils_ov_inference.py

    Args:
        features: [N_proposals, D] proposal features
        text_features: [N_labels, D] text features
        scale_factor: Temperature scaling factor
        duplicate: Allow duplicate labels (top-K per label)
        topk: Number of top predictions to keep

    Returns:
        Dictionary with prediction results
    """
    N_proposals = features.shape[0]
    N_labels = text_features.shape[0]

    # Compute similarity scores
    similarity = scale_factor * (features @ text_features.T)  # [N_proposals, N_labels]
    similarity = F.softmax(similarity, dim=-1)

    if duplicate:
        # Top-K instances across all (proposal, label) pairs
        similarity_flat = similarity.reshape(-1)  # [N_proposals * N_labels]

        labels_expanded = (
            torch.arange(N_labels, device=similarity.device)
            .unsqueeze(0)
            .repeat(N_proposals, 1)
            .flatten()
        )

        # Get top-K
        scores_final, idx = torch.topk(similarity_flat, k=min(topk, len(similarity_flat)), largest=True)
        mask_idx = torch.div(idx, N_labels, rounding_mode="floor")
        labels_final = labels_expanded[idx]

        return {
            'pred_classes': labels_final.cpu().numpy(),
            'pred_scores': scores_final.cpu().numpy(),
            'pred_mask_indices': mask_idx.cpu().numpy(),
        }
    else:
        # Best label per proposal
        scores_final, labels_final = similarity.max(dim=1)

        return {
            'pred_classes': labels_final.cpu().numpy(),
            'pred_scores': scores_final.cpu().numpy(),
            'pred_mask_indices': np.arange(N_proposals),
        }


def assign_part_labels(
    features_object: torch.Tensor,
    features_part: torch.Tensor,
    text_features_object: torch.Tensor,
    text_features_part: torch.Tensor,
    part_to_object_mapping: Dict[int, int],
    scale_factor: float = 300.0,
) -> Dict[str, np.ndarray]:
    """
    Assign part-level labels with object-part fusion.

    Based on ov_inference_part() from utils_new/utils_ov_inference.py

    Args:
        features_object: [N_proposals, D] object-level features
        features_part: [N_proposals, D] part-level features
        text_features_object: [N_object_labels, D] object text features
        text_features_part: [N_part_labels, D] part text features
        part_to_object_mapping: Mapping from part label idx to object label idx
        scale_factor: Temperature scaling

    Returns:
        Dictionary with part predictions
    """
    # Object-level scores
    similarity_object = scale_factor * (features_object @ text_features_object.T)
    similarity_object = F.softmax(similarity_object, dim=-1)

    # Part-level scores
    similarity_part = scale_factor * (features_part @ text_features_part.T)
    similarity_part = F.softmax(similarity_part, dim=-1)

    # Map object scores to part label space
    N_proposals = features_part.shape[0]
    N_part_labels = text_features_part.shape[0]

    similarity_object_mapped = torch.zeros(
        (N_proposals, N_part_labels),
        device=similarity_part.device,
    )

    for part_idx, obj_idx in part_to_object_mapping.items():
        similarity_object_mapped[:, part_idx] = similarity_object[:, obj_idx]

    # Fuse scores (average)
    similarity_fused = (similarity_part + similarity_object_mapped) / 2

    # Get best predictions
    scores_final, labels_final = similarity_fused.max(dim=1)

    return {
        'pred_classes': labels_final.cpu().numpy(),
        'pred_scores': scores_final.cpu().numpy(),
        'similarity_part': similarity_part.cpu().numpy(),
        'similarity_object': similarity_object.cpu().numpy(),
        'similarity_fused': similarity_fused.cpu().numpy(),
    }


def compute_mask_scores(
    point_features: torch.Tensor,
    text_features: torch.Tensor,
    instance_masks: torch.Tensor,
    scale_factor: float = 300.0,
) -> torch.Tensor:
    """
    Compute mask-wise semantic scores.

    Args:
        point_features: [N_points, D] per-point features
        text_features: [N_labels, D] text features
        instance_masks: [N_proposals, N_points] binary masks
        scale_factor: Temperature scaling

    Returns:
        Mask-wise scores [N_proposals, N_labels]
    """
    # Per-point classification scores
    predicted_class = scale_factor * (point_features @ text_features.T)  # [N_points, N_labels]
    predicted_class = F.softmax(predicted_class, dim=-1)

    # Aggregate to mask level
    inst_class_scores = torch.einsum(
        "kn,nc->kc",
        instance_masks.float(),
        predicted_class,
    )  # [N_proposals, N_labels]

    # Normalize by mask size
    inst_class_scores = inst_class_scores / instance_masks.float().sum(dim=1, keepdim=True)

    return inst_class_scores
