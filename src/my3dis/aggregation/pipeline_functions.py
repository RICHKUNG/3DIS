"""
Helper functions from pipeline.ipynb

這些函數直接從 pipeline.ipynb 複製，用於：
1. 建立 2D→3D proposal masks
2. 合併和過濾 proposals
3. 更新 family tree relations
"""

import numpy as np
import torch
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ==================== Mask Building (from cell: 82ce062b) ====================

def build_proposal_masks_from_masklets(
    masklet_path: str,
    pts_coords_per_frame: dict,
    pts_idx_per_frame: dict,
    P: int,
    frequency: int,
):
    """
    從 masklet segment 文件建立 proposal masks。

    Args:
        masklet_path: Level-X masklets 路徑
        pts_coords_per_frame: 從 get_visible_points 得到的座標
        pts_idx_per_frame: 從 get_visible_points 得到的索引
        P: 3D 點的數量
        frequency: Frame-to-mask index 轉換因子

    Returns:
        proposal_masks: (P, N_proposals) bool tensor
        sorted_obj_ids: 按列順序的 object IDs list
    """
    from my3dis.tracking import load_legacy_video_segments
    from my3dis.mask.encoding import unpack_binary_mask

    frames, _ = load_legacy_video_segments(masklet_path, unpack_masks=False)

    per_obj_point_sets = {}   # obj_id → set(point indices)

    for frame in frames.keys():
        f = frame / frequency
        coords = pts_coords_per_frame.get(f, None)
        idxs = pts_idx_per_frame.get(f, None)
        if coords is None or idxs is None or len(idxs) == 0:
            continue

        rows = coords[:, 0]
        cols = coords[:, 1]

        for obj_id, payload in frames[frame].items():
            mask = unpack_binary_mask(payload)
            h, w = mask.shape

            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            if not np.any(valid):
                continue

            inside = mask[rows[valid], cols[valid]]
            if not np.any(inside):
                continue

            chosen_points = idxs[valid][inside]
            s = per_obj_point_sets.setdefault(int(obj_id), set())
            s.update(map(int, chosen_points.tolist()))

    if len(per_obj_point_sets) == 0:
        logger.warning(f"No proposals found in {masklet_path}")
        return torch.zeros((P, 0), dtype=torch.bool), []

    sorted_obj_ids = sorted(per_obj_point_sets.keys())

    proposal_masks = torch.zeros((P, len(sorted_obj_ids)), dtype=torch.bool)
    for j, obj_id in enumerate(sorted_obj_ids):
        pts = list(per_obj_point_sets[obj_id])
        if len(pts) > 0:
            proposal_masks[pts, j] = True

    return proposal_masks, sorted_obj_ids


# ==================== Helper Functions (from cell: 2b57b34c) ====================

def _areas_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """mask: [P, N] bool → areas: [N]"""
    return mask.sum(dim=0)


def _iou_matrix(mask: torch.Tensor) -> torch.Tensor:
    """
    計算列之間的 IoU。
    mask: [P, N] bool
    return: [N, N] float
    """
    mask_f = mask.float()
    inter = torch.matmul(mask_f.t(), mask_f)             # [N, N]
    areas = mask_f.sum(dim=0, keepdim=True)              # [1, N]
    union = areas.t() + areas - inter + 1e-8
    iou = inter / union
    iou.fill_diagonal_(1.0)
    return iou


def _connected_components_by_thresh(iou: torch.Tensor, thresh: float):
    """
    建立無向圖，edge i<->j iff IoU>=thresh，返回 components。
    """
    N = iou.shape[0]
    adj = [[] for _ in range(N)]
    for i in range(N):
        neigh = torch.nonzero(iou[i] >= thresh, as_tuple=False).squeeze(1).tolist()
        for j in neigh:
            if j != i:
                adj[i].append(j)

    visited = [False]*N
    comps = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    comp.append(v)
        comps.append(sorted(comp))
    return comps


def _parent_groups_for_level(relations, level_key):
    """
    returns: dict[parent_id_or_None -> list[int proposal_ids]]
    """
    L = relations.get("hierarchy", {}).get(level_key, {})
    groups = defaultdict(list)
    for k, node in L.items():
        try:
            pid = int(k)
        except:
            continue
        par = node.get("parent", None)
        if par is not None:
            par = int(par)
        groups[par].append(pid)
    for par in list(groups.keys()):
        groups[par] = sorted(groups[par])
    return groups


def _ids_to_cols(
    ids,
    *,
    id_to_col: Dict[int, int] = None,
    total_cols: int = None,
    level_label: str = "",
):
    """
    Map proposal IDs to zero-based column indices.

    Args:
        ids: iterable of proposal ids (any type convertible to int)
        id_to_col: optional explicit mapping {proposal_id -> column_index}
        total_cols: optional column count for range checking
        level_label: optional string for logging context

    Returns:
        valid_ids: list[int] ids present in the mask
        cols: list[int] matching column indices
        missing_ids: list[int] ids not found / invalid
    """
    valid_ids = []
    cols = []
    missing = []

    for raw_id in ids:
        try:
            pid = int(raw_id)
        except (TypeError, ValueError):
            missing.append(raw_id)
            continue

        col_idx = None
        if id_to_col is not None:
            col_idx = id_to_col.get(pid, None)
        if col_idx is None:
            if id_to_col is None:
                col_idx = pid - 1
            else:
                missing.append(pid)
                continue

        if total_cols is not None and (col_idx < 0 or col_idx >= total_cols):
            missing.append(pid)
            continue

        valid_ids.append(pid)
        cols.append(col_idx)

    if missing and level_label:
        preview = ", ".join(str(x) for x in missing[:5])
        logger.warning(
            f"{level_label}: skipping {len(missing)} ids without proposal masks (showing up to 5): {preview}"
        )

    return valid_ids, cols, missing


def _safe_get_level(relations, level_key):
    return relations.setdefault("hierarchy", {}).setdefault(level_key, {})


# ==================== Level 2 Merge/Drop (from cell: cfe3a16a) ====================

def merge_drop_level2_mask(
    mask_l2: torch.Tensor,
    iou_thresh: float,
    area_thresh: int,
    proposal_ids: List[int] = None,
):
    """
    Returns:
      new_mask_l2: [P, N_keep] bool
      col_ids: list[int] - proposal_id for each surviving column
      old_to_rep_id: dict[int,int] - original proposal id -> representative id
      dropped_ids: set[int] - original ids removed
      merged_groups: dict[int,list[int]] - rep_id -> members (original ids)
    """
    assert mask_l2.dtype == torch.bool
    P, N = mask_l2.shape
    if proposal_ids is not None and len(proposal_ids) != N:
        raise ValueError(
            f"proposal_ids (len={len(proposal_ids)}) must match mask columns (N={N})"
        )

    # Normalize IDs so downstream logic never relies on 1-based indices
    id_lookup: List[int] = [
        int(proposal_ids[i]) if proposal_ids is not None else i + 1
        for i in range(N)
    ]

    # 1) Drop tiny first
    areas = _areas_from_mask(mask_l2)
    keep_small = (areas >= area_thresh)
    drop_idxs = torch.nonzero(~keep_small, as_tuple=False).squeeze(1).tolist()
    dropped_small_ids = [id_lookup[i] for i in drop_idxs]

    keep_mask = mask_l2[:, keep_small]
    keep_ids = [id_lookup[i] for i in range(N) if keep_small[i].item()]

    # 2) Merge by IoU
    if keep_mask.shape[1] == 0:
        return keep_mask, [], {}, set(dropped_small_ids), {}

    iou = _iou_matrix(keep_mask)
    comps = _connected_components_by_thresh(iou, iou_thresh)

    new_cols = []
    col_ids  = []
    merged_groups = {}
    old_to_rep_id = {}
    dropped_ids = set(dropped_small_ids)

    for comp in comps:
        comp_ids = [keep_ids[j] for j in comp]
        rep_id = min(comp_ids)
        merged_groups[rep_id] = sorted(comp_ids)

        # OR the columns
        col = keep_mask[:, comp[0]].clone()
        for j in comp[1:]:
            col |= keep_mask[:, j]
        new_cols.append(col)
        col_ids.append(rep_id)

        for old_id in comp_ids:
            old_to_rep_id[old_id] = rep_id
        for old_id in comp_ids:
            if old_id != rep_id:
                dropped_ids.add(old_id)

    new_mask_l2 = torch.stack(new_cols, dim=1) if len(new_cols) > 0 else torch.zeros((P,0), dtype=torch.bool)
    col_ids = list(map(int, col_ids))
    return new_mask_l2, col_ids, old_to_rep_id, dropped_ids, merged_groups


def split_merged_vs_small(
    dropped_ids: Set[int],
    merged_groups: Dict[int, List[int]],
) -> Tuple[Set[int], Set[int]]:
    """
    Returns:
        merged_away_ids: ids merged into representative
        dropped_small_ids: ids dropped for size
    """
    all_members = set()
    reps = set()
    for rep, members in merged_groups.items():
        reps.add(int(rep))
        for m in members:
            all_members.add(int(m))
    merged_away_ids = all_members - reps
    dropped_small_ids = set(int(x) for x in dropped_ids) - merged_away_ids
    return merged_away_ids, dropped_small_ids


# ==================== Relations Update (from cell: 1c7e94e5) ====================

def update_relations_level2_inplace_v2(
    relations: dict,
    merged_groups: Dict[int, List[int]],
    dropped_small_ids: Set[int],
):
    """
    修改 relations["hierarchy"]["2"] 和 ["4"] in place。
    """
    H = relations.setdefault("hierarchy", {})
    L2 = H.setdefault("2", {})
    L4 = H.setdefault("4", {})

    s = lambda x: str(int(x))

    # 1) Merge groups
    for rep_id, members in merged_groups.items():
        rep_id = int(rep_id)
        rep_key = s(rep_id)
        if rep_key not in L2:
            L2[rep_key] = {"parent": None, "children": [], "descendant_count": 0}

        merged_children = set(int(c) for c in L2.get(rep_key, {}).get("children", []))
        for mid in members:
            mid = int(mid)
            mkey = s(mid)
            if mkey in L2:
                merged_children.update(int(c) for c in L2[mkey].get("children", []))

        merged_children_list = sorted(merged_children)
        L2[rep_key]["children"] = merged_children_list
        L2[rep_key]["descendant_count"] = len(merged_children_list)

        for cid in merged_children_list:
            ckey = s(cid)
            if ckey in L4:
                L4[ckey]["parent"] = rep_id

        for mid in members:
            mid = int(mid)
            if mid != rep_id:
                mkey = s(mid)
                if mkey in L2:
                    del L2[mkey]

    # 2) Drop small
    dropped_children = {}
    for drop_id in list(dropped_small_ids):
        dkey = s(drop_id)
        node = L2.get(dkey)
        if node is None:
            continue
        children = [int(c) for c in node.get("children", [])]
        dropped_children[drop_id] = children

        for cid in children:
            ckey = s(cid)
            if ckey in L4 and L4[ckey].get("parent", None) == drop_id:
                L4[ckey]["parent"] = None

        del L2[dkey]

    # 3) Update descendant_count
    for k, node in L2.items():
        node["descendant_count"] = len(node.get("children", []))

    H["2"], H["4"] = L2, L4
    relations["hierarchy"] = H
    return dropped_children


# ==================== Level 4 Merge/Drop (from cell: 4bc6682c) ====================

def merge_drop_level4_within_siblings(
    mask_l4,
    relations,
    iou_thresh,
    area_thresh,
    proposal_ids=None,
):
    """
    Returns:
      new_mask_l4: [P, N4_keep] bool
      survivors_ids: list[int] representatives (1-based)
      old_to_rep: dict[int,int] (1-based -> 1-based)
      dropped_small_ids: set[int]
      merged_groups: dict[int, list[int]]
    Mutates relations["hierarchy"]["2"], ["4"], ["6"].
    proposal_ids: optional list mapping mask columns to existing L4 ids.
    """
    assert mask_l4.dtype == torch.bool
    P, N = mask_l4.shape

    L2 = _safe_get_level(relations, "2")
    L4 = _safe_get_level(relations, "4")
    L6 = _safe_get_level(relations, "6")

    groups = _parent_groups_for_level(relations, "4")
    id_to_col = None
    if proposal_ids is not None:
        id_to_col = {int(pid): idx for idx, pid in enumerate(proposal_ids)}

    new_cols = []
    survivors_ids = []
    old_to_rep = {}
    merged_groups = {}
    dropped_small_ids = set()

    for parent_id, l4_ids in groups.items():
        if len(l4_ids) == 0:
            continue

        valid_ids, cols, missing_ids = _ids_to_cols(
            l4_ids,
            id_to_col=id_to_col,
            total_cols=N,
            level_label="Level 4"
        )
        if missing_ids:
            for mid in missing_ids:
                try:
                    dropped_small_ids.add(int(mid))
                except (TypeError, ValueError):
                    continue
        if len(cols) == 0:
            continue

        submask = mask_l4[:, cols]
        areas = _areas_from_mask(submask)
        keep = (areas >= area_thresh)
        if keep.sum().item() == 0:
            for pid in valid_ids:
                dropped_small_ids.add(int(pid))
            continue

        keep_flags = keep.tolist()
        kept_ids = [valid_ids[i] for i, k in enumerate(keep_flags) if k]
        kept_cols = [c for c, k in zip(cols, keep_flags) if k]
        dropped_here = [valid_ids[i] for i, k in enumerate(keep_flags) if not k]
        dropped_small_ids.update(dropped_here)

        submask_kept = mask_l4[:, kept_cols]

        iou = _iou_matrix(submask_kept)
        comps = _connected_components_by_thresh(iou, iou_thresh)

        for comp in comps:
            member_ids = [kept_ids[j] for j in comp]
            rep_id = min(member_ids)
            merged_groups[rep_id] = sorted(member_ids)

            col = submask_kept[:, comp[0]].clone()
            for j in comp[1:]:
                col |= submask_kept[:, j]

            new_cols.append(col)
            survivors_ids.append(rep_id)
            for mid in member_ids:
                old_to_rep[mid] = rep_id

    new_mask_l4 = torch.stack(new_cols, dim=1) if len(new_cols) else torch.zeros((P,0), dtype=torch.bool)

    def s(x): return str(int(x))

    # Relations update
    for rep_id, members in merged_groups.items():
        rep_k = s(rep_id)
        if rep_k not in L4:
            L4[rep_k] = {"parent": None, "children": [], "descendant_count": 0}
        merged_children = set(int(c) for c in L4.get(rep_k, {}).get("children", []))
        for mid in members:
            mk = s(mid)
            if mk in L4:
                merged_children.update(int(c) for c in L4[mk].get("children", []))
        merged_children = sorted(merged_children)
        L4[rep_k]["children"] = merged_children
        L4[rep_k]["descendant_count"] = len(merged_children)

        for mid in members:
            if mid == rep_id:
                continue
            mk = s(mid)
            if mk in L4:
                for cid in L4[mk].get("children", []):
                    ck = s(cid)
                    if ck in L6:
                        L6[ck]["parent"] = rep_id
                del L4[mk]

    for drop_id in list(dropped_small_ids):
        dk = s(drop_id)
        node = L4.get(dk)
        if node is None:
            continue
        for cid in node.get("children", []):
            ck = s(cid)
            if ck in L6 and L6[ck].get("parent", None) == drop_id:
                L6[ck]["parent"] = None
        del L4[dk]

    # Rebuild L2 children
    parent_to_children = defaultdict(list)
    for l4k, node in L4.items():
        pid = node.get("parent", None)
        if pid is not None:
            parent_to_children[int(pid)].append(int(l4k))
    for p2k, p2node in L2.items():
        p2node["children"] = []
        p2node["descendant_count"] = 0
    for pid, kids in parent_to_children.items():
        p2k = s(pid)
        if p2k in L2:
            ks = sorted(set(kids))
            L2[p2k]["children"] = ks
            L2[p2k]["descendant_count"] = len(ks)

    for k, node in L4.items():
        node["descendant_count"] = len(node.get("children", []))

    # Sort
    order = sorted(range(len(survivors_ids)), key=lambda k: survivors_ids[k])
    survivors_ids = [survivors_ids[k] for k in order]
    new_mask_l4 = new_mask_l4[:, order]

    return new_mask_l4, survivors_ids, old_to_rep, dropped_small_ids, merged_groups


# ==================== Level 6 Merge/Drop (from cell: ee66a822) ====================

def merge_drop_level6_within_siblings(
    mask_l6,
    relations,
    iou_thresh,
    area_thresh,
    proposal_ids=None,
):
    """
    Returns:
      new_mask_l6: [P, N6_keep] bool
      survivors_ids: list[int]
      old_to_rep: dict[int,int]
      dropped_small_ids: set[int]
      merged_groups: dict[int, list[int]]
    Mutates relations["hierarchy"]["4"], ["6"].
    proposal_ids: optional list mapping mask columns to existing L6 ids.
    """
    assert mask_l6.dtype == torch.bool
    P, N = mask_l6.shape

    L4 = _safe_get_level(relations, "4")
    L6 = _safe_get_level(relations, "6")

    groups = _parent_groups_for_level(relations, "6")
    id_to_col = None
    if proposal_ids is not None:
        id_to_col = {int(pid): idx for idx, pid in enumerate(proposal_ids)}

    new_cols = []
    survivors_ids = []
    old_to_rep = {}
    merged_groups = {}
    dropped_small_ids = set()

    for parent_id, l6_ids in groups.items():
        if len(l6_ids) == 0:
            continue
        valid_ids, cols, missing_ids = _ids_to_cols(
            l6_ids,
            id_to_col=id_to_col,
            total_cols=N,
            level_label="Level 6"
        )
        if missing_ids:
            for mid in missing_ids:
                try:
                    dropped_small_ids.add(int(mid))
                except (TypeError, ValueError):
                    continue
        if len(cols) == 0:
            continue

        submask = mask_l6[:, cols]
        areas = _areas_from_mask(submask)
        keep = (areas >= area_thresh)
        if keep.sum().item() == 0:
            for pid in valid_ids:
                dropped_small_ids.add(int(pid))
            continue

        keep_flags = keep.tolist()
        kept_ids = [valid_ids[i] for i, k in enumerate(keep_flags) if k]
        kept_cols = [c for c, k in zip(cols, keep_flags) if k]
        dropped_here = [valid_ids[i] for i, k in enumerate(keep_flags) if not k]
        dropped_small_ids.update(dropped_here)

        submask_kept = mask_l6[:, kept_cols]
        iou = _iou_matrix(submask_kept)
        comps = _connected_components_by_thresh(iou, iou_thresh)

        for comp in comps:
            member_ids = [kept_ids[j] for j in comp]
            rep_id = min(member_ids)
            merged_groups[rep_id] = sorted(member_ids)

            col = submask_kept[:, comp[0]].clone()
            for j in comp[1:]:
                col |= submask_kept[:, j]
            new_cols.append(col)
            survivors_ids.append(rep_id)
            for mid in member_ids:
                old_to_rep[mid] = rep_id

    new_mask_l6 = torch.stack(new_cols, dim=1) if len(new_cols) else torch.zeros((P,0), dtype=torch.bool)

    def s(x): return str(int(x))

    # Relations update
    child_rep = {}
    for rep_id, members in merged_groups.items():
        for m in members:
            child_rep[int(m)] = int(rep_id)
    for sid in survivors_ids:
        child_rep.setdefault(int(sid), int(sid))

    for l4k, node in _safe_get_level(relations, "4").items():
        kids = [int(c) for c in node.get("children", [])]
        newkids = []
        for cid in kids:
            if cid in dropped_small_ids:
                continue
            rid = child_rep.get(cid, cid)
            newkids.append(int(rid))
        newkids = sorted(set(newkids))
        node["children"] = newkids
        node["descendant_count"] = len(newkids)

    keep_ids = set(survivors_ids)
    for drop_id in list(dropped_small_ids):
        k = s(drop_id)
        if k in L6:
            del L6[k]
    for rep_id, members in merged_groups.items():
        for mid in members:
            if mid != rep_id:
                mk = s(mid)
                if mk in L6:
                    del L6[mk]
    for sid in survivors_ids:
        sk = s(sid)
        if sk not in L6:
            L6[sk] = {"parent": None, "children": [], "descendant_count": 0}

    # Sort
    order = sorted(range(len(survivors_ids)), key=lambda k: survivors_ids[k])
    survivors_ids = [survivors_ids[k] for k in order]
    new_mask_l6 = new_mask_l6[:, order]

    return new_mask_l6, survivors_ids, old_to_rep, dropped_small_ids, merged_groups


# ==================== Feature Pooling (from cell: 7c698705) ====================

def pool_point_to_proposal_features(pc_features, proposal_masks, mode='average'):
    """
    Aggregate point-level features to proposal-level features.

    Args:
        pc_features: [P, D] point features
        proposal_masks: [P, N] bool or {0,1} masks
        mode: 'average' or 'voting' (default: 'average')
            - 'average': Average point features, then L2 normalize
                Formula: normalize((M.T @ X) / counts)
                Final features are independent of proposal size.
            - 'voting': Sum point features, average by count, NO L2 normalization
                Formula: (M.T @ X) / counts  (no normalize step)
                Matches utils_ov_inference.py line 54-55 behavior.
                Features maintain their original scale/magnitude.

    Returns:
        [N, D] proposal features

    Note:
        The key difference is L2 normalization:
        - 'average' mode: normalizes features to unit length (for cosine similarity)
        - 'voting' mode: keeps features unnormalized (as in utils_ov_inference.py)

        In utils_ov_inference.py, features are NOT normalized because they represent
        averaged class scores, and the magnitude information is meaningful.
    """
    import torch.nn.functional as F

    M = proposal_masks.to(pc_features.device).float()
    X = pc_features

    if mode == 'average':
        # Average then normalize (original behavior)
        # Equivalent to: mean pooling + L2 normalization
        counts = M.sum(dim=0, keepdim=True).clamp_min(1.0)
        proposal_feats = (M.T @ X) / counts.T
        proposal_feats = F.normalize(proposal_feats, dim=1)

    elif mode == 'voting':
        # Sum then average (NO normalization)
        # Matches utils_ov_inference.py behavior:
        # Line 54: einsum("kn,nc->kc") = sum
        # Line 55: divide by instance.sum() = average
        # No L2 normalization!
        counts = M.sum(dim=0, keepdim=True).clamp_min(1.0)
        proposal_feats = (M.T @ X) / counts.T
        # NO F.normalize() here - keep original magnitude

    else:
        raise ValueError(f"Unknown pooling mode: {mode}. Expected 'average' or 'voting'")

    return proposal_feats
