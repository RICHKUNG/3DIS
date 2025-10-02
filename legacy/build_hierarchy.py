"""
Build per-frame, per-level hierarchy (containment) from filtered candidates.

For each level and frame, we load filtered.json and seg_frame_*.npy, then:
  - For each mask i, find the smallest-area mask j that contains i
    (coverage ratio sum(i & j) / sum(i) >= contain_thr and area(j) > area(i)).
  - Emit edges as parent->child (j -> i). If no parent, node is a root.

Outputs per level:
  - hierarchy/frame_XXXXX.json  (nodes, edges)
  - hierarchy/summary.csv       (level,frame,id,parent,area)
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_filtered(level_root: str):
    filt_dir = os.path.join(level_root, 'filtered')
    with open(os.path.join(filt_dir, 'filtered.json'), 'r') as f:
        meta = json.load(f)
    frames_meta = meta.get('frames', [])
    frames = []
    for fm in frames_meta:
        fidx = fm['frame_idx']
        items = fm['items']
        seg_path = os.path.join(filt_dir, f'seg_frame_{fidx:05d}.npy')
        segs = np.load(seg_path) if os.path.exists(seg_path) else None
        frames.append((fidx, items, segs))
    return frames


def build_containment_tree(items: List[Dict[str, Any]], segs: np.ndarray, contain_thr: float = 0.98):
    # items: list of meta dicts with at least 'id' and 'area'
    # segs: (N, H, W) uint8 masks
    n = len(items)
    if n == 0 or segs is None or len(segs) == 0:
        return {"nodes": [], "edges": []}
    # areas and order
    areas = np.array([int(x.get('area', 0)) for x in items])
    # prepare boolean masks
    seg_bool = segs.astype(bool)
    nodes = []
    for it in items:
        nodes.append({"id": it.get('id'), "area": int(it.get('area', 0)), "parent": None})
    edges = []
    # for each child, find smallest container
    for i in range(n):
        if areas[i] == 0:
            continue
        child = seg_bool[i]
        best_parent = None
        best_area = None
        for j in range(n):
            if j == i or areas[j] <= areas[i]:
                continue
            inter = np.logical_and(child, seg_bool[j]).sum()
            cover = inter / float(areas[i]) if areas[i] > 0 else 0.0
            if cover >= contain_thr:
                if best_parent is None or areas[j] < best_area:
                    best_parent = j
                    best_area = areas[j]
        if best_parent is not None:
            pid = items[best_parent].get('id')
            cid = items[i].get('id')
            nodes[i]['parent'] = pid
            edges.append({"parent": pid, "child": cid})
    return {"nodes": nodes, "edges": edges}


def main():
    ap = argparse.ArgumentParser(description="Build hierarchy (containment) per frame and level")
    ap.add_argument('--candidates-root', required=True, help='Root containing level_*/filtered')
    ap.add_argument('--levels', default='2,4,6')
    ap.add_argument('--contain-thr', type=float, default=0.98)
    args = ap.parse_args()

    levels = [int(x) for x in str(args.levels).split(',') if str(x).strip()]

    for L in levels:
        level_root = os.path.join(args.candidates_root, f'level_{L}')
        frames = load_filtered(level_root)
        out_dir = ensure_dir(os.path.join(level_root, 'hierarchy'))
        rows = ["level,frame,id,parent,area"]
        for fidx, items, segs in frames:
            # align items order with segs order via index 'id'
            # items already assigned 'id' sequentially at generation time
            tree = build_containment_tree(items, segs, contain_thr=args.contain_thr)
            with open(os.path.join(out_dir, f'frame_{fidx:05d}.json'), 'w') as f:
                json.dump(tree, f, indent=2)
            # summary rows
            id_to_area = {it.get('id'): int(it.get('area', 0)) for it in items}
            parent_map = {n['id']: n['parent'] for n in tree['nodes']}
            for nid in parent_map:
                rows.append(f"{L},{fidx},{nid},{parent_map[nid] if parent_map[nid] is not None else ''},{id_to_area.get(nid,0)}")
        with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
            f.write("\n".join(rows) + "\n")


if __name__ == '__main__':
    main()

