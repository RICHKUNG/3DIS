# Reading SAM2 Masks After the Streaming Refactor

This note targets downstream consumers that relied on the legacy
`np.savez_compressed` bundle (`video_segments.npz`) emitted before 2025-09-26.
The tracking stage now stores per-frame JSON manifests inside a ZIP container,
which breaks direct `np.load(..., allow_pickle=True).item()` calls. To keep
code changes minimal, the tracking package exposes helpers that return the same
frame→object structures you used previously.

## Quick Start

```python
from my3dis.tracking import load_legacy_video_segments
from my3dis.common_utils import unpack_binary_mask

frames, manifest = load_legacy_video_segments(
    "/path/to/level_6/video_segments_scale0.3x.npz",
    unpack_masks=False,  # keep packed bits to match the old workflow
)

frame_idx = 1200
obj_id = 3
payload = frames[frame_idx][obj_id]
mask_bits = payload["packed_mask"]          # uint8 packed bit array
mask_shape = payload["mask_shape"]          # (H, W) at the stored scale
full_res_shape = payload.get("mask_orig_shape")

# When you need a boolean mask (identical to the legacy helper)
mask = unpack_binary_mask(payload)
```

The helper returns:

- `frames`: `Dict[int, Dict[int, Dict[str, Any]]]` with the same packed payload
  (frame index → object id → packed mask dict).
- `manifest`: Raw metadata from `manifest.json` (frame ordering, scale ratio,
  linked object archive, etc.).

Switching existing scripts is typically a one-line change:

```diff
- archive = np.load(video_npz_path, allow_pickle=True)["frames"].item()
+ archive, _ = load_legacy_video_segments(video_npz_path)
```

Keep any downstream logic (mask unpacking, camera projection, fusion) untouched.

## Streaming Access

For large scenes you can iterate without materialising the full dictionary:

```python
from my3dis.tracking import iter_legacy_frames

for legacy_frame in iter_legacy_frames(video_npz_path):
    frame_index = legacy_frame.frame_index
    for obj_id, payload in legacy_frame.objects.items():
        mask = unpack_binary_mask(payload)
        # project mask with your existing routines
```

Pass `frame_filter=[...]` to iterate a subset or `unpack_masks=True` to receive
boolean arrays directly (adds `mask` to each payload but keeps the packed bits
for compatibility).

## Object-Major Lookups

If you previously read `object_segments.npz` to map object ids back to frame
entries, call:

```python
from my3dis.tracking import load_object_segments_manifest

object_map = load_object_segments_manifest(
    "/path/to/level_6/object_segments_scale0.3x.npz"
)
# object_map[obj_id] -> {frame_index: {"frame_entry": "frames/frame_012000.json"}}
```

Combine this with `load_legacy_video_segments` when you need to follow object
trajectories across frames.

## Migration Checklist

1. Replace direct `np.load` calls with `load_legacy_video_segments`.
2. Keep existing `unpack_binary_mask` and camera projection code unchanged.
3. When memory pressure is a concern, switch aggregation loops to
   `iter_legacy_frames`.
4. (Optional) Use `load_object_segments_manifest` if your workflow depended on
   the `object_segments.npz` lookup table.

With these helpers the downstream pipeline preserves the legacy data layout
without reintroducing the high peak RAM that motivated the streaming refactor.
