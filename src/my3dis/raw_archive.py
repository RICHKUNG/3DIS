"""Utilities for chunked persistence of raw SSAM candidates."""

from __future__ import annotations

import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from my3dis.common_utils import RAW_DIR_NAME, RAW_MASK_TEMPLATE, RAW_META_TEMPLATE, ensure_dir

MANIFEST_NAME = "manifest.json"
ARCHIVE_FORMAT_TAG = "my3dis.raw_candidates.v1"


@dataclass
class _PendingFrame:
    frame_idx: int
    frame_name: str
    meta_bytes: bytes
    mask_bytes: Optional[bytes]
    candidate_count: int
    mask_count: int
    meta_entry: str
    mask_entry: Optional[str]


class RawCandidateArchiveWriter:
    """Stream raw SSAM frames into chunked tar archives."""

    def __init__(
        self,
        level_root: Path | str,
        *,
        chunk_size: int = 32,
        compression: str = "gz",
    ) -> None:
        self.raw_dir = Path(level_root) / RAW_DIR_NAME
        ensure_dir(self.raw_dir)
        self.chunk_size = max(1, int(chunk_size))
        self._compression = compression
        self._buffer: List[_PendingFrame] = []
        self._frames: List[Dict[str, object]] = []
        self._chunks: List[Dict[str, object]] = []
        self._chunk_index = 0
        self._manifest_path = self.raw_dir / MANIFEST_NAME
        self._closed = False

    def __enter__(self) -> "RawCandidateArchiveWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def add_frame(
        self,
        *,
        frame_idx: int,
        frame_name: str,
        meta_bytes: bytes,
        mask_bytes: Optional[bytes],
        candidate_count: int,
        mask_count: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("RawCandidateArchiveWriter already closed")

        stem = Path(RAW_META_TEMPLATE.format(frame_idx=frame_idx)).stem
        frame_dir = f"{stem}"
        meta_entry = f"{frame_dir}/meta.json"
        mask_entry = f"{frame_dir}/masks.npz" if mask_bytes else None

        self._buffer.append(
            _PendingFrame(
                frame_idx=frame_idx,
                frame_name=frame_name,
                meta_bytes=meta_bytes,
                mask_bytes=mask_bytes,
                candidate_count=candidate_count,
                mask_count=mask_count,
                meta_entry=meta_entry,
                mask_entry=mask_entry,
            )
        )

        if len(self._buffer) >= self.chunk_size:
            self._flush()

    def close(self) -> Optional[Path]:
        if self._closed:
            return self._manifest_path if self._manifest_path.exists() else None
        try:
            if self._buffer:
                self._flush()
            manifest = {
                "format": ARCHIVE_FORMAT_TAG,
                "chunk_size": self.chunk_size,
                "compression": self._compression,
                "frames": self._frames,
                "chunks": self._chunks,
            }
            with self._manifest_path.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2)
            return self._manifest_path
        finally:
            self._closed = True

    def _make_chunk_name(self, index: int) -> str:
        suffix = ".tar"
        mode = "w"
        if self._compression:
            if self._compression == "gz":
                mode = "w:gz"
                suffix = ".tar.gz"
            elif self._compression == "bz2":
                mode = "w:bz2"
                suffix = ".tar.bz2"
            elif self._compression == "xz":
                mode = "w:xz"
                suffix = ".tar.xz"
            else:
                raise ValueError(f"Unsupported compression={self._compression!r}")
        return f"chunk_{index:04d}{suffix}", mode

    def _flush(self) -> None:
        if not self._buffer:
            return
        chunk_name, mode = self._make_chunk_name(self._chunk_index)
        chunk_path = self.raw_dir / chunk_name
        now = int(time.time())
        with tarfile.open(chunk_path, mode=mode) as tf:
            for pending in self._buffer:
                meta_info = tarfile.TarInfo(name=pending.meta_entry)
                meta_info.size = len(pending.meta_bytes)
                meta_info.mtime = now
                tf.addfile(meta_info, io.BytesIO(pending.meta_bytes))

                if pending.mask_entry and pending.mask_bytes is not None:
                    mask_info = tarfile.TarInfo(name=pending.mask_entry)
                    mask_info.size = len(pending.mask_bytes)
                    mask_info.mtime = now
                    tf.addfile(mask_info, io.BytesIO(pending.mask_bytes))

                frame_record = {
                    "frame_idx": int(pending.frame_idx),
                    "frame_name": pending.frame_name,
                    "candidate_count": int(pending.candidate_count),
                    "mask_count": int(pending.mask_count),
                    "chunk": chunk_name,
                    "meta_entry": pending.meta_entry,
                }
                if pending.mask_entry:
                    frame_record["mask_entry"] = pending.mask_entry
                self._frames.append(frame_record)
        self._chunks.append(
            {
                "name": chunk_name,
                "frame_indices": [int(p.frame_idx) for p in self._buffer],
            }
        )
        self._buffer.clear()
        self._chunk_index += 1


class RawCandidateArchiveReader:
    """Load raw SSAM frames from chunked archives (with legacy fallback)."""

    def __init__(self, level_root: Path | str) -> None:
        self.raw_dir = Path(level_root) / RAW_DIR_NAME
        self.manifest_path = self.raw_dir / MANIFEST_NAME
        self._manifest: Optional[Dict[str, object]] = None
        self._frames_by_idx: Dict[int, Dict[str, object]] = {}
        self._chunks_by_name: Dict[str, Dict[str, object]] = {}
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as fh:
                    manifest = json.load(fh)
            except json.JSONDecodeError:
                manifest = None
            if isinstance(manifest, dict):
                self._manifest = manifest
                frames = manifest.get("frames", [])
                if isinstance(frames, list):
                    for entry in frames:
                        if not isinstance(entry, dict):
                            continue
                        try:
                            frame_idx = int(entry.get("frame_idx"))
                        except (TypeError, ValueError):
                            continue
                        entry["frame_idx"] = frame_idx
                        self._frames_by_idx[frame_idx] = entry
                chunks = manifest.get("chunks", [])
                if isinstance(chunks, list):
                    for record in chunks:
                        if not isinstance(record, dict):
                            continue
                        name = record.get("name")
                        if isinstance(name, str):
                            self._chunks_by_name[name] = record

    def has_manifest(self) -> bool:
        return self._manifest is not None

    def frame_indices(self) -> List[int]:
        if self._manifest is not None:
            return sorted(self._frames_by_idx.keys())
        return self._legacy_frame_indices()

    def load_frame(self, frame_idx: int) -> Optional[Dict[str, object]]:
        if self._manifest is not None:
            entry = self._frames_by_idx.get(int(frame_idx))
            if not entry:
                return None
            chunk_name = entry.get("chunk")
            if not isinstance(chunk_name, str):
                return None
            chunk_path = self.raw_dir / chunk_name
            if not chunk_path.exists():
                return None
            meta_entry = entry.get("meta_entry")
            if not isinstance(meta_entry, str):
                return None
            mask_entry_raw = entry.get("mask_entry")
            mask_entry = mask_entry_raw if isinstance(mask_entry_raw, str) else None
            try:
                with tarfile.open(chunk_path, mode="r:*") as tf:
                    meta_file = tf.extractfile(meta_entry)
                    if meta_file is None:
                        return None
                    meta = json.load(meta_file)
                    mask_stack = None
                    packed_masks = None
                    mask_shape = None
                    has_mask = None
                    if mask_entry:
                        mask_file = tf.extractfile(mask_entry)
                        if mask_file is not None:
                            with np.load(io.BytesIO(mask_file.read())) as npz:
                                if "packed_masks" in npz:
                                    packed_masks = np.asarray(npz["packed_masks"], dtype=np.uint8)
                                    has_mask = np.asarray(npz.get("has_mask"), dtype=bool) if "has_mask" in npz else None
                                    shape_entry = npz.get("mask_shape")
                                    if shape_entry is not None:
                                        mask_shape = tuple(int(v) for v in np.array(shape_entry).tolist())
                                elif "masks" in npz:
                                    mask_stack = np.asarray(npz["masks"], dtype=np.bool_)
                                    has_mask = np.asarray(npz.get("has_mask"), dtype=bool) if "has_mask" in npz else None
                    return {
                        "meta": meta,
                        "mask_stack": mask_stack,
                        "packed_masks": packed_masks,
                        "mask_shape": mask_shape,
                        "has_mask": has_mask,
                    }
            except (tarfile.TarError, json.JSONDecodeError, OSError):
                return None
        return self._legacy_load_frame(frame_idx)

    # Legacy helpers -----------------------------------------------------

    def _legacy_frame_indices(self) -> List[int]:
        if not self.raw_dir.exists():
            return []
        frame_indices: List[int] = []
        for fname in os.listdir(self.raw_dir):
            if not fname.endswith(".json"):
                continue
            stem = fname[:-5]
            if not stem.startswith("frame_"):
                continue
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            try:
                frame_idx = int(parts[1])
            except ValueError:
                continue
            frame_indices.append(frame_idx)
        return sorted(frame_indices)

    def _legacy_load_frame(self, frame_idx: int) -> Optional[Dict[str, object]]:
        meta_path = self.raw_dir / RAW_META_TEMPLATE.format(frame_idx=frame_idx)
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except json.JSONDecodeError:
            return None
        mask_path = self.raw_dir / RAW_MASK_TEMPLATE.format(frame_idx=frame_idx)
        mask_stack = None
        packed_masks = None
        mask_shape = None
        has_mask = None
        if mask_path.exists():
            try:
                with np.load(mask_path) as npz:
                    if "packed_masks" in npz:
                        packed_masks = np.asarray(npz["packed_masks"], dtype=np.uint8)
                        has_mask = np.asarray(npz.get("has_mask"), dtype=bool) if "has_mask" in npz else None
                        shape_entry = npz.get("mask_shape")
                        if shape_entry is not None:
                            mask_shape = tuple(int(v) for v in np.array(shape_entry).tolist())
                    elif "masks" in npz:
                        mask_stack = np.asarray(npz["masks"], dtype=np.bool_)
                        has_mask = np.asarray(npz.get("has_mask"), dtype=bool) if "has_mask" in npz else None
            except OSError:
                return None
        return {
            "meta": meta,
            "mask_stack": mask_stack,
            "packed_masks": packed_masks,
            "mask_shape": mask_shape,
            "has_mask": has_mask,
        }
