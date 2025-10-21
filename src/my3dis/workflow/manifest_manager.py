"""Manifest file management with automatic save and context manager support.

This module provides a cleaner interface for reading/writing manifest.json files,
reducing I/O operations and eliminating manual file handling code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Manifest:
    """Manifest data container with automatic save tracking.

    Provides dict-like access to manifest data while tracking modifications
    and supporting automatic save via context manager.

    Example:
        # Manual save
        manifest = Manifest.load(run_dir)
        manifest.set('tracking', {...})
        manifest.save()

        # Automatic save with context manager
        with Manifest.load(run_dir) as manifest:
            manifest.set('tracking', {...})
        # Automatically saved on exit

    Attributes:
        run_dir: Directory containing manifest.json
        data: Manifest data dictionary
        _modified: Internal flag tracking whether data has been modified
    """

    run_dir: Path
    data: Dict[str, Any] = field(default_factory=dict)
    _modified: bool = field(default=False, init=False, repr=False)

    @classmethod
    def load(cls, run_dir: Path) -> Manifest:
        """Load manifest from run directory, creating empty if not found.

        Args:
            run_dir: Directory containing manifest.json

        Returns:
            Manifest object with loaded or empty data
        """
        run_dir = Path(run_dir)
        manifest_path = run_dir / 'manifest.json'

        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                import logging
                logging.warning(
                    'Failed to load manifest from %s: %s. Using empty manifest.',
                    manifest_path, exc
                )
                data = {}
        else:
            data = {}

        return cls(run_dir=run_dir, data=data)

    def save(self, force: bool = False) -> None:
        """Save manifest to disk if modified.

        Args:
            force: If True, save even if not modified
        """
        if not self._modified and not force:
            return

        manifest_path = self.run_dir / 'manifest.json'
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(manifest_path, 'w') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            import logging
            logging.error('Failed to save manifest to %s: %s', manifest_path, exc)
            raise

        self._modified = False

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from manifest data.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value for key, or default if not found
        """
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in manifest data and mark as modified.

        Args:
            key: Key to set
            value: Value to set
        """
        self.data[key] = value
        self._modified = True

    def update(self, updates: Dict[str, Any]) -> None:
        """Batch update manifest data and mark as modified.

        Args:
            updates: Dictionary of updates to apply
        """
        self.data.update(updates)
        self._modified = True

    # Convenience properties for common manifest fields

    @property
    def levels(self) -> Optional[List[int]]:
        """Get levels list from manifest."""
        return self.data.get('levels')

    @property
    def frames(self) -> Optional[Dict[str, Any]]:
        """Get frames metadata from manifest."""
        return self.data.get('frames')

    @property
    def mask_scale_ratio(self) -> float:
        """Get mask scale ratio, default to 1.0 if not set."""
        try:
            return float(self.data.get('mask_scale_ratio', 1.0))
        except (TypeError, ValueError):
            return 1.0

    # Context manager support for automatic save

    def __enter__(self) -> Manifest:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, automatically save if no exception."""
        if exc_type is None:
            self.save()
