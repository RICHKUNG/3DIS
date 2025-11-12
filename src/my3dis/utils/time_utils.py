"""Time formatting utilities for My3DIS."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Render duration as mm:ss or HH:MM:SS when hours present.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string

    Examples:
        >>> format_duration(45)
        '00:45'
        >>> format_duration(125)
        '02:05'
        >>> format_duration(3665)
        '01:01:05'
    """
    total_seconds = max(0.0, float(seconds))
    rounded = int(round(total_seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
