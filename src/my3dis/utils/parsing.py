"""Parsing utilities for configuration and data formats."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union


def list_to_csv(values: Iterable[object] | None) -> str:
    """Convert an iterable to a comma-separated string.

    Args:
        values: Iterable of values to join

    Returns:
        Comma-separated string
    """
    if not values:
        return ""
    return ",".join(str(v) for v in values)


def parse_levels(levels: Union[str, Sequence[int], None]) -> List[int]:
    """Parse level input (string or iterable) into a list of ints.

    Args:
        levels: Levels as comma-separated string or sequence

    Returns:
        List of integer levels

    Examples:
        >>> parse_levels("2,4,6")
        [2, 4, 6]
        >>> parse_levels([2, 4, 6])
        [2, 4, 6]
    """
    if levels is None:
        return []
    if isinstance(levels, (list, tuple, set)):
        return [int(v) for v in levels]
    return [int(x) for x in str(levels).split(',') if str(x).strip()]


def parse_range(range_str: Union[str, Sequence[int]]) -> Tuple[int, int, int]:
    """Parse ``start:end:step`` strings or 3-element iterables.

    Args:
        range_str: Range specification as "start:end:step" or (start, end, step)

    Returns:
        Tuple of (start, end, step)

    Raises:
        ValueError: If format is invalid or step is not positive

    Examples:
        >>> parse_range("100:200:10")
        (100, 200, 10)
        >>> parse_range([0, 1000, 20])
        (0, 1000, 20)
    """
    if isinstance(range_str, (list, tuple)):
        if len(range_str) != 3:
            raise ValueError(f"Range iterable must have 3 elements, got {range_str!r}")
        start, end, step = range_str
    else:
        parts = str(range_str).split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid range spec: {range_str!r}")
        start, end, step = parts

    start = int(start) if start != '' else 0
    end = int(end) if end != '' else -1
    step = int(step) if step != '' else 1

    if step <= 0:
        raise ValueError('step must be positive')

    return start, end, step
