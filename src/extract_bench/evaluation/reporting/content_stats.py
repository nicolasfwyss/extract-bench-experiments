"""Content statistics collector for gold/extracted JSON."""

from collections import defaultdict
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

from ...infra.nodes import ObjectSchema, RootSchema
from .models import ArrayStats, ContentStats, CoverageStats


def _get_value_type(value: Any) -> str:
    """Map Python type to schema type name."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _walk_json(
    data: Any,
    path: str = "",
) -> Tuple[int, Dict[str, int], List[int]]:
    """Recursively walk JSON and collect stats.

    Returns:
        Tuple of (key_count, type_counts, array_lengths)
    """
    key_count = 0
    type_counts: Dict[str, int] = defaultdict(int)
    array_lengths: List[int] = []

    if isinstance(data, dict):
        for key, value in data.items():
            key_count += 1
            value_type = _get_value_type(value)
            type_counts[value_type] += 1

            child_path = f"{path}.{key}" if path else key
            child_keys, child_types, child_arrays = _walk_json(value, child_path)
            key_count += child_keys
            for t, c in child_types.items():
                type_counts[t] += c
            array_lengths.extend(child_arrays)

    elif isinstance(data, list):
        array_lengths.append(len(data))
        for i, item in enumerate(data):
            child_path = f"{path}[{i}]"
            child_keys, child_types, child_arrays = _walk_json(item, child_path)
            key_count += child_keys
            for t, c in child_types.items():
                type_counts[t] += c
            array_lengths.extend(child_arrays)

    return key_count, dict(type_counts), array_lengths


def collect_content_stats(data: dict, label: str) -> ContentStats:
    """Collect content statistics from a JSON dict."""
    key_count, type_counts, array_lengths = _walk_json(data)

    array_stats: Optional[ArrayStats] = None
    if array_lengths:
        array_stats = ArrayStats(
            array_field_count=len(array_lengths),
            total_items=sum(array_lengths),
            min_length=min(array_lengths),
            median_length=median(array_lengths),
            max_length=max(array_lengths),
        )

    return ContentStats(
        label=label,
        total_keys=key_count,
        counts_by_type=type_counts,
        array_stats=array_stats,
        null_count=type_counts.get("null", 0),
        missing_count=0,
    )


def _collect_paths(data: Any, prefix: str = "") -> Set[str]:
    """Collect all leaf paths from a JSON structure."""
    paths: Set[str] = set()

    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)) and value:
                paths.update(_collect_paths(value, path))
            else:
                paths.add(path)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            paths.update(_collect_paths(item, f"{prefix}[{i}]"))
    else:
        if prefix:
            paths.add(prefix)

    return paths


def _collect_required_paths(
    schema: ObjectSchema | RootSchema,
    prefix: str = "",
) -> Set[str]:
    """Collect paths of required fields from schema."""
    required_paths: Set[str] = set()

    if not hasattr(schema, "properties") or not schema.properties:
        return required_paths

    required_set = set(schema.required or [])

    for key, child_schema in schema.properties.items():
        path = f"{prefix}.{key}" if prefix else key
        if key in required_set:
            required_paths.add(path)

        if isinstance(child_schema, (ObjectSchema,)):
            required_paths.update(_collect_required_paths(child_schema, path))

    return required_paths


def compute_coverage(
    gold: dict,
    extracted: dict,
    schema: RootSchema,
) -> CoverageStats:
    """Compute coverage statistics comparing gold vs extracted."""
    gold_paths = _collect_paths(gold)
    extracted_paths = _collect_paths(extracted)
    required_paths = _collect_required_paths(schema)

    present_in_both = gold_paths & extracted_paths
    missing_in_extracted = gold_paths - extracted_paths
    spurious_in_extracted = extracted_paths - gold_paths

    required_missing = missing_in_extracted & required_paths

    return CoverageStats(
        present_in_both=len(present_in_both),
        missing_in_extracted=len(missing_in_extracted),
        spurious_in_extracted=len(spurious_in_extracted),
        required_missing=len(required_missing),
        missing_paths=sorted(missing_in_extracted),
        spurious_paths=sorted(spurious_in_extracted),
    )
