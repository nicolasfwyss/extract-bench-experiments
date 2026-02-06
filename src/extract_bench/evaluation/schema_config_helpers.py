"""Helper functions for working with evaluation configs on schema nodes."""

from typing import Any, Dict

from ..infra.nodes import Schema
from .evaluation_config import EvaluationConfig
from .presets import (
    ARRAY_DEFAULT_PRESET,
    BOOLEAN_DEFAULT_PRESET,
    INTEGER_DEFAULT_PRESET,
    NUMBER_DEFAULT_PRESET,
    SKIP_PRESET,
    STRING_DEFAULT_PRESET,
    get_preset_config,
    get_preset_configs,
)


def get_evaluation_config(schema: Schema) -> EvaluationConfig:
    """Get evaluation config for a schema node.

    Returns type-specific defaults if not explicitly set.
    Validates and converts the config on access.
    """
    if schema.evaluation_config is not None:
        if isinstance(schema.evaluation_config, str):
            return get_preset_config(schema.evaluation_config)

        if isinstance(schema.evaluation_config, dict):
            try:
                return EvaluationConfig.model_validate(schema.evaluation_config)
            except Exception as e:
                raise ValueError(
                    f"Invalid evaluation_config for schema node: {e}. "
                    f"Expected format: {{'metrics': [...]}} with valid MetricConfig objects."
                ) from e

        return schema.evaluation_config

    return get_default_evaluation_config(schema)


def get_default_evaluation_config(schema: Schema) -> EvaluationConfig:
    """Get type-specific default evaluation config for a schema."""
    schema_type = schema.get_type()
    preset_configs = get_preset_configs()

    match schema_type:
        case "string":
            return preset_configs[STRING_DEFAULT_PRESET]
        case "integer":
            return preset_configs[INTEGER_DEFAULT_PRESET]
        case "number":
            return preset_configs[NUMBER_DEFAULT_PRESET]
        case "boolean":
            return preset_configs[BOOLEAN_DEFAULT_PRESET]
        case "array":
            return preset_configs[ARRAY_DEFAULT_PRESET]
        case "object":
            return preset_configs[SKIP_PRESET]
        case "null":
            return preset_configs[SKIP_PRESET]
        case _:
            return preset_configs[SKIP_PRESET]


def should_evaluate(schema: Schema) -> bool:
    """Check if a schema node should be evaluated."""
    config = get_evaluation_config(schema)
    return len(config.metrics) > 0


def add_evaluation_configs_to_export(
    schema: Schema, include_defaults: bool = True
) -> Dict[str, Any]:
    """Export schema to dict and include evaluation configs."""
    result = schema.export_to_dict()

    if include_defaults:
        _add_config_recursive(result, schema)

    return result


def _add_config_recursive(result_dict: Dict[str, Any], schema: Schema) -> None:
    """Recursively add evaluation configs to exported dict."""
    config = get_evaluation_config(schema)
    if config.metrics:
        result_dict["evaluation_config"] = config.model_dump(exclude_none=True)

    if hasattr(schema, "properties") and schema.properties:
        for key, child_schema in schema.properties.items():
            if "properties" in result_dict and key in result_dict["properties"]:
                _add_config_recursive(result_dict["properties"][key], child_schema)

    if hasattr(schema, "items") and "items" in result_dict:
        _add_config_recursive(result_dict["items"], schema.items)

    if hasattr(schema, "any_of"):
        for i, child_schema in enumerate(schema.any_of):
            if "anyOf" in result_dict and i < len(result_dict["anyOf"]):
                _add_config_recursive(result_dict["anyOf"][i], child_schema)

    if hasattr(schema, "defs") and schema.defs:
        for key, def_schema in schema.defs.items():
            if "$defs" in result_dict and key in result_dict["$defs"]:
                _add_config_recursive(result_dict["$defs"][key], def_schema)
