import json
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Union

import numpy as np
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class AgentFeatures(BaseModel):
    """Per-observation agent record. 'block' refers to census block group throughout."""

    agent_id: Optional[int] = None
    home_block_id: Optional[int] = None
    home_coord: Optional[tuple[float, ...]] = None
    race: Optional[str] = None
    crime_type: Optional[str] = None
    incident_block_id: Optional[int] = None
    incident_block_coord: Optional[tuple[float, ...]] = None


class BlockFeatures(BaseModel):
    """Census block group features. 'block' is shorthand for block group throughout."""

    block_id: Optional[int] = None
    home_coord: Optional[tuple[float, ...]] = None
    racial_dist: Optional[dict[str, float]] = None
    log_median_income: Optional[float] = None
    log_total_population: Optional[float] = None
    log_total_employees: Optional[float] = None
    log_landsize: Optional[float] = None
    avg_household_size: Optional[float] = None
    home_owners_perc: Optional[float] = None
    underage_perc: Optional[float] = None
    log_attractions: Optional[float] = None
    log_transit_stops: Optional[float] = None
    extra_features: Optional[dict[str, float]] = None


class Estimators(BaseModel):
    distance: Optional[float] = None
    race: Optional[float] = None
    income: Optional[float] = None
    features: Optional[dict[str, float]] = None


def load_data(
    file_path: Union[str, Path],
    model_class: Type[T],
    filter_dict: Optional[dict] = None,
    filter_func_dict: Optional[dict] = None,
) -> list[T]:
    """Load JSONL rows into pydantic models with optional filtering."""
    data = []

    with open(file_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue

            json_data = json.loads(line)

            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    data_value = json_data.get(key)
                    if isinstance(value, list):
                        if data_value not in value:
                            match = False
                            break
                    elif data_value != value:
                        match = False
                        break
                if not match:
                    continue

            instance = model_class(**json_data)

            if filter_func_dict:
                passes = all(
                    func(getattr(instance, key))
                    for key, func in filter_func_dict.items()
                )
                if not passes:
                    continue

            data.append(instance)

    return data


def make_args(
    instances: list[T],
    field_names: Union[list[str], list[Union[str, list[str]]]],
    stack: bool = False,
) -> tuple[np.ndarray, ...]:
    """Extract model fields into numpy arrays, preserving grouped fields."""
    if not instances:
        if stack:
            return (np.array([]),)
        num_outputs = len(field_names)
        return tuple(np.array([]) for _ in range(num_outputs))

    field_groups: list[list[str]] = []
    if stack:
        flat_fields: list[str] = []
        for field in field_names:
            if isinstance(field, list):
                flat_fields.extend(field)
            else:
                flat_fields.append(field)
        field_groups = [flat_fields]
    else:
        for field in field_names:
            field_groups.append(field if isinstance(field, list) else [field])

    arrays = []
    for group in field_groups:
        group_arrays = []
        for field_name in group:
            values = []
            for instance in instances:
                value = getattr(instance, field_name, None)
                if value is None:
                    if "coord" in field_name:
                        value = (np.nan, np.nan)
                    elif "id" in field_name:
                        value = -1
                    else:
                        value = np.nan
                if isinstance(value, tuple):
                    value = list(value)
                values.append(value)

            try:
                array = np.array(values)
            except Exception:
                array = np.array(values, dtype=object)

            if array.ndim == 1:
                array = array[:, None]
            group_arrays.append(array)

        if len(group_arrays) > 1:
            arrays.append(np.concatenate(group_arrays, axis=-1))
            continue

        array = group_arrays[0]
        if not stack and array.shape[-1] == 1 and len(group) == 1:
            array = array.squeeze(-1)
        arrays.append(array)

    return tuple(arrays)


def nonzero_features(
    agents: List[AgentFeatures],
    blocks: List[BlockFeatures],
    feature_names: List[str],
) -> List[AgentFeatures]:
    """Keep agents whose chosen block has at least one non-zero feature."""
    block_map = {
        block.block_id: block for block in blocks if block.block_id is not None
    }

    filtered_agents = []
    for agent in agents:
        if agent.incident_block_id is None:
            continue
        if agent.incident_block_id not in block_map:
            raise ValueError(f"{agent.incident_block_id=} not in block id set")

        block = block_map[agent.incident_block_id]
        has_nonzero = False
        for feature_name in feature_names:
            feature_value = getattr(block, feature_name, None)
            if feature_value is None:
                continue
            if isinstance(feature_value, (int, float)) and not np.isnan(feature_value):
                if feature_value != 0:
                    has_nonzero = True
                    break

        if has_nonzero:
            filtered_agents.append(agent)

    return filtered_agents


class DataConfig(BaseModel):
    data_root: str = "data/features"
    agent: str = "offenders"
    block: str = "blocks"
    agent_filter_dict: Optional[dict] = None
    block_filter_dict: Optional[dict] = None
    filter_nonzero_features: bool = False


class ModelConfig(BaseModel):
    model_type: str = "base"
    distance_interaction: str = "l2_log"
    race_interaction: str = "dissimilarity"
    income_interaction: str = "abs_diff"
    feature_names: list[str] = [
        "log_total_population",
        "log_total_employees",
        "log_landsize",
        "avg_household_size",
        "home_owners_perc",
        "underage_perc",
        "log_attractions",
        "log_transit_stops",
    ]
    include_extra_features: bool = False


class OptimizerConfig(BaseModel):
    chunk_size: int = 16384
    max_iter: int = 2000
    gtol: float = 1e-6
    ftol: float = 1e-9


class Config(BaseModel):
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    output_file: Optional[str] = None
