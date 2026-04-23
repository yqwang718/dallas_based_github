import argparse
import json
import logging
from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.optimize import minimize as scipy_minimize

from dcm.mle_utils import calculate_bic, calculate_se
from dcm.models import dcm_model_chunked_sum, dcm_model_samples
from dcm.protocols import (
    AgentFeatures,
    BlockFeatures,
    Config,
    Estimators,
    load_data,
    make_args,
    nonzero_features,
)

logger = logging.getLogger(__name__)

CRIME_TYPES = [
    "burglary_breaking_entering",
    "motor_vehicle_theft",
    "larceny_theft_offenses",
    "assault_offenses",
    "robbery",
    "drug_narcotic_violations",
]

VICTIM_CRIME_TYPES = [
    "motor_vehicle_theft",
    "larceny_theft_offenses",
    "assault_offenses",
    "robbery",
]


def extract_race_income_data(
    blocks: List[BlockFeatures],
    agents: List[AgentFeatures],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Extract race distributions, agent one-hots, and block incomes."""
    race_dists = []
    incomes = []

    all_race_keys = set()
    for block in blocks:
        if block.racial_dist:
            all_race_keys.update(block.racial_dist.keys())
    race_order = sorted(all_race_keys)
    if not race_order:
        raise ValueError("No racial distribution data found in any block")

    for index, block in enumerate(blocks):
        if not block.racial_dist:
            raise ValueError(
                f"Block at index {index} (id: {getattr(block, 'block_id', 'unknown')}) "
                "is missing racial_dist data"
            )
        if block.log_median_income is None:
            raise ValueError(
                f"Block at index {index} (id: {getattr(block, 'block_id', 'unknown')}) "
                "is missing log_median_income data"
            )

        race_dists.append([block.racial_dist.get(race, 0.0) for race in race_order])
        incomes.append(block.log_median_income)

    race_agent_onehots = []
    for index, agent in enumerate(agents):
        if not agent.race:
            raise ValueError(
                f"Agent at index {index} (id: {getattr(agent, 'agent_id', 'unknown')}) "
                "is missing race data"
            )
        race_agent_onehots.append(
            [1.0 if race == agent.race else 0.0 for race in race_order]
        )

    return (jnp.array(race_dists), jnp.array(race_agent_onehots)), jnp.array(incomes)


def extract_extra_features(blocks: List[BlockFeatures]) -> Tuple[jnp.ndarray, List[str]]:
    """Extract optional extra block features when present."""
    if not blocks or not blocks[0].extra_features:
        return jnp.array([]).reshape(len(blocks), 0), []

    extra_feature_names = sorted(blocks[0].extra_features.keys())
    extra_features = []
    for index, block in enumerate(blocks):
        if not block.extra_features:
            raise ValueError(
                f"Block at index {index} has no extra_features, but first block "
                f"has keys: {extra_feature_names}"
            )

        block_keys = set(block.extra_features.keys())
        expected_keys = set(extra_feature_names)
        if block_keys != expected_keys:
            raise ValueError(
                f"Block at index {index} has different extra_features keys. "
                f"Expected: {expected_keys}, Got: {block_keys}"
            )

        extra_features.append([block.extra_features[key] for key in extra_feature_names])

    return jnp.array(extra_features), extra_feature_names


def prepare_base_data(
    agents: List[AgentFeatures],
    blocks: List[BlockFeatures],
    feature_names: List[str],
    include_extra_features: bool = False,
) -> tuple:
    """Prepare arrays for the base Dallas DCM."""
    block_coords = make_args(blocks, ["home_coord"])[0]
    if feature_names:
        features = make_args(blocks, feature_names, stack=True)[0]
    else:
        features = np.zeros((len(blocks), 0), dtype=float)

    returned_feature_names = feature_names.copy()
    if include_extra_features:
        extra_features, extra_feature_names = extract_extra_features(blocks)
        if extra_feature_names:
            features = np.concatenate([features, np.array(extra_features)], axis=1)
            returned_feature_names.extend(extra_feature_names)
            logger.info(
                "Added %s extra features: %s",
                len(extra_feature_names),
                extra_feature_names,
            )

    agent_home_ids = make_args(agents, ["home_block_id"])[0]
    agent_coords = make_args(agents, ["home_coord"])[0]
    chosen_block_ids = make_args(agents, ["incident_block_id"])[0]
    chosen_block_coords = make_args(agents, ["incident_block_coord"])[0]
    (block_race_dists, agent_race_onehots), block_incomes = extract_race_income_data(
        blocks,
        agents,
    )

    return (
        jnp.array(agent_coords, dtype=jnp.float32),
        jnp.array(block_coords, dtype=jnp.float32),
        block_race_dists,
        agent_race_onehots,
        block_incomes,
        jnp.array(features, dtype=jnp.float32),
        jnp.array(agent_home_ids, dtype=jnp.int32),
        jnp.array(chosen_block_ids, dtype=jnp.int32),
        jnp.array(chosen_block_coords, dtype=jnp.float32),
        returned_feature_names,
    )


def to_estimators(
    params: jnp.ndarray,
    feature_names: List[str],
) -> Estimators:
    """Convert a parameter vector into the estimator JSON shape."""
    beta_distance = float(params[0])
    beta_race = float(params[1])
    beta_income = float(params[2])

    betas_features = params[3 : 3 + len(feature_names)]
    features = {
        name: float(betas_features[idx]) for idx, name in enumerate(feature_names)
    }

    return Estimators(
        distance=beta_distance,
        race=beta_race,
        income=beta_income,
        features=features if features else None,
    )


def optimize_dcm_model(
    agent_features: List[AgentFeatures],
    block_features: List[BlockFeatures],
    config: Config,
) -> Tuple[Estimators, Estimators, float, bool, float]:
    """Run base-model DCM optimization and compute SEs and BIC."""
    if config.model.model_type != "base":
        raise ValueError(
            "This replication package only supports model_type='base'. "
            f"Received {config.model.model_type!r}."
        )

    feature_names = config.model.feature_names
    chunk_size = config.optimizer.chunk_size
    max_iter = config.optimizer.max_iter
    gtol = config.optimizer.gtol
    ftol = config.optimizer.ftol
    distance_interaction = config.model.distance_interaction
    race_interaction = config.model.race_interaction
    income_interaction = config.model.income_interaction
    include_extra_features = config.model.include_extra_features

    (
        agent_coords,
        block_coords,
        block_race_dists,
        agent_race_onehots,
        block_incomes,
        features,
        agent_home_ids,
        chosen_block_ids,
        chosen_block_coords,
        returned_feature_names,
    ) = prepare_base_data(
        agent_features,
        block_features,
        feature_names,
        include_extra_features=include_extra_features,
    )

    key = jax.random.PRNGKey(42)
    num_features = features.shape[1]
    initial_params = jax.random.normal(key, (3 + num_features,)) * 0.1

    def objective(params):
        model_inputs = (
            params,
            agent_home_ids,
            agent_coords,
            agent_race_onehots,
            chosen_block_ids,
            chosen_block_coords,
            block_coords,
            block_race_dists,
            block_incomes,
            features,
        )
        return (
            dcm_model_chunked_sum(
                *model_inputs,
                chunk_size=chunk_size,
                distance_interaction=distance_interaction,
                race_interaction=race_interaction,
                income_interaction=income_interaction,
            )
            / agent_coords.shape[0]
        )

    logger.info("Running optimization with scipy L-BFGS-B...")

    def scipy_objective(params_np):
        return float(objective(jnp.array(params_np)))

    def scipy_gradient(params_np):
        return np.array(jax.grad(objective)(jnp.array(params_np)))

    scipy_result = scipy_minimize(
        scipy_objective,
        np.array(initial_params),
        method="L-BFGS-B",
        jac=scipy_gradient,
        options={
            "maxiter": max_iter,
            "gtol": gtol,
            "ftol": ftol,
        },
    )

    params_opt = jnp.array(scipy_result.x)

    model_fn_partial = partial(
        dcm_model_samples,
        distance_interaction=distance_interaction,
        race_interaction=race_interaction,
        income_interaction=income_interaction,
    )
    se_args = (
        agent_home_ids,
        agent_coords,
        agent_race_onehots,
        chosen_block_ids,
        chosen_block_coords,
        block_coords,
        block_race_dists,
        block_incomes,
        features,
    )

    logger.info("Computing standard errors...")
    se = calculate_se(model_fn_partial, params_opt, se_args, chunk_size)

    logger.info("Computing BIC...")
    bic = calculate_bic(model_fn_partial, params_opt, se_args, chunk_size)

    estimators = to_estimators(params_opt, returned_feature_names)
    standard_errors = to_estimators(se, returned_feature_names)

    return (
        estimators,
        standard_errors,
        float(scipy_result.fun),
        bool(scipy_result.success),
        bic,
    )


def load_config(config_path: str) -> Config:
    """Load YAML config into the base replication Config model."""
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config_data = yaml.safe_load(file_obj)
    return Config(**config_data)


def determine_analyses(config: Config) -> Tuple[list[tuple[str, object]], list[str], list[str]]:
    """Determine the crime-type analyses to run for the base workflow."""
    agent_filter_dict = config.data.agent_filter_dict or {}
    if "crime_type" in agent_filter_dict:
        crime_label = agent_filter_dict["crime_type"]
        return [(crime_label, None)], [crime_label], [crime_label]

    individual_crime_types = (
        VICTIM_CRIME_TYPES if config.data.agent.startswith("victims") else CRIME_TYPES
    )
    analyses = [(crime_type, crime_type) for crime_type in individual_crime_types]
    analyses.append(("all_crime_types", individual_crime_types))
    return analyses, individual_crime_types, individual_crime_types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the base Dallas DCM replication pipeline.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    analyses, individual_crime_types, all_crime_types_pooled = determine_analyses(
        config
    )

    agent_file = f"{config.data.data_root}/{config.data.agent}.jsonl"
    block_file = f"{config.data.data_root}/{config.data.block}.jsonl"
    all_results = {}

    for label, crime_filter in analyses:
        logger.info("\n%s", "=" * 60)
        logger.info("Running analysis for: %s", label)
        logger.info("%s", "=" * 60)

        agent_filter_dict = (config.data.agent_filter_dict or {}).copy()
        if crime_filter is not None:
            agent_filter_dict["crime_type"] = crime_filter

        agents = load_data(agent_file, AgentFeatures, agent_filter_dict)
        logger.info("Loaded %s %s for %s", len(agents), config.data.agent, label)
        if not agents:
            logger.warning("No agents found for %s, skipping...", label)
            continue

        logger.info("Loading block features...")
        blocks = load_data(block_file, BlockFeatures, config.data.block_filter_dict)
        logger.info("Loaded %s blocks (class=BlockFeatures)", len(blocks))

        if config.data.filter_nonzero_features:
            original_count = len(agents)
            agents = nonzero_features(agents, blocks, config.model.feature_names)
            logger.info(
                "Filtered agents based on non-zero features: %s -> %s (%.1f%% retained)",
                original_count,
                len(agents),
                len(agents) / original_count * 100,
            )
            if not agents:
                logger.warning(
                    "No agents remain after non-zero feature filtering for %s, skipping...",
                    label,
                )
                continue

        try:
            estimators, standard_errors, final_loss, converged, bic = optimize_dcm_model(
                agents,
                blocks,
                config,
            )
            result_entry = {
                "estimators": estimators.model_dump(),
                "standard_errors": standard_errors.model_dump(),
                "final_loss": float(final_loss),
                "converged": bool(converged),
                "bic": float(bic),
                "num_agents": len(agents),
            }
            if label == "all_crime_types":
                result_entry["crime_types_included"] = all_crime_types_pooled
            all_results[label] = result_entry
            logger.info("Completed analysis for %s", label)
        except Exception as exc:
            logger.error("Error processing %s: %s", label, str(exc))
            all_results[label] = {"error": str(exc)}

    final_results = {
        "metadata": {
            "model_type": config.model.model_type,
            "distance_interaction": config.model.distance_interaction,
            "race_interaction": config.model.race_interaction,
            "income_interaction": config.model.income_interaction,
            "config_used": config.model.model_dump(),
            "individual_crime_types_analyzed": individual_crime_types,
            "all_crime_types_pooled": all_crime_types_pooled,
            "replication_scope": "base_only",
        },
        "results": all_results,
    }

    if config.output_file:
        with open(config.output_file, "w", encoding="utf-8") as file_obj:
            json.dump(final_results, file_obj, indent=2)
        logger.info("\nAll results saved to %s", config.output_file)

        logger.info("\n%s", "=" * 60)
        logger.info("SUMMARY OF RESULTS")
        logger.info("%s", "=" * 60)
        for label, result in all_results.items():
            if "error" in result:
                logger.info("%s: ERROR - %s", label, result["error"])
                continue

            logger.info(
                "%s: n=%s, converged=%s, BIC=%.2f",
                label,
                result.get("num_agents", "N/A"),
                result.get("converged", "N/A"),
                result.get("bic", float("nan")),
            )
        return

    logger.info("No output file specified - results not saved")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
