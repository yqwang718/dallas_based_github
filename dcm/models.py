from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, vmap

from dcm.interactions import (
    agent_agent_interaction,
    agent_block_interaction,
    block_block_interaction,
)


@partial(
    jax.jit,
    static_argnames=[
        "distance_interaction",
        "race_interaction",
        "income_interaction",
    ],
)
def dcm_model(
    betas,
    agent_home_id,
    agent_coord,
    agent_race_onehot,
    chosen_block_id,
    chosen_block_coord,
    block_coords,
    block_race_dists,
    block_incomes,
    features,
    distance_interaction: str = "l2_log",
    race_interaction: str = "dissimilarity",
    income_interaction: str = "abs_diff",
):
    """Base Dallas DCM: conditional-logit choice over block groups."""
    beta_distance = betas[0]
    beta_race = betas[1]
    beta_income = betas[2]

    num_features = features.shape[1]
    betas_features = betas[3 : 3 + num_features]

    distances = agent_block_interaction(distance_interaction)(agent_coord, block_coords)
    actual_distance = agent_agent_interaction(distance_interaction)(
        agent_coord,
        chosen_block_coord,
    )
    same_home_choice = (agent_home_id == chosen_block_id) & (
        jnp.arange(block_coords.shape[0]) == agent_home_id
    )
    distance = jnp.where(same_home_choice, actual_distance, distances)

    if race_interaction == "threshold":
        race_diss = agent_block_interaction(race_interaction)(
            agent_race_onehot,
            block_race_dists,
        )
    else:
        race_diss = block_block_interaction(race_interaction)(
            agent_home_id,
            block_race_dists,
        )

    income_diss = block_block_interaction(income_interaction)(
        agent_home_id,
        block_incomes,
    )

    logit = (
        beta_distance * distance
        + beta_race * race_diss
        + beta_income * income_diss
        + (betas_features * features).sum(-1)
    )

    logsoftmax = jax.nn.log_softmax(logit)
    return -logsoftmax[chosen_block_id]


def generalized_chunked_sum(model_fn, in_axes, static_argnames=None):
    """Create a chunked sum wrapper for a vmappable per-sample loss."""
    all_static_argnames = ["chunk_size"]
    if static_argnames:
        all_static_argnames.extend(static_argnames)

    @partial(jax.jit, static_argnames=all_static_argnames)
    def chunked_model(*args, chunk_size=1024, **kwargs):
        chunked_indices = [idx for idx, axis in enumerate(in_axes) if axis == 0]
        if not chunked_indices:
            return model_fn(*args, **kwargs)

        model_fn_partial = partial(model_fn, **kwargs)
        first_chunked_idx = chunked_indices[0]
        num_samples = args[first_chunked_idx].shape[0]
        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        padded_size = num_chunks * chunk_size
        pad_size = padded_size - num_samples
        mask = jnp.arange(padded_size) < num_samples

        chunked_args = []
        for idx in chunked_indices:
            arg = args[idx]
            if arg.ndim == 1:
                padded = jnp.pad(arg, (0, pad_size), mode="constant")
                chunked = padded.reshape(num_chunks, chunk_size)
            else:
                pad_width = [(0, pad_size)] + [(0, 0)] * (arg.ndim - 1)
                padded = jnp.pad(arg, pad_width, mode="constant")
                chunked = padded.reshape((num_chunks, chunk_size) + arg.shape[1:])
            chunked_args.append(chunked)

        mask_chunked = mask.reshape(num_chunks, chunk_size)

        @jax.checkpoint
        def scan_fn(total_loss, chunk_data):
            *chunk_values, chunk_mask = chunk_data
            chunk_args_list = list(args)
            for chunk_idx, arg_idx in enumerate(chunked_indices):
                chunk_args_list[arg_idx] = chunk_values[chunk_idx]
            chunk_losses = vmap(model_fn_partial, in_axes)(*chunk_args_list)
            chunk_sum = jnp.sum(chunk_losses * chunk_mask)
            return total_loss + chunk_sum, None

        final_loss, _ = lax.scan(
            scan_fn,
            0.0,
            (*chunked_args, mask_chunked),
        )
        return final_loss

    return chunked_model


dcm_model_chunked_sum = generalized_chunked_sum(
    dcm_model,
    in_axes=(None, 0, 0, 0, 0, 0, None, None, None, None),
    static_argnames=[
        "distance_interaction",
        "race_interaction",
        "income_interaction",
    ],
)


def create_dcm_model_samples():
    """Create a vectorized sample-loss function for the base Dallas model."""

    def dcm_model_samples(
        betas,
        agent_home_ids,
        agent_coords,
        agent_race_onehots,
        chosen_block_ids,
        chosen_block_coords,
        block_coords,
        block_race_dists,
        block_incomes,
        features,
        distance_interaction="l2_log",
        race_interaction="dissimilarity",
        income_interaction="abs_diff",
    ):
        model_fn = partial(
            dcm_model,
            distance_interaction=distance_interaction,
            race_interaction=race_interaction,
            income_interaction=income_interaction,
        )
        return vmap(model_fn, (None, 0, 0, 0, 0, 0, None, None, None, None), 0)(
            betas,
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

    return jax.jit(
        dcm_model_samples,
        static_argnames=[
            "distance_interaction",
            "race_interaction",
            "income_interaction",
        ],
    )


dcm_model_samples = create_dcm_model_samples()
