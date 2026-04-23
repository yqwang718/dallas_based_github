import jax
import jax.numpy as jnp
from typing import Callable, Dict

# Unary function table - functions taking one jnp array
UNARY_FUNCTIONS: Dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "identity": lambda a: a,
    "log": lambda a: jnp.log(a),
    "exp": lambda a: jnp.exp(a),
    "squared": lambda a: a**2,
}

# Combined function table - functions taking two jnp arrays
FUNCTIONS: Dict[str, Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = {
    # Scalar operations
    "dummy": lambda a, b: b,
    "diff": lambda a, b: b - a,
    "abs_diff": lambda a, b: jnp.abs(b - a),
    "product": lambda a, b: a * b,
    "product_diff": lambda a, b: b * (b - a),
    # Vector operations
    "l1": lambda a, b: jnp.sum(jnp.abs(a - b), axis=-1),
    "l2": lambda a, b: jnp.sqrt(jnp.sum((a - b) ** 2, axis=-1))
    / 1000,  # Convert meters to km
    "l2_log": lambda a, b: 0.5
    * jnp.log(jnp.maximum(jnp.sum((a - b) ** 2, axis=-1), 1e-8)),
    "cosine": lambda a, b: 1.0
    - (
        jnp.sum(a * b, axis=-1)
        / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1))
    ),
    "dissimilarity": lambda a, b: 1.0 - jnp.sum(a * b, axis=-1),
    "threshold": lambda a, b: 1.0
    - jnp.sum(jnp.where(a > 0.75, 1.0, 0.0) * jnp.where(b > 0.75, 1.0, 0.0), axis=-1),
}


def feature(key: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if key not in UNARY_FUNCTIONS:
        raise ValueError(f"Unknown function key: {key}")
    func = UNARY_FUNCTIONS[key]
    return jax.jit(func)


def block_block_interaction(key: str) -> Callable[[int, jnp.ndarray], jnp.ndarray]:
    if key not in FUNCTIONS:
        raise ValueError(f"Unknown function key: {key}")
    func = FUNCTIONS[key]

    def inner_function(agent_block_id: int, block_features: jnp.ndarray) -> jnp.ndarray:
        return func(block_features[agent_block_id], block_features)

    return jax.jit(inner_function)


def agent_block_interaction(
    key: str,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    if key not in FUNCTIONS:
        raise ValueError(f"Unknown function key: {key}")
    func = FUNCTIONS[key]

    def inner_function(
        agent_feature: jnp.ndarray, block_features: jnp.ndarray
    ) -> jnp.ndarray:
        return func(agent_feature, block_features)

    return jax.jit(inner_function)


def agent_agent_interaction(
    key: str,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    if key not in FUNCTIONS:
        raise ValueError(f"Unknown function key: {key}")
    func = FUNCTIONS[key]

    def inner_function(
        agent_feature_1: jnp.ndarray, agent_feature_2: jnp.ndarray
    ) -> jnp.ndarray:
        return func(agent_feature_1, agent_feature_2)

    return jax.jit(inner_function)
