import logging

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit

logger = logging.getLogger(__name__)


def loss_sum(func):
    """Create a jitted function that sums the loss over all samples."""
    return jit(lambda betas, args: func(betas, *args).sum())


def loss_hess(func):
    """Create a jitted function that computes the Hessian of the sum of losses."""
    return jit(lambda betas, args: jacfwd(jacrev(loss_sum(func)))(betas, args))


def _chunk_args(args: tuple, chunk_indices: jnp.ndarray, n: int) -> tuple:
    """Slice sample-major arrays down to the given chunk indices."""
    chunked = []
    for arg in args:
        if isinstance(arg, jnp.ndarray) and arg.shape[0] == n:
            chunked.append(arg[chunk_indices])
        else:
            chunked.append(arg)
    return tuple(chunked)


def calculate_se(
    model_fn,
    params: jnp.ndarray,
    args: tuple,
    chunk_size: int = 16384,
) -> jnp.ndarray:
    """Calculate standard errors for parameters using the Hessian.

    Args:
        model_fn: The model function (e.g. dcm_model_samples)
        params: Optimized parameters
        args: Model arguments tuple
        chunk_size: Size of chunks for processing

    Returns:
        Array of standard errors
    """
    n = args[0].shape[0] if hasattr(args[0], "shape") else len(args[0])
    d = params.shape[0]

    if n <= chunk_size:
        hess = loss_hess(model_fn)(params, args)
    else:
        hess = jnp.zeros((d, d))
        num_chunks = (n + chunk_size - 1) // chunk_size
        indices = jnp.arange(n)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            chunk_indices = indices[start_idx:end_idx]
            hess += loss_hess(model_fn)(params, _chunk_args(args, chunk_indices, n))

    logger.info(f"Hessian computed: shape {hess.shape}")

    # Move to CPU for numerically stable matrix inversion.
    hess_cpu = np.array(hess)
    inv_hess_cpu = np.linalg.inv(hess_cpu)
    se_cpu = np.sqrt(np.diag(inv_hess_cpu))
    return jnp.array(se_cpu)


def calculate_bic(
    model_fn,
    params: jnp.ndarray,
    args: tuple,
    chunk_size: int = 16384,
) -> float:
    """Calculate BIC (Bayesian Information Criterion).

    Args:
        model_fn: The model function
        params: Optimized parameters
        args: Model arguments tuple
        chunk_size: Size of chunks for processing

    Returns:
        BIC value
    """
    n = args[0].shape[0] if hasattr(args[0], "shape") else len(args[0])
    d = params.shape[0]

    if n <= chunk_size:
        loss_total = loss_sum(model_fn)(params, args)
    else:
        loss_total = 0.0
        num_chunks = (n + chunk_size - 1) // chunk_size
        indices = jnp.arange(n)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)
            chunk_indices = indices[start_idx:end_idx]
            loss_total += loss_sum(model_fn)(params, _chunk_args(args, chunk_indices, n))

    bic = d * jnp.log(n) + 2 * loss_total
    return float(bic)
