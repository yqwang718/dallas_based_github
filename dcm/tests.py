import jax
import jax.numpy as jnp
import pytest
from jax import grad
from jax.scipy.optimize import minimize

from dcm.interactions import block_block_interaction
from dcm.models import dcm_model_chunked_sum, dcm_model_samples


def create_inputs(num_samples, num_blocks, num_features=128, num_races=16):
    """Create test inputs for DCM model functions.

    Args:
        num_samples (int): Number of agent samples (N)
        num_blocks (int): Number of blocks/choices (C)
        num_features (int): Number of features (F), default 128
        num_races (int): Number of race categories (R), default 16

    Returns:
        tuple: Tuple containing all model inputs in the correct order for dcm_model
    """
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 11)

    beta_distance = jax.random.normal(keys[0], ()) * 0.1
    beta_race = jax.random.normal(keys[1], ()) * 0.1
    beta_income = jax.random.normal(keys[2], ()) * 0.1
    betas_features = jax.random.normal(keys[7], (num_features,)) * 0.01

    betas = jnp.concatenate(
        [jnp.array([beta_distance, beta_race, beta_income]), betas_features]
    )

    block_coords = jax.random.normal(keys[3], (num_blocks, 2)) * 10.0
    block_race_dists = jax.random.dirichlet(keys[4], jnp.ones(num_races), (num_blocks,))
    log_block_incomes = jax.random.exponential(keys[5], (num_blocks,)) * 10
    features = jax.random.normal(keys[6], (num_blocks, num_features))

    agent_coord = jax.random.normal(keys[8], (num_samples, 2)) * 10.0
    agent_home_id = jax.random.randint(keys[9], (num_samples,), 0, num_blocks)

    agent_race_idx = jax.random.randint(keys[10], (num_samples,), 0, num_races)
    agent_race_onehot = jax.nn.one_hot(agent_race_idx, num_races)

    choice_key = jax.random.split(keys[0], num_samples)
    chosen_block_id = jax.random.randint(choice_key[0], (num_samples,), 0, num_blocks)

    chosen_block_coord = block_coords[chosen_block_id]

    return (
        betas,  # (3 + F)
        agent_home_id,  # (N)
        agent_coord,  # (N, 2)
        agent_race_onehot,  # (N, R)
        chosen_block_id,  # (N)
        chosen_block_coord,  # (N, 2)
        block_coords,  # (C, 2)
        block_race_dists,  # (C, R)
        log_block_incomes,  # (C)
        features,  # (C, F)
    )


def test_chunked_vs_samples_consistency():
    """Test that dcm_model_chunked_sum and summed dcm_model_samples give similar results."""
    inputs = create_inputs(num_samples=10000, num_blocks=2000)

    chunked_result = dcm_model_chunked_sum(*inputs)
    samples_result = jnp.sum(dcm_model_samples(*inputs))

    relative_diff = jnp.abs((chunked_result - samples_result) / samples_result)
    assert (
        relative_diff < 0.01
    ), f"Results differ by {relative_diff:.4f}, expected < 0.01"

    print(f"Chunked result: {chunked_result:.6f}")
    print(f"Samples result: {samples_result:.6f}")
    print(f"Relative difference: {relative_diff:.6f}")


def test_chunked_vs_samples_gradient_consistency():
    """Test that gradients from dcm_model_chunked_sum and dcm_model_samples are consistent."""
    inputs = create_inputs(num_samples=10000, num_blocks=2000)

    def chunked_objective(betas):
        modified_inputs = (betas,) + inputs[1:]
        return dcm_model_chunked_sum(*modified_inputs)

    def samples_objective(betas):
        modified_inputs = (betas,) + inputs[1:]
        return jnp.sum(dcm_model_samples(*modified_inputs))

    chunked_grad_fn = grad(chunked_objective)
    samples_grad_fn = grad(samples_objective)

    chunked_gradient = chunked_grad_fn(inputs[0])
    samples_gradient = samples_grad_fn(inputs[0])

    gradient_diff = chunked_gradient - samples_gradient
    diff_magnitude = jnp.linalg.norm(gradient_diff)
    samples_magnitude = jnp.linalg.norm(samples_gradient)
    relative_diff = diff_magnitude / samples_magnitude

    if relative_diff >= 0.01:
        print(f"\nChunked gradient: {chunked_gradient}")
        print(f"\nSamples gradient: {samples_gradient}")

    assert (
        relative_diff < 0.01
    ), f"Relative gradient vector difference: {relative_diff:.4f}, expected < 0.01"

    print(f"Chunked gradient norm: {jnp.linalg.norm(chunked_gradient):.6f}")
    print(f"Samples gradient norm: {samples_magnitude:.6f}")
    print(f"Gradient difference norm: {diff_magnitude:.6e}")
    print(f"Relative gradient vector difference: {relative_diff:.6f}")


def test_large_scale_load():
    """Load test with 1M samples and 10K choices to verify it runs without memory issues."""
    inputs = create_inputs(num_samples=1000000, num_blocks=10000)

    result = dcm_model_chunked_sum(*inputs)

    assert jnp.isfinite(result), "Result should be finite"
    assert result > 0, "Loss should be positive"

    print(f"Large scale result: {result:.6f}")


def test_large_scale_gradients():
    """Test gradient computation on large scale with 1M samples and 10K choices."""
    inputs = create_inputs(num_samples=1000000, num_blocks=10000)

    def objective(betas):
        modified_inputs = (betas,) + inputs[1:]
        return dcm_model_chunked_sum(*modified_inputs)

    grad_fn = grad(objective)
    gradient = grad_fn(inputs[0])

    assert gradient.shape == inputs[0].shape, "Gradient shape mismatch"
    assert jnp.all(jnp.isfinite(gradient)), "Gradient should be finite"
    print(f"Gradient norm: {jnp.linalg.norm(gradient):.6f}")


def test_large_scale_optimize():
    """Test full parameter optimization using JAX scipy.optimize.minimize.

    This test optimizes all beta parameters in a single array.
    """
    inputs = create_inputs(num_samples=100000, num_blocks=5000)

    agent_home_id = inputs[1]
    agent_coord = inputs[2]
    agent_race_onehot = inputs[3]
    chosen_block_id = inputs[4]
    chosen_block_coord = inputs[5]
    block_coords = inputs[6]
    block_race_dists = inputs[7]
    block_incomes = inputs[8]
    features = inputs[9]

    initial_params = inputs[0]

    def objective(params):
        model_inputs = (
            params,
            agent_home_id,
            agent_coord,
            agent_race_onehot,
            chosen_block_id,
            chosen_block_coord,
            block_coords,
            block_race_dists,
            block_incomes,
            features,
        )
        return dcm_model_chunked_sum(*model_inputs) / agent_coord.shape[0]

    print(f"Initial objective value: {objective(initial_params):.6f}")
    print(f"Initial parameter norm: {jnp.linalg.norm(initial_params):.6f}")

    result = minimize(
        objective, initial_params, method="BFGS", options={"maxiter": 100}
    )

    print(f"Optimization converged: {result.success}")
    print(f"Final objective value: {result.fun:.6f}")
    print(f"Number of iterations: {result.nit}")
    print(f"Final parameter norm: {jnp.linalg.norm(result.x):.6f}")

    assert result.fun < objective(
        initial_params
    ), "Optimization should reduce objective value"
    assert jnp.isfinite(result.fun), "Final objective should be finite"
    assert jnp.all(jnp.isfinite(result.x)), "Final parameters should be finite"

    print(f"Beta distance: {initial_params[0]:.6f} -> {result.x[0]:.6f}")
    print(f"Beta race: {initial_params[1]:.6f} -> {result.x[1]:.6f}")
    print(f"Beta income: {initial_params[2]:.6f} -> {result.x[2]:.6f}")
    print(
        f"Features betas norm: {jnp.linalg.norm(initial_params[3:]):.6f} -> "
        f"{jnp.linalg.norm(result.x[3:]):.6f}"
    )


def test_interaction_shapes():
    """Check block-block interactions preserve expected shapes for scalar and vector inputs."""
    scalar_data = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    vector_data = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

    diff_result = block_block_interaction("diff")(0, scalar_data)
    assert (
        diff_result.shape == scalar_data.shape
    ), f"Expected {scalar_data.shape}, got {diff_result.shape}"

    l1_result = block_block_interaction("l1")(0, vector_data)
    expected_shape = vector_data.shape[:-1]
    assert (
        l1_result.shape == expected_shape
    ), f"Expected {expected_shape}, got {l1_result.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
