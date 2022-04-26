import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax

# Hyper parameters
L2_REG = 1e-3


def train(input_dataset, mlp_output_sizes, max_iter):
    NUM_EPOCHS = max_iter

    def model_fn(x):
        """A Haiku MLP model function - three hidden layer network with tanh."""
        return hk.nets.MLP(
            output_sizes=mlp_output_sizes,
            with_bias=False,
            activation=jax.nn.tanh,
            activate_final=False,
        )(x)

    # The Haiku transformed model
    hk_model = hk.without_apply_rng(hk.transform(model_fn))

    def loss_fn(model_params, model_batch):
        """The loss function to optimize."""
        x, y = model_batch
        y_hats = hk_model.apply(model_params, x)
        kfac_jax.register_squared_error_loss(y_hats, y)
        loss = jnp.mean(jnp.square(y_hats - y))

        # The optimizer assumes that the function you provide has already added
        # the L2 regularizer to its gradients.
        # return loss + L2_REG * kfac_jax.utils.inner_product(model_params, model_params) / 2.0
        return loss

    # Create the optimizer
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=L2_REG,
        value_func_has_aux=False,
        value_func_has_state=False,
        value_func_has_rng=False,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        initial_damping=1.0,
        multi_device=False,
    )

    input_dataset = input_dataset[:]
    x, y = input_dataset
    x, y = x.cpu().numpy(), y.cpu().numpy()
    input_dataset = (x, y)
    rng = jax.random.PRNGKey(42)
    dummy_xs, dummy_ys = input_dataset
    rng, key = jax.random.split(rng)
    params = hk_model.init(key, dummy_xs)
    rng, key = jax.random.split(rng)
    opt_state = optimizer.init(params, key, (dummy_xs, dummy_ys))

    # Training loop
    for epoch in range(1, 1 + NUM_EPOCHS):
        rng, key = jax.random.split(rng)
        params, opt_state, stats = optimizer.step(params, opt_state, key, batch=input_dataset)
    return float(stats['loss']), hk_model, params


def test(model, model_params, input_dataset):
    input_dataset = input_dataset[:]
    x, y = input_dataset
    x, y = x.cpu().numpy(), y.cpu().numpy()
    y_hats = model.apply(model_params, x)
    loss = jnp.mean(jnp.square(y_hats - y))
    return float(loss)


class Kfac:
    pass