
import haiku as hk
import jax
import jax.numpy as jnp

from . import kfac_jax

# Hyper parameters
L2_REG = 0


def train(input_dataset, mlp_output_sizes, max_iter, damping, tol, print_every=10):
    model_logs = {'condition': None, 'losses': []}

    def model_fn(x):
        """A Haiku MLP model function - three hidden layer network with tanh."""
        width = mlp_output_sizes[0]
        return hk.nets.MLP(
            output_sizes=mlp_output_sizes,
            with_bias=True,
            activation=jax.nn.tanh,
            activate_final=False,
            w_init=hk.initializers.RandomNormal(stddev=1/jnp.sqrt(width)),
            b_init=hk.initializers.RandomNormal(stddev=1),
        )(x)

    # The Haiku transformed model
    hk_model = hk.without_apply_rng(hk.transform(model_fn))

    def loss_fn(model_params, model_batch):
        """The loss function to optimize."""
        x, y = model_batch
        y_hats = hk_model.apply(model_params, x).squeeze()
        kfac_jax.register_squared_error_loss(y_hats, y)
        loss = jnp.mean(jnp.square(y_hats - y))

        # The optimizer assumes that the function you provide has already added
        # the L2 regularizer to its gradients.
        return loss  # + L2_REG * kfac_jax.utils.inner_product(model_params, model_params) / 2.0

    # Create the optimizer
    optimizer = kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=L2_REG,
        value_func_has_aux=False,
        value_func_has_state=False,
        value_func_has_rng=False,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=False,
        use_adaptive_damping=False,
        # initial_damping=damping,
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

    current_loss = float('inf')
    epoch = 0
    # stop if 5 consecutive epochs have no improvement
    no_improvement_counter = 0
    condition = None

    while not condition:
        rng, key = jax.random.split(rng)
        previous_loss = current_loss
        params, opt_state, stats = optimizer.step(params, opt_state, key, batch=input_dataset, momentum=0.0,
                                                  damping=damping)

        current_loss = float(stats['loss'])

        epoch += 1

        # Print statistics
        if epoch == 1 or epoch % print_every == 0:
            print(f'Epoch {epoch}: Train loss: {current_loss:.4f}')

        # Update condition
        delta_loss = current_loss - previous_loss
        no_improvement_counter += 1 if jnp.abs(delta_loss) < 1e-6 else 0
        if no_improvement_counter > 5:  # stagnation
            condition = 'stagnation'
        elif current_loss <= tol:
            condition = 'tol'
        elif epoch >= max_iter:
            condition = 'max_iter'

        model_logs['losses'].append(current_loss)

    # Final print
    print('*** FINAL EPOCH ***')
    print(f'Epoch {epoch}: Train loss: {current_loss:.4f}, Stop condition: {condition}')

    # Save train logs
    model_logs['condition'] = condition

    # Return loss

    return current_loss, hk_model, params, model_logs


def test(model, model_params, input_dataset):
    input_dataset = input_dataset[:]
    x, y = input_dataset
    x, y = x.cpu().numpy(), y.cpu().numpy()
    y_hats = model.apply(model_params, x)
    loss = jnp.mean(jnp.square(y_hats - y))
    return float(loss)


class Kfac:
    pass
