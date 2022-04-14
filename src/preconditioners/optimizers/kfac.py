import haiku as hk
import jax
import jax.numpy as jnp
import kfac_jax

# Hyper parameters

NUM_CLASSES = 10
L2_REG = 1e-3
NUM_BATCHES = 100


def make_dataset_iterator(batch_size):
    # Dummy dataset, in practice this should be your dataset pipeline
    for _ in range(NUM_BATCHES):
        yield jnp.zeros([batch_size, 100]), jnp.ones([batch_size], dtype="int32")


def softmax_cross_entropy(logits: jnp.ndarray, targets: jnp.ndarray):
    """Softmax cross entropy loss."""
    # We assume integer labels
    assert logits.ndim == targets.ndim + 1

    # Tell KFAC-JAX this model represents a classifier
    # See https://kfac-jax.readthedocs.io/en/latest/overview.html#supported-losses
    kfac_jax.register_softmax_cross_entropy_loss(logits, targets)
    log_p = jax.nn.log_softmax(logits, axis=-1)
    return - jax.vmap(lambda x, y: x[y])(log_p, targets)


def model_fn(x):
    """A Haiku MLP model function - three hidden layer network with tanh."""
    return hk.nets.MLP(
        output_sizes=(50, 50, 50, NUM_CLASSES),
        with_bias=True,
        activation=jax.nn.tanh,
    )(x)


# The Haiku transformed model
hk_model = hk.without_apply_rng(hk.transform(model_fn))


def loss_fn(model_params, model_batch):
    """The loss function to optimize."""
    x, y = model_batch
    logits = hk_model.apply(model_params, x)
    loss = jnp.mean(softmax_cross_entropy(logits, y))

    # The optimizer assumes that the function you provide has already added
    # the L2 regularizer to its gradients.
    return loss + L2_REG * kfac_jax.utils.inner_product(params, params) / 2.0


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

input_dataset = make_dataset_iterator(128)
rng = jax.random.PRNGKey(42)
dummy_images, dummy_labels = next(input_dataset)
rng, key = jax.random.split(rng)
params = hk_model.init(key, dummy_images)
rng, key = jax.random.split(rng)
opt_state = optimizer.init(params, key, (dummy_images, dummy_labels))

# Training loop
for i, batch in enumerate(input_dataset):
    rng, key = jax.random.split(rng)
    params, opt_state, stats = optimizer.step(
        params, opt_state, key, batch=batch, global_step_int=i)
    print(i, stats)