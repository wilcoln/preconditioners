import argparse
import numpy as np
import operator
from datetime import datetime as dt
import os
import pickle
import json

from torch.utils.data import random_split
import haiku as hk
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers, stax
from jax.nn.initializers import normal
from jax.tree_util import tree_multimap

from preconditioners.utils import generate_true_parameter, generate_c, generate_W_star
from preconditioners.datasets import CenteredLinearGaussianDataset, CenteredQuadraticGaussianDataset
from preconditioners.optimizers.kfac import kfac_jax

L2_REG = 0.

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', help='Number of layers', default=3, type=int)
parser.add_argument('--width', help='Hidden channels', type=int, default=5)
parser.add_argument('--max_iter', help='Max epochs', default=100, type=int)
parser.add_argument('--save_every', help='Number of epochs per log', default=10, type=int)
parser.add_argument('--damping', help='damping', type=float, default=1.0)
parser.add_argument('--save_folder', help='Experiments are saved here', type=str, default="experiments/")
args = parser.parse_args()

def train(model, model_lin, params, params_lin, loss_fn, loss_fn_lin, train_data, test_data, key, num_test_points=7, max_iter=float('inf'), save_every=10):
    """Trains both models"""
    # Detach dataset
    x_train, y_train = train_data[:]
    x_train, y_train = x_train.cpu().numpy(), y_train.cpu().numpy()
    train_data = (x_train, y_train)

    # Choose random test points to analyse
    x_test, y_test = test_data[:]
    x_test, y_test = x_test.cpu().numpy(), y_test.cpu().numpy()
    test_data = (x_test, y_test)
    test_indices = np.random.randint(0, x_test.shape[0], size=num_test_points)
    x_rand_test = np.array([x_test[i] for i in test_indices])

    # Create optimizer for MLP
    optimizer = create_optimizer(loss_fn)
    key, sub_key = jax.random.split(key)
    opt_state = optimizer.init(params, sub_key, train_data)

    # Create optimizer for linearized model
    optimizer_lin = create_optimizer(loss_fn_lin)
    key, sub_key = jax.random.split(key)
    opt_state_lin = optimizer_lin.init(params_lin, sub_key, train_data)

    results = []
    epoch = 0

    init_params = params

    stats = None
    stats_lin = None

    print("Starting training...")
    print("Epoch\tLoss\tLinear\tTest\tLinear")

    def log_result():
        # Get train and test loss
        loss = test(model, params, train_data)
        loss_lin = test(model_lin, params_lin, train_data)
        test_loss = test(model, params, test_data)
        test_loss_lin = test(model_lin, params_lin, test_data)

        print(f'{epoch}\t{loss:.4f}\t{loss_lin:.4f}\t{test_loss:.4f}\t{test_loss_lin:.4f}')

        # Distance of parameter from initialization
        dist_from_init = param_dist(params, init_params)

        # Get output on random test points
        random_test_output = model(params, x_rand_test)
        random_test_output = [float(random_test_output[i]) for i in range(x_rand_test.shape[0])]
        random_test_output_lin = model_lin(params_lin, x_rand_test)
        random_test_output_lin = [float(random_test_output_lin[i]) for i in range(x_rand_test.shape[0])]

        result = {
            'epoch': epoch,
            'train_loss': loss,
            'train_loss_lin': loss_lin,
            'test_loss': test_loss,
            'test_loss_lin': test_loss_lin,
            'frob_distance': dist_from_init,
            'random_test_output': random_test_output,
            'random_test_output_lin': random_test_output_lin
        }
        if epoch == 0:
            result['param_norm'] = param_dist(params, 0)

        results.append(result)

    # Sanity check
    log_result()

    while epoch < max_iter:
        # Train models
        key, sub_key = jax.random.split(key)
        params, opt_state, stats = optimizer.step(params, opt_state, sub_key, batch=train_data)
        key, sub_key = jax.random.split(key)
        params_lin, opt_state_lin, stats_lin = optimizer_lin.step(params_lin, opt_state_lin, sub_key, batch=train_data)

        epoch += 1

        # Print statistics
        if epoch % save_every == 0:
            log_result()

    # Final log
    log_result()

    return results

def param_dist(params_1, params_2=0):
    """Computes the Frobenius distance between two parameter lists"""
    norm = 0
    for i in range(len(params)):
        p = params_1[i]
        for j in range(len(p)):
            v = p[j]

            if len(jnp.shape(v)) <= 1:
                ord = None
            else:
                ord = 'fro'

            if params_2 == 0:
                norm += jnp.square(jnp.linalg.norm(v, ord=ord))
            else:
                v_2 = params_2[i][j]
                norm += jnp.square(jnp.linalg.norm(v - v_2, ord=ord))
    return jnp.sqrt(norm)

def create_model(width, num_layers, in_dim, out_dim, key):
    """Creates MLP and square loss function using stax"""
    layers = []
    scaling = 1./np.sqrt(width)
    for i in range(num_layers - 1):
        layers.append(stax.Dense(width, W_init=normal(scaling)))
        layers.append(stax.Tanh)
    layers.append(stax.Dense(out_dim, W_init=normal(scaling)))

    init_fn, f = stax.serial(*layers)

    _, params = init_fn(key, (-1, in_dim))

    def loss_fn(params, batch):
        x, y = batch
        y_hats = f(params, x)

        kfac_jax.register_squared_error_loss(y_hats, y)
        loss = jnp.mean(jnp.square(y_hats - y)) +  + L2_REG * kfac_jax.utils.inner_product(params, params) / 2.0
        return loss

    return f, loss_fn, params

def create_linearized_model(model, init_params):
    """Creates linearized model"""
    def f_lin(params, *args, **kwargs):
        params_lin = tree_multimap(operator.sub, params, init_params)
        f_params_x, proj = jax.jvp(lambda param: model(param, *args, **kwargs),
                               (init_params,), (params_lin,))
        return tree_multimap(operator.add, f_params_x, proj)

    def loss_fn_lin(params, batch):
        x, y = batch
        y_hats = f_lin(params, x)

        kfac_jax.register_squared_error_loss(y_hats, y)
        loss = jnp.mean(jnp.square(y_hats - y)) +  + L2_REG * kfac_jax.utils.inner_product(params, params) / 2.0
        return loss

    params_lin = params
    return f_lin, loss_fn_lin, params_lin

def create_optimizer(loss_fn):
    """Creates the KFAC optimizer"""
    return kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=L2_REG,
        value_func_has_aux=False,
        value_func_has_state=False,
        value_func_has_rng=False,
        use_adaptive_learning_rate=True,
        use_adaptive_momentum=True,
        use_adaptive_damping=True,
        initial_damping=1.,
        multi_device=False
    )

def generate_quadratic_data(sigma2):
    """Generates test data. Taken from optimizer_benchmark"""
    global d, train_size, test_size, extra_size
    w_star = generate_true_parameter(d, 1, m=np.eye(d))
    W_star = generate_W_star(d, 1)
    c = generate_c(0.5, regime='autoregressive', d=d)
    dataset = CenteredQuadraticGaussianDataset(
        W_star=W_star, w_star=w_star, d=d, c=c,
        n=train_size + test_size + extra_size, sigma2=sigma2)

    train_data, test_data, extra_data = random_split(dataset, [train_size, test_size, extra_size])

    return train_data, test_data, extra_data

def test(model, model_params, input_dataset):
    """Tests the model on labelled data"""
    x, y = input_dataset
    y_hats = model(model_params, x)
    loss = jnp.mean(jnp.square(y_hats - y))

    return float(loss)

def save_results(results, params_dict, folder_path):
    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    experiment_name = "TMP_training_comparison_" + dtstamp

    os.makedirs(os.path.join(folder_path, experiment_name))

    with open(os.path.join(folder_path, experiment_name, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(folder_path, experiment_name, 'params.json'), 'w') as f:
        json.dump(params_dict, f)

if __name__ == "__main__":
    # Generate data
    d = 10
    train_size, test_size, extra_size = 100, 100, 0
    train_data, test_data, extra_data = generate_quadratic_data(sigma2=1)

    width, num_layers, max_iter, save_every = args.width, args.num_layers, args.max_iter, args.save_every

    # Create MLP
    key = jax.random.PRNGKey(42)
    model, loss_fn, params = create_model(width, num_layers, in_dim=d, out_dim=1, key=key)

    # Create linearized model
    key, sub_key = jax.random.split(key)
    model_lin, loss_fn_lin, params_lin = create_linearized_model(model, params)

    # Training loop
    results = train(model, model_lin, params, params_lin, loss_fn, loss_fn_lin, train_data, test_data, key, max_iter=max_iter, save_every=save_every)

    params_dict = {
        'extra_size': extra_size,
        'd': d,
        'train_size': train_size,
        'test_size': test_size,
        'num_layers': num_layers,
        'max_iter': max_iter,
        'width': width
    }

    if args.save_folder:
        save_results(results, params_dict, args.save_folder)
