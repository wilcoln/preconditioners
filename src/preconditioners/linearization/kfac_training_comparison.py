"""Compares training of MLP and lienarized model using KFAC"""
import argparse
import numpy as np
import operator
from datetime import datetime as dt
import os
import pickle
import json

import jax
import jax.numpy as jnp
from jax.example_libraries import stax
from jax.nn.initializers import normal
from jax.tree_util import tree_map

from preconditioners.datasets import generate_data, data_random_split
from preconditioners.optimizers.kfac import kfac_jax

parser = argparse.ArgumentParser()
# Model params
parser.add_argument('--num_layers', help='Number of layers', type=int, default=3)
parser.add_argument('--width', help='Hidden channels', type=int, default=128)
# Optimizer params
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-1)
parser.add_argument('--damping', help='Damping coefficient', type=float, default=1e-1)
parser.add_argument('--l2', help='L2-regularization coefficient', type=float, default=0.)
parser.add_argument('--full_model_fisher', help='If set, will apply the FI of '\
    'the full model to the linear model', action='store_true')
# Data params
parser.add_argument('--dataset', help='Type of dataset', choices=['linear', 'quadratic'], default='linear')
parser.add_argument('--train_size', help='Number of train examples', type=int, default=64)
parser.add_argument('--test_size', help='Number of test examples', type=int, default=64)
parser.add_argument('--in_dim', help='Dimension of features', type=int, default=8)
# Experiment params
parser.add_argument('--max_iter', help='Max epochs', type=int, default=256)
parser.add_argument('--save_every', help='Number of epochs per log', type=int, default=8)
parser.add_argument('--save_folder', help='Experiments are saved here', type=str, default="experiments/")
parser.add_argument('--num_test_points', help='Number of test points to analyse', type=int, default=32)
args = parser.parse_args()

class TrainingComparison:
    """KFAC experiment for comparing MLP against its linearization"""

    def __init__(self, model, params, lr=1e-1, damping=1e-1, l2=0.):
        self.model = model
        self.params = params
        self.init_params = params
        self.results = []
        self.epoch = 0

        self.lr = lr
        self.damping = damping
        self.l2 = l2

        # Create linearized model
        self.model_lin, self.params_lin = create_linearized_model(model, params)

        self.loss_fn = create_loss_fn(self.model, self.l2)
        self.loss_fn_lin = create_loss_fn(self.model_lin, self.l2)

        self.optimizer = None
        self.opt_state = None

        # Used for computing the Fisher
        self._optimizer_lin = None
        self._opt_state_lin = None
        self._grad_loss_lin = None

        self._done_initial_print = False

    def setup(self, train_data, key):
        """Prepares both models for training"""
        # Create optimizer for MLP
        self.optimizer = create_optimizer(self.loss_fn, self.l2)
        key, sub_key = jax.random.split(key)
        self.opt_state = self.optimizer.init(self.params, sub_key, train_data)

        # Create optimizer for linearized model
        self._optimizer_lin = create_optimizer(self.loss_fn, self.l2)
        key, sub_key = jax.random.split(key)
        self._opt_state_lin = self._optimizer_lin.init(self.params_lin, sub_key, train_data)
        self._grad_loss_lin = jax.grad(self.loss_fn_lin)

        # Make one step to compute Fisher at initialization
        key, sub_key = jax.random.split(key)
        _, self._opt_state_lin, _ = self._optimizer_lin.step(self.params_lin,
            self._opt_state_lin, sub_key, momentum=0., damping=self.damping,
            learning_rate=self.lr, batch=train_data)

    def step(self, train_data, key):
        """Performs one step of training for both the MLP and linearized model"""
        # Update MLP
        key, sub_key = jax.random.split(key)
        self.params, self.opt_state, _ = self.optimizer.step(
            self.params, self.opt_state, sub_key, momentum=0., damping=self.damping,
            learning_rate=self.lr, batch=train_data)

        # Update linearized model
        grad = self._grad_loss_lin(self.params_lin, train_data)
        grad_update = self._optimizer_lin._estimator.multiply_inverse(
            self._opt_state_lin.estimator_state, grad, self.damping,
            exact_power=False, use_cached=True)
        grad_update = kfac_jax.utils.scalar_mul(grad_update, self.lr)
        self.params_lin = jax.tree_map(jnp.subtract, self.params_lin, grad_update)

        self.epoch += 1

    def log_result(self, train_data, test_data, random_test_features=None, verbose=True):
        # Get train and test loss
        loss, loss_lin = self._eval(train_data)
        test_loss, test_loss_lin = self._eval(test_data)

        if verbose:
            if not self._done_initial_print:
                print("Epoch\tLoss\tLinear\tTest\tLinear")
                self._done_initial_print = True
            print(f'{self.epoch}\t{loss:.4f}\t{loss_lin:.4f}\t{test_loss:.4f}\t{test_loss_lin:.4f}')

        # Distance of parameter from initialization
        dist_from_init = param_dist(self.params, self.init_params)

        result = {
            'epoch': self.epoch,
            'train_loss': loss,
            'train_loss_lin': loss_lin,
            'test_loss': test_loss,
            'test_loss_lin': test_loss_lin,
            'frob_distance': dist_from_init
        }

        # Get output on random test points
        if random_test_features is not None:
            random_test_output = self.model(self.params, random_test_features)
            random_test_output = [float(random_test_output[i]) for i in range(random_test_features.shape[0])]
            random_test_output_lin = self.model_lin(self.params_lin, random_test_features)
            random_test_output_lin = [float(random_test_output_lin[i]) for i in range(random_test_features.shape[0])]

            result['random_test_output'] = random_test_output
            result['random_test_output_lin'] = random_test_output_lin

        if self.epoch == 0:
            result['param_norm'] = param_dist(self.params, 0)

        self.results.append(result)

    def save_results(self, folder_path, params_dict={}):
        """Saves experiment results"""
        # Create experiment folder
        dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        experiment_name = "training_comparison_" + dtstamp
        os.makedirs(os.path.join(folder_path, experiment_name))

        params_dict['max_epoch'] = self.epoch

        # Dump results and params
        with open(os.path.join(folder_path, experiment_name, 'results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        with open(os.path.join(folder_path, experiment_name, 'params.json'), 'w') as f:
            json.dump(params_dict, f)

    def _eval(self, input_dataset):
        """Tests the models on labelled data"""
        x, y = input_dataset
        y_hats = self.model(self.params, x)
        loss = jnp.mean(jnp.square(y_hats - y))

        y_hats_lin = self.model_lin(self.params_lin, x)
        loss_lin = jnp.mean(jnp.square(y_hats_lin - y))

        return float(loss), float(loss_lin)

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

    return f, params

def create_linearized_model(model, init_params):
    """Creates linearized model"""
    def f_lin(params, *args, **kwargs):
        params_lin = tree_map(operator.sub, params, init_params)
        f_params_x, proj = jax.jvp(lambda param: model(param, *args, **kwargs),
                               (init_params,), (params_lin,))
        return tree_map(operator.add, f_params_x, proj)
    params_lin = params

    return f_lin, params_lin

def create_loss_fn(model, l2):
    """Creates l2 loss function for a given model"""
    def loss_fn(params, batch):
        x, y = batch
        y_hats = model(params, x)

        kfac_jax.register_squared_error_loss(y_hats, y)
        reg = kfac_jax.utils.inner_product(params, params) / 2.0
        loss = jnp.mean(jnp.square(y_hats - y)) + l2 * reg
        return loss

    return loss_fn

def create_optimizer(loss_fn, l2):
    """Creates the KFAC optimizer"""
    return kfac_jax.Optimizer(
        value_and_grad_func=jax.value_and_grad(loss_fn),
        l2_reg=l2,
        value_func_has_aux=False,
        value_func_has_state=False,
        value_func_has_rng=False,
        multi_device=False
    )

if __name__ == "__main__":
    train_size, test_size = args.train_size, args.test_size
    dataset, in_dim = args.dataset, args.in_dim
    width, num_layers = args.width, args.num_layers
    lr, damping, l2 = args.lr, args.damping, args.l2
    max_iter, save_every, num_test_points = args.max_iter, args.save_every, args.num_test_points

    # Generate data
    print("Generating data...")
    dataset = generate_data(dataset, n=train_size + test_size, d=in_dim,
        regime='autoregressive', ro=0.5, r1=1, sigma2=1)
    train_data, test_data = data_random_split(dataset, (train_size, test_size))

    # Choose random test points to analyse
    x_test, y_test = test_data
    test_indices = np.random.randint(0, x_test.shape[0], size=num_test_points)
    x_rand_test = np.array([x_test[i] for i in test_indices])

    # Create MLP
    key = jax.random.PRNGKey(42)
    model, params = create_model(width, num_layers, in_dim=in_dim, out_dim=1, key=key)

    # Set up experiment
    print("Setting up optimizers...")
    experiment = TrainingComparison(model, params, lr=lr, damping=damping, l2=l2)
    key, sub_key = jax.random.split(key)
    experiment.setup(train_data, sub_key)

    # Run experiment
    print("Starting training...")
    experiment.log_result(train_data, test_data, x_rand_test)
    while experiment.epoch < max_iter:
        key, sub_key = jax.random.split(key)
        experiment.step(train_data, sub_key)

        if experiment.epoch % save_every == 0:
            experiment.log_result(train_data, test_data, x_rand_test)

    # Save results
    if args.save_folder:
        params_dict = vars(args)
        experiment.save_results(args.save_folder, params_dict)
        print("Saved results")
