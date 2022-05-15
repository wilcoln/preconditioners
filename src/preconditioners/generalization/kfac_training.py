"""Wrappers for experiments with KFAC and KFAC extra"""
import numpy as np
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

class KFACExperiment:
    """Runs the KFAC optimizer"""

    def __init__(self, model, params, lr=1e-1, damping=1e-1, l2=0):
        self.model = model
        self.params = params

        self.results = []
        self.epoch = 0

        self.lr = lr
        self.damping = damping
        self.l2 = l2

        self.loss_fn = create_loss_fn(self.model, self.l2)
        self.optimizer = None
        self.opt_state = None

    def setup(self, train_data, key):
        """Prepares both models for training"""
        # Create optimizer for KFAC
        self.optimizer = create_optimizer(self.loss_fn, self.l2)
        key, sub_key = jax.random.split(key)
        self.opt_state = self.optimizer.init(self.params, sub_key, train_data)

    def step(self, train_data, key):
        """Performs one step of training for both models"""
        # Update KFAC
        key, sub_key = jax.random.split(key)
        self.params, self.opt_state, _ = self.optimizer.step(
            self.params, self.opt_state, sub_key, momentum=0., damping=self.damping,
            learning_rate=self.lr, batch=train_data)

        self.epoch += 1

    def log_result(self, train_data, test_data, verbose=True):
        # Get train and test loss
        loss = self._eval(train_data)
        test_loss = self._eval(test_data)

        result = {
            'epoch': self.epoch,
            'train_loss': loss,
            'test_loss': test_loss
        }

        self.results.append(result)

        return loss, test_loss

    def save_results(self, folder_path, params_dict={}):
        """Saves experiment results"""
        # Create experiment folder
        dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        experiment_name = "kfac_" + dtstamp
        os.makedirs(os.path.join(folder_path, experiment_name))

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

        return float(loss)

class KFACExtraExperiment:
    """Runs the KFAC extra optimizer"""

    def __init__(self, model, params, lr=1e-1, damping=1e-1, l2=0):
        self.model = model
        self.params = params

        self.results = []
        self.epoch = 0

        self.lr = lr
        self.damping = damping
        self.l2 = l2

        self.loss_fn = create_loss_fn(self.model, self.l2)
        self.optimizer = None
        self.opt_state = None
        self.grad_loss = None

    def setup(self, extra_data, key):
        """Prepares the optimizer for training"""
        # Create optimizer for KFAC-extra
        self.optimizer = create_optimizer(self.loss_fn, self.l2)
        key, sub_key = jax.random.split(key)
        self.opt_state = self.optimizer.init(self.params, sub_key, extra_data)
        self.grad_loss = jax.grad(self.loss_fn)

    def step(self, train_data, extra_data, key):
        """Performs one step of training"""
        # Update KFAC-extra
        # Compute curvature via step
        _, self.opt_state, _ = self.optimizer.step(
            self.params, self.opt_state, sub_key, momentum=0.,
            damping=self.damping, learning_rate=self.lr, batch=extra_data)
        grad = self.grad_loss(self.params, train_data)
        grad_update = self.optimizer._estimator.multiply_inverse(
            self.opt_state.estimator_state, grad, self.damping,
            exact_power=False, use_cached=True)
        grad_update = kfac_jax.utils.scalar_mul(grad_update, self.lr)
        self.params = jax.tree_map(jnp.subtract, self.params, grad_update)

        self.epoch += 1

    def log_result(self, train_data, test_data, verbose=True):
        """Store train/test loss for current epoch"""
        # Get train and test loss
        loss = self._eval(train_data)
        test_loss = self._eval(test_data)

        result = {
            'epoch': self.epoch,
            'train_loss': loss,
            'test_loss': test_loss
        }

        self.results.append(result)

        return loss, test_loss

    def save_results(self, folder_path, params_dict={}):
        """Saves experiment results"""
        # Create experiment folder
        dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        experiment_name = "kfac_extra_" + dtstamp
        os.makedirs(os.path.join(folder_path, experiment_name))

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

        return float(loss)

def param_dist(params_1, params_2=0):
    """Computes the Frobenius distance between two parameter lists"""
    norm = 0
    for i in range(len(params_1)):
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

def create_mlp(width, num_layers, in_dim, out_dim, key):
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
    # Just to test it out
    key = jax.random.PRNGKey(42)
    model, params = create_mlp(5, 3, 1, 1, key)

    kfac = KFACExperiment(model, params)
    kfac_extra = KFACExtraExperiment(model, params)

    data = (np.array([[1], [2], [3]]), np.array([[2], [4], [8]]))

    # Setup optimizers
    key, sub_key = jax.random.split(key)
    kfac.setup(data, key)
    key, sub_key = jax.random.split(key)
    kfac_extra.setup(data, key)

    print("KFAC\t\t\t\tKFAC Extra")
    print("Loss\tTest\t\tLoss\tTest")

    # Print initial loss
    loss, test_loss = kfac.log_result(data, data)
    loss_extra, test_loss_extra = kfac_extra.log_result(data, data)
    print(f"{loss}\t{test_loss}\t\t{loss_extra}\t{test_loss_extra}")

    # Train
    for epoch in range(10):
        key, sub_key = jax.random.split(key)
        kfac.step(data, key)
        key, sub_key = jax.random.split(key)
        kfac_extra.step(data, data, key)

    # Print final loss
    loss, test_loss = kfac.log_result(data, data)
    loss_extra, test_loss_extra = kfac_extra.log_result(data, data)
    print(f"{loss}\t{test_loss}\t\t{loss_extra}\t{test_loss_extra}")
    print("Finished")
