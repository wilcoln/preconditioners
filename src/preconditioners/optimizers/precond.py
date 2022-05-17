from mimetypes import init
import torch
import numpy as np
from preconditioners.optimizers.precond_base import PrecondBase
from preconditioners.utils import model_gradients_using_backprop, model_gradients_using_batched_backprop
from preconditioners.settings import DEVICE
from sklearn.covariance import graphical_lasso, GraphicalLasso


class PrecondGD(PrecondBase):
    """Implements preconditioned gradient descent"""

    def __init__(self, model, lr, labeled_data, unlabeled_data, damping=0., is_linear=False, verbose=True) -> None:
        super(PrecondGD, self).__init__(model, lr, labeled_data, unlabeled_data, damping, verbose)
        self.last_p_inv = None
        self.recompute_p_inv = True
        self.is_linear = is_linear

    def _compute_fisher(self) -> torch.Tensor:
        group = self.param_groups[0]
        unlabeled_data = group['unlabeled_data']

        # Compute the gradient of the output on the labeled and unlabeled data w.r.t the model parameters
        p = 0
        if isinstance(unlabeled_data, torch.Tensor):
            unlabeled_data = torch.unbind(unlabeled_data)

        # TODO: deadline hack, fix this
        size = 0
        for x in unlabeled_data:
            grad = model_gradients_using_backprop(self.model, x).detach()
            p += grad @ grad.T
            size += 1

        # Compute the inverse of the fisher information matrix
        p /= size

        return p

    def _compute_p_inv_old(self) -> torch.Tensor:
        """Compute the inverse matrix"""

        group = self.param_groups[0]

        if self.recompute_p_inv:
            p = self._compute_fisher()

            # add damping to avoid division by zero
            p += torch.eye(p.shape[0], device=DEVICE) * group['damping']
            p_inv = torch.cholesky_inverse(p)

            self.last_p_inv = p_inv
            self.recompute_p_inv = False
            return p_inv

        else:
            return self.last_p_inv

    def _compute_p_inv(self) -> torch.Tensor:
        """Compute the inverse Fisher using graphical lasso"""

        group = self.param_groups[0]
        unlabeled_data = group['unlabeled_data']

        if self.recompute_p_inv:
            # get first gradient to get its shape
            #grad = model_gradients_using_backprop(self.model, unlabeled_data[0]).detach()
            #X = np.zeros((int(len(unlabeled_data)), len(grad)))
            #X[0] = grad.detach().numpy().flatten()

            #for i in range(len(unlabeled_data)-1):
            #    x = unlabeled_data[i+1]
            #    grad = model_gradients_using_backprop(self.model, x).detach()
            #    X[i+1] = grad.detach().numpy().flatten()

            grad = model_gradients_using_backprop(self.model, unlabeled_data[0]).detach().numpy().flatten()
            p = len(grad)

            emp_cov = np.eye(p)
            for x in unlabeled_data:
                grad = model_gradients_using_backprop(self.model, x).detach().numpy().flatten()
                emp_cov += grad.dot(grad.T)
            
            emp_cov = emp_cov / len(unlabeled_data)
            

            #cov = GraphicalLasso().fit(unlabeled_grad.detach().numpy())
            init = self.last_p_inv.detach().numpy() if self.last_p_inv else None
            #cov = GraphicalLasso(alpha = 0.2, cov_init = init).fit(X)
            _, precision, _, _ = graphical_lasso(emp_cov, alpha = 1, cov_init = init)
            p_inv = torch.from_numpy(precision).to(DEVICE)

            self.last_p_inv = p_inv
            self.recompute_p_inv = False
            return p_inv

        else:
            return self.last_p_inv

        return p_inv

