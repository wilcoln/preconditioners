import torch

from preconditioners.optimizers.precond_base import PrecondBase
from preconditioners.utils import model_gradients_using_backprop
from preconditioners.settings import DEVICE


class PrecondGD(PrecondBase):
    """Implements preconditioned gradient descent"""

    def __init__(self, model, lr, labeled_data, unlabeled_data, damping=0., is_linear=False, verbose=True) -> None:
        super(PrecondGD, self).__init__(model, lr, labeled_data, unlabeled_data, damping, verbose)
        self.last_p_inv = None
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

    def _compute_p_inv(self) -> torch.Tensor:
        """Compute the inverse matrix"""

        group = self.param_groups[0]

        # if the model is linear, check if p_inv has previously been computed
        if self.is_linear and self.last_p_inv is not None:
            return self.last_p_inv

        p = self._compute_fisher()

        # add damping to avoid division by zero
        p += torch.eye(p.shape[0], device=DEVICE) * group['damping']
        p_inv = torch.cholesky_inverse(p)

        if self.is_linear:
            self.last_p_inv = p_inv

        return p_inv
