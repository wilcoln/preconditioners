import torch

from preconditioners.optimizers.precond_base import PrecondBase
from preconditioners.utils import model_gradients_using_batched_backprop


class PrecondGD3(PrecondBase):
    """Implements preconditioned gradient descent"""

    def _compute_p_inv(self) -> torch.Tensor:
        """Compute the inverse matrix"""
        group = self.param_groups[0]

        labeled_data = group['labeled_data']
        unlabeled_data = group['unlabeled_data']

        labeled_grad = model_gradients_using_batched_backprop(self.model, labeled_data)
        unlabeled_grad = model_gradients_using_batched_backprop(self.model, unlabeled_data)

        p = labeled_grad @ labeled_grad.T + unlabeled_grad @ unlabeled_grad.T
        p *= 1 / (labeled_data.shape[0] + unlabeled_data.shape[0])

        # Compute the inverse of the fisher information matrix
        return torch.pinverse(p)
