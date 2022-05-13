import torch

from preconditioners.optimizers.precond_base import PrecondBase
from preconditioners.utils import model_gradients_using_backprop


class PrecondGD(PrecondBase):
    """Implements preconditioned gradient descent"""

    def _compute_p_inv(self) -> torch.Tensor:
        """Compute the inverse matrix"""
        group = self.param_groups[0]

        labeled_data = group['labeled_data']
        unlabeled_data = group['unlabeled_data']

        # Compute the gradient of the output on the labeled and unlabeled data w.r.t the model parameters
        p = torch.sum(torch.stack([
                grad @ grad.T
                for grad in [
                    model_gradients_using_backprop(self.model, x)
                    for x in torch.unbind(labeled_data)
                ]
            ]
        ), 0) + torch.sum(torch.stack([
                grad @ grad.T
                for grad in [
                    model_gradients_using_backprop(self.model, x)
                    for x in torch.unbind(unlabeled_data)
                ]
            ]
        ), 0)

        p *= 1 / (labeled_data.shape[0] + unlabeled_data.shape[0])
        # Compute the inverse of the fisher information matrix
        # add damping to avoid division by zero
        p += torch.eye(p.shape[0]) * group['damping']
        return torch.cholesky_inverse(p)
