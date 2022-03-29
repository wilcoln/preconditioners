import torch

from preconditioners.optimizers.precond_base import PrecondBase
from preconditioners.utils import model_gradients_using_direct_computation


class PrecondGD2(PrecondBase):
    """Implements preconditioned gradient descent"""

    def _compute_p_inv(self) -> torch.Tensor:
        """Compute the inverse matrix"""
        group = self.param_groups[0]

        labeled_data = group['labeled_data']
        unlabeled_data = group['unlabeled_data']

        # Compute the output of the model on the labeled and unlabeled data
        y_labeled = self.model(labeled_data)
        y_unlabeled = self.model(unlabeled_data)

        # Fix to allow computation of gradients on non-leaf nodes
        y_labeled.retain_grad()
        y_unlabeled.retain_grad()

        # Compute the gradient of the output on the labeled and unlabeled data w.r.t the model parameters
        labeled_grad_list = [model_gradients_using_direct_computation(y, self.model) for y in torch.unbind(y_labeled)]
        unlabeled_grad_list = [model_gradients_using_direct_computation(y, self.model) for y in torch.unbind(y_unlabeled)]

        # Compute the fisher information matrix at this iteration
        stacked_labeled_grads = torch.stack([grad @ grad.T for grad in labeled_grad_list])
        stacked_unlabeled_grads = torch.stack([grad @ grad.T for grad in unlabeled_grad_list])

        p = torch.sum(stacked_labeled_grads, 0) + torch.sum(stacked_unlabeled_grads, 0)
        p *= 1 / (labeled_data.shape[0] + unlabeled_data.shape[0])

        # Compute the inverse of the fisher information matrix
        return torch.pinverse(p)
