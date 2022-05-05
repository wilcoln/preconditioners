import torch

from preconditioners.optimizers.precond_base import PrecondBase
from preconditioners.utils import model_gradients_using_backprop


class PrecondGD(PrecondBase):
    """Implements preconditioned gradient descent"""

    def __init__(self, model, lr, labeled_data, unlabeled_data, damping=1.0, is_linear=False) -> None:
        super(PrecondGD, self).__init__(model, lr, labeled_data, unlabeled_data, damping)
        self.last_p_inv = None
        self.is_linear = is_linear

    def _compute_p_inv(self) -> torch.Tensor:
        """Compute the inverse matrix"""
        # if the model is linear, check if p_inv has previously been computed
        if self.is_linear and self.last_p_inv is not None:
            return self.last_p_inv

        group = self.param_groups[0]

        labeled_data = group['labeled_data']
        unlabeled_data = group['unlabeled_data']

        # Compute the gradient of the output on the labeled and unlabeled data w.r.t the model parameters
        labeled_grad_list = [model_gradients_using_backprop(self.model, x) for x in torch.unbind(labeled_data)]
        unlabeled_grad_list = [model_gradients_using_backprop(self.model, x) for x in torch.unbind(unlabeled_data)]

        # Compute the fisher information matrix at this iteration
        stacked_labeled_grads = torch.stack([grad @ grad.T for grad in labeled_grad_list])
        stacked_unlabeled_grads = torch.stack([grad @ grad.T for grad in unlabeled_grad_list])

        p = torch.sum(stacked_labeled_grads, 0) + torch.sum(stacked_unlabeled_grads, 0)
        p *= 1 / (labeled_data.shape[0] + unlabeled_data.shape[0])

        # Compute the inverse of the fisher information matrix
        # add damping to avoid division by zero
        p += torch.eye(p.shape[0]) * group['damping'] / torch.norm(p)
        p_inv = torch.cholesky_inverse(p)

        if self.is_linear:
            self.last_p_inv = p_inv

        return p_inv
