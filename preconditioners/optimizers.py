import torch
from torch.optim.optimizer import Optimizer


class PrecondGD(Optimizer):
    r"""Implements the algorithm proposed in https://arxiv.org/pdf/1704.08227.pdf, which is a provably accelerated method
    for stochastic optimization. This has been employed in https://openreview.net/forum?id=rJTutzbA- for training several
    deep learning models of practical interest. This code has been implemented by building on the construction of the SGD
    optimization module found in pytorch codebase.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (required)
        kappa (float, optional): ratio of long to short step (default: 1000)
        xi (float, optional): statistical advantage parameter (default: 10)
        smallConst (float, optional): any value <=1 (default: 0.7)
    Example:
        >>> import numpy as np
        >>> from preconditioners.optimizers import PrecondGD
        >>> params = model.parameters()
        >>> optimizer = PrecondGD(params=params, lr=0.1, p_inv = np.eye(len(params)))
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr, p_inv):
        defaults = dict(lr=lr, p_inv=p_inv)
        super(PrecondGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PrecondGD, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                p.data.add_(-group['lr'], d_p)

        return loss
