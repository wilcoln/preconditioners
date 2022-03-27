from torch.optim.optimizer import Optimizer


class GradientDescent(Optimizer):

    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(GradientDescent, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GradientDescent, self).__setstate__(state)

    def step(self, closure=None):
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
