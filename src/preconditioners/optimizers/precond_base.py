import torch
from torch.optim.optimizer import Optimizer


class PrecondBase(Optimizer):
    """Implements preconditionned gradient descent"""

    def __init__(self, model, lr, labeled_data, unlabeled_data):
        defaults = dict(lr=lr, labeled_data=labeled_data, unlabeled_data=unlabeled_data)
        super(PrecondBase, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear'}
        self.modules = []

        self.model = model
        self._prepare_model()

        self.steps = 0

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep the following layers in PrecondDG. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                print('(%s): %s' % (count, module))
                count += 1

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    @staticmethod
    def _get_vector_form_grad(m, p_grad_mat):
        """
        :param m: the layer
        :param p_grad_mat: the matrix form of the gradient
        :return: a vector form of the gradient. it should be a output_dim * input_dim dimension vector.
        """
        return p_grad_mat[m].view(-1)

    def _get_natural_grad(self, i, m, p_grad_vec: dict, p_grad_mat: dict, p_inv):
        """
        :param i: the layer index
        :param m:  the layer
        :param p_grad_vec: the gradients in vector form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        k = len(p_grad_vec)
        m_size = p_grad_vec[m].shape[0]

        # Eduard Comment: I think it would be best to write a separate helper function for the
        # 4 lines of code below and write a couple of unit tests just for it, to be sure that
        # it does what you think it should be doing.

        # Natural gradient computation for layer m
        # As a vector of dimension output_dim * input_dim
        stacked_blocks = torch.stack([
            p_inv[i * k:i * k + m_size, j * k:j * k + p_grad_vec[mj].shape[0]] @ p_grad_vec[mj]
            for j, mj in enumerate(self.modules)
        ])
        v = torch.sum(stacked_blocks, 0)

        # v = np.zeros(p_grad_vec[m].shape)
        # for j, mj in enumerate(self.modules):
        #     mj_size = len(p_grad_vec[mj])
        #     block = p_inv[i*k:i*k+m_size, j*k:j*k+mj_size]
        #     v += block @ p_grad_vec[mj].numpy()

        # Reshaping as a matrix of dimension [output_dim , input_dim ]
        v = v.view(p_grad_mat[m].shape)

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _update_grad(self, updates):
        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])

    def _step(self):
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p.data.add_(-group['lr'], d_p)

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        updates = {}
        p_grad_mat = {m: self._get_matrix_form_grad(m, m.__class__.__name__) for m in self.modules}
        p_grad_vec = {m: self._get_vector_form_grad(m, p_grad_mat) for m in self.modules}
        p_inv = self._compute_p_inv()

        for i, m in enumerate(self.modules):
            v = self._get_natural_grad(i, m, p_grad_vec, p_grad_mat, p_inv)
            updates[m] = v
        self._update_grad(updates)

        self._step()
        self.steps += 1

    def _compute_p_inv(self) -> torch.Tensor:
        raise NotImplementedError
