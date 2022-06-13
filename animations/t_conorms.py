# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import gendr.cuda.generalized_renderer as generalized_renderer_cuda
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from tqdm import tqdm

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray"})


class TConorm(torch.autograd.Function):
    def forward(ctx, t_conorm_id: int, a: torch.Tensor, b: torch.tensor, p: float=0.):
        a_ = a.reshape(-1)
        b_ = b.reshape(-1)
        r_ = []
        for a__, b__ in zip(a_, b_):
            r__ = generalized_renderer_cuda.t_conorm_forward(
                t_conorm_id,
                a__.item(),
                b__.item(),
                1,
                p,
            )
            r_.append(r__)
        r_ = torch.tensor(r_)
        ctx.t_conorm_id = t_conorm_id
        ctx.p = p
        ctx.shape = list(a.shape)
        r = r_.reshape(*a.shape)
        ctx.save_for_backward(r, b)
        return r

    def backward(ctx, r_grad):
        r, b = ctx.saved_tensors
        r_ = r.reshape(-1)
        b_ = b.reshape(-1)
        b_grad = []
        for r__, b__ in zip(r_, b_):
            b__grad = generalized_renderer_cuda.t_conorm_backward(
                ctx.t_conorm_id,
                r__.item(),
                b__.item(),
                2,
                ctx.p,
            )
            b_grad.append(b__grad)
        b_grad = torch.tensor(b_grad)
        b_grad = b_grad.reshape(*ctx.shape)
        b_grad = b_grad * r_grad
        return None, None, b_grad, None


if __name__ == '__main__':

    os.makedirs('display', exist_ok=True)

    X = np.arange(0, 1, 0.01)
    Y = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(X, Y)
    X, Y = torch.tensor(X), torch.tensor(Y)



    for t_conorm_idx, (t_conorm_name, ps) in enumerate(tqdm([
        ('max', [0.]),
        ('probabilistic', [0.]),
        ('einstein', [0.]),
        #
        ('yager', [.5, 1., 2., 4.]),
        ('aczel_alsina', [.5, 1., 2., 4.]),
        #
        ('hamacher', [2**x for x in np.arange(-4, 4.00001, .025)[::-1]]),
        ('yager', [2**x for x in np.arange(-4, 4.00001, .025)]),
        ('aczel_alsina', [2**x for x in np.arange(-4, 4.00001, .025)]),
    ])):
        t_conorm_id = {
            'hard': 0,
            'max': 1,
            'probabilistic': 2,
            'einstein': 3,
            'hamacher': 4,
            'frank': 5,
            'yager': 6,
            'aczel_alsina': 7,
            'dombi': 8,
            'schweizer_sklar': 9,
        }[t_conorm_name]

        for p_idx, p in enumerate(ps):
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(azim=240)

            Z = TConorm.apply(t_conorm_id, X, Y, p).detach().numpy()

            surf = ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
            ax.set_zlim(0., 1.)
            ax.zaxis.set_major_locator(LinearLocator(11))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

            # ax.set_title('{}, p={}'.format(t_conorm_name, p))

            plt.savefig('display/t_conorm_{}_p{:03d}.png'.format(t_conorm_idx, p_idx), bbox_inches='tight', dpi=200, transparent=True)
