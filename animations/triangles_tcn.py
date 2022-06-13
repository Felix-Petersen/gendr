# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import gendr
import imageio
import os
from tqdm import tqdm


RESOLUTION = 768


if __name__ == '__main__':
    device = 'cuda'
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    output_dir = os.path.join(data_dir, 'results/output_render40')
    os.makedirs(output_dir, exist_ok=True)

    verts1 = torch.tensor(
        [
            [-0.25/1.5-(0.25/1.5/2), -.2165065/1.5, 0.],
            [0.0-(0.25/1.5/2), 0.2165065/1.5, 0.],
            [0.25/1.5-(0.25/1.5/2), -.2165065/1.5, 0.],
            [-0.25/1.5+(0.25/1.5/2), -.2165065/1.5, 0.],
            [0.0+(0.25/1.5/2), 0.2165065/1.5, 0.],
            [0.25/1.5+(0.25/1.5/2), -.2165065/1.5, 0.],
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )
    faces1 = torch.tensor(
        [
            [1, 0, 2],
            [4, 3, 5],
        ],
        dtype=torch.int64,
        device=device,
    )

    camera_distance = 2.
    elevation = 0
    azimuth = 0

    mesh_ = gendr.Mesh(vertices=verts1, faces=faces1)

    sigmoid_functions = [
        ('uniform', 0),
        # ('gaussian', 0),  # Gaussian
        ('logistic', 0),  # Logistic
        # ('laplace', 0),  # Laplace
        # ('cubic_hermite', 0),  # Cubic Hermite
        # ('cauchy', 0),  # Cauchy
        # ('gamma', 2.),  # Gamma w/ p=2 p2=0
        # ('gamma', .5),  # Gamma w/ p=.5 p2=0
        # ('gamma_rev', 2.),  # Neg-Gamma w/ p=2 p2=0
        # ('gamma_rev', .5),  # Neg-Gamma w/ p=.5 p2=0
    ]

    t_conorms = [
        ('max', 0.),
        ('probabilistic', 0.),
        ('einstein', 0.),
        #
        ('yager', .5),
        ('yager', 1.),
        ('yager', 2.),
        ('yager', 4.),
        #
        ('aczel_alsina', .5),
        ('aczel_alsina', 1.),
        ('aczel_alsina', 2.),
        ('aczel_alsina', 4.),
    ]

    # for varying p: hamacher, yager, aczel_alsina

    for dist_id, (dist_func, dist_shape) in enumerate(tqdm(sigmoid_functions)):
        for aggr_id, (aggr_func, t_conorm_p) in enumerate(t_conorms):
            for tau_idx, log_tau in enumerate(np.arange(-5, 2, .025)):
            # for tau_idx, log_tau in enumerate(np.arange(-5, 2, .5)):

                transform = gendr.LookAt(viewing_angle=15)
                lighting = gendr.Lighting()
                renderer = gendr.GenDR(
                    image_size=RESOLUTION,
                    anti_aliasing=True,
                    dist_func=dist_func,
                    dist_scale=10**log_tau,
                    dist_shape=dist_shape,
                    dist_shift=0.,
                    dist_eps=10e10,
                    aggr_alpha_func=aggr_func,
                    aggr_alpha_t_conorm_p=t_conorm_p,
                    aggr_rgb_func='hard',
                )

                transform.set_eyes_from_angles(camera_distance, 0, 0)

                mesh = lighting(mesh_)
                mesh = transform(mesh)

                images = renderer(mesh)

                image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))

                # White triangle:
                image[:, :, :3] = 1. - image[:, :, :3]

                # image = 1. - image[:, :, 3]

                imageio.imsave(os.path.join(output_dir, 'tri_tcn_{}_{}_t{:03d}.png'.format(
                    dist_id, aggr_id, tau_idx
                )), (255 * image).astype(np.uint8))
