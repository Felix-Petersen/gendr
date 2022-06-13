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


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    output_dir = os.path.join(data_dir, 'results/output_render40')
    os.makedirs(output_dir, exist_ok=True)

    obj_file = os.path.join(current_dir, 'panda/Origami_Panda.obj')

    # other settings
    camera_distance = 3
    elevation = 20
    azimuth = 180

    # load from Wavefront .obj file
    mesh_ = gendr.Mesh.from_obj(obj_file, load_texture=True, texture_res=5, texture_type='surface')

    ####################################################################################################################
    # normalize into a [-1, 1]^3 box

    mesh_._vertices = mesh_._vertices - mesh_._vertices[0].min(dim=0)[0].reshape(1, 1, 3)

    max_dim_max, max_dim_idx = torch.max(mesh_._vertices[0].max(dim=0)[0], dim=0)

    mesh_._vertices = mesh_._vertices / max_dim_max

    mesh_._vertices = mesh_._vertices * 2 - mesh_._vertices[0].max(dim=0)[0].reshape(1, 1, 3)

    ####################################################################################################################

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

    for dist_id, (dist_func, dist_shape) in enumerate(tqdm(sigmoid_functions)):
        for aggr_id, (aggr_func, t_conorm_p) in enumerate(t_conorms):

            transform = gendr.LookAt()
            lighting = gendr.Lighting()
            renderer = gendr.GenDR(
                image_size=RESOLUTION,
                anti_aliasing=True,
                dist_func=dist_func,
                dist_shape=dist_shape,
                dist_shift=0.,
                aggr_alpha_func=aggr_func,
                aggr_alpha_t_conorm_p=t_conorm_p,
                # background_color=[126, 171, 55],
                # background_color=[66/255, 145/255, 0],
            )

            transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
            mesh = lighting(mesh_)
            mesh = transform(mesh)

            for gamma in [-2.5]:
                # for tau_idx, log_tau in enumerate(np.arange(-6, 1, .5)):
                for tau_idx, log_tau in enumerate(np.arange(-6, 1, .025)):
                    for eps in [-3]:
                        for dist_eps in [10]:

                            renderer.dist_scale = 10 ** log_tau
                            renderer.aggr_rgb_gamma = 10 ** gamma
                            renderer.aggr_rgb_eps = 10 ** eps
                            renderer.dist_eps = 10 ** dist_eps

                            images = renderer(mesh)

                            image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))

                            image = image[:, :, 3:] * image[:, :, :3] + (1-image[:, :, 3:]) * np.array([66/255, 145/255, 0.]).reshape((1, 1, 3))

                            imageio.imsave(os.path.join(output_dir, 'panda_tcn_{}_{}_t{:03d}.png'.format(
                                dist_id, aggr_id, tau_idx,
                            )), (255 * image).astype(np.uint8))


if __name__ == '__main__':
    main()
