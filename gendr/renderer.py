# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

import gendr


class GenDR(nn.Module):
    def __init__(self,
                 image_size=256,
                 background_color=[0, 0, 0],
                 anti_aliasing=False,
                 #
                 dist_func='uniform',
                 dist_scale=1e-2,
                 dist_squared=False,
                 dist_shape=None,
                 dist_shift=None,
                 dist_eps=1e4,
                 #
                 aggr_alpha_func='probabilistic',
                 aggr_alpha_t_conorm_p=None,
                 #
                 aggr_rgb_func='softmax',
                 aggr_rgb_eps=1e-3,
                 aggr_rgb_gamma=1e-3,
                 #
                 near=1,
                 far=100,
                 double_side=False,
                 texture_type='surface',
                 ):
        super(GenDR, self).__init__()

        if aggr_rgb_func not in ['hard', 'softmax']:
            raise ValueError('Aggregate function (RGB) currently only supports hard and softmax.')
        if texture_type not in ['surface', 'vertex']:
            raise ValueError('Texture type only support surface and vertex.')

        self.image_size = image_size
        self.background_color = background_color
        self.anti_aliasing = anti_aliasing

        self.dist_func = dist_func
        self.dist_scale = dist_scale
        self.dist_squared = dist_squared
        self.dist_shape = dist_shape
        self.dist_shift = dist_shift
        self.dist_eps = dist_eps

        self.aggr_alpha_func = aggr_alpha_func
        self.aggr_alpha_t_conorm_p = aggr_alpha_t_conorm_p

        self.aggr_rgb_func = aggr_rgb_func
        self.aggr_rgb_eps = aggr_rgb_eps
        self.aggr_rgb_gamma = aggr_rgb_gamma

        self.near = near
        self.far = far
        self.double_side = double_side
        self.texture_type = texture_type

    def forward(self, mesh):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)

        images = gendr.functional.render(
            face_vertices=mesh.face_vertices,
            textures=mesh.face_textures,
            image_size=image_size,
            background_color=self.background_color,
            dist_func=self.dist_func,
            dist_scale=self.dist_scale,
            dist_squared=self.dist_squared,
            dist_shape=self.dist_shape,
            dist_shift=self.dist_shift,
            dist_eps=self.dist_eps,
            aggr_alpha_func=self.aggr_alpha_func,
            aggr_alpha_t_conorm_p=self.aggr_alpha_t_conorm_p,
            aggr_rgb_func=self.aggr_rgb_func,
            aggr_rgb_eps=self.aggr_rgb_eps,
            aggr_rgb_gamma=self.aggr_rgb_gamma,
            near=self.near,
            far=self.far,
            double_side=self.double_side,
            texture_type=self.texture_type,
        )

        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images

    def forward_tensors(self, face_vertices, face_textures):
        image_size = self.image_size * (2 if self.anti_aliasing else 1)

        images = gendr.functional.render(
            face_vertices=face_vertices,
            textures=face_textures,
            image_size=image_size,
            background_color=self.background_color,
            dist_func=self.dist_func,
            dist_scale=self.dist_scale,
            dist_squared=self.dist_squared,
            dist_shape=self.dist_shape,
            dist_shift=self.dist_shift,
            dist_eps=self.dist_eps,
            aggr_alpha_func=self.aggr_alpha_func,
            aggr_alpha_t_conorm_p=self.aggr_alpha_t_conorm_p,
            aggr_rgb_func=self.aggr_rgb_func,
            aggr_rgb_eps=self.aggr_rgb_eps,
            aggr_rgb_gamma=self.aggr_rgb_gamma,
            near=self.near,
            far=self.far,
            double_side=self.double_side,
            texture_type=self.texture_type,
        )

        if self.anti_aliasing:
            images = F.avg_pool2d(images, kernel_size=2, stride=2)

        return images
