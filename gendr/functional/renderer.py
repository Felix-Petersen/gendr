# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Function
import gendr.cuda.generalized_renderer as generalized_renderer_cuda


class GenDRFunction(Function):
    @staticmethod
    def forward(
            ctx,
            face_vertices,
            textures,
            #
            image_size=256,
            background_color=[0, 0, 0],
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
            double_side=True,
            texture_type='surface',
    ):

        # face_vertices: [nb, nf, 9]
        # textures: [nb, nf, 9]

        dist_func_map = {
            'hard': 0, 'heaviside': 0,
            'uniform': 1,
            'cubic_hermite': 2,
            'wigner_semicircle': 3,
            'gaussian': 4,
            'laplace': 5,
            'logistic': 6,
            'gudermannian': 7, 'hyperbolic_secant': 7,
            'cauchy': 8,
            'reciprocal': 9,
            'gumbel_max': 10,
            'gumbel_min': 11,
            'exponential': 12,
            'exponential_rev': 13,
            'gamma': 14,
            'gamma_rev': 15,
            'levy': 16,
            'levy_rev': 17,
        }
        aggr_rgb_func_map = {
            'hard': 0,
            'softmax': 1,
        }
        aggr_alpha_func_map = {
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
        }
        texture_type_map = {
            'surface': 0,
            'vertex': 1,
        }

        # Core rendering parameters
        ctx.image_size = image_size
        ctx.background_color = background_color

        ################################################################################################################
        # Distributions
        if isinstance(dist_func, int):
            ctx.dist_func = dist_func
        else:
            ctx.dist_func = dist_func_map[dist_func]
        ctx.dist_scale = dist_scale  # tau in the paper
        assert dist_scale >= 0, dist_scale  # a negative scale is invalid
        ctx.dist_squared = dist_squared  # whether the squared input / the square-root distribution is used
        ctx.dist_shape = dist_shape  # shape parameter of the distribution
        ctx.dist_shift = dist_shift  # for shifting the distribution
        ctx.dist_eps = dist_eps  # pixels further away than dist_scale*dist_eps are ignored for performance reasons
        assert dist_eps >= 1, dist_eps  # ignoring too close to the edge makes no sense
        ################################################################################################################

        ################################################################################################################
        # T-conorms
        if isinstance(aggr_alpha_func, int):
            ctx.aggr_alpha_func = aggr_alpha_func
        else:
            ctx.aggr_alpha_func = aggr_alpha_func_map[aggr_alpha_func]
        # the shape parameter p of the t-conorm (assuming the chosen one uses this param)
        ctx.aggr_alpha_t_conorm_p = aggr_alpha_t_conorm_p
        ################################################################################################################

        ################################################################################################################
        # Shading (either hard, i.e., the closest, or using softmax as proposed in Pix2Vex and SoftRas)
        if isinstance(aggr_rgb_func, int):
            ctx.aggr_rgb_func = aggr_rgb_func
        else:
            ctx.aggr_rgb_func = aggr_rgb_func_map[aggr_rgb_func]
        ctx.aggr_rgb_eps = aggr_rgb_eps
        ctx.aggr_rgb_gamma = aggr_rgb_gamma
        ################################################################################################################

        # Some additional rendering parameters
        ctx.near = near
        ctx.far = far
        ctx.double_side = double_side  # render both sides of the triangle
        ctx.texture_type = texture_type_map[texture_type]

        face_vertices = face_vertices.clone()
        textures = textures.clone()

        ctx.device = face_vertices.device
        ctx.batch_size, ctx.num_faces = face_vertices.shape[:2]

        faces_info = torch.zeros(
            (ctx.batch_size, ctx.num_faces, 9*3),
            dtype=torch.float32,
            device=ctx.device)  # [inv*9, sym*9, obt*3, 0*6]
        aggrs_info = torch.zeros(
            (ctx.batch_size, 2, ctx.image_size, ctx.image_size),
            dtype=torch.float32,
            device=ctx.device)
        soft_colors = torch.ones(
            (ctx.batch_size, 4, ctx.image_size, ctx.image_size),
            dtype=torch.float32,
            device=ctx.device)

        soft_colors[:, 0, :, :] *= background_color[0]
        soft_colors[:, 1, :, :] *= background_color[1]
        soft_colors[:, 2, :, :] *= background_color[2]

        faces_info, aggrs_info, soft_colors = \
            generalized_renderer_cuda.forward_render(
                face_vertices,
                textures,
                faces_info,
                aggrs_info,
                soft_colors,
                #
                ctx.image_size,
                #
                ctx.dist_func,
                ctx.dist_scale,
                ctx.dist_squared,
                ctx.dist_shape,
                ctx.dist_shift,
                ctx.dist_eps,
                #
                ctx.aggr_alpha_func,
                ctx.aggr_alpha_t_conorm_p,
                #
                ctx.aggr_rgb_func,
                ctx.aggr_rgb_eps,
                ctx.aggr_rgb_gamma,
                #
                ctx.near,
                ctx.far,
                ctx.double_side,
                ctx.texture_type,
       )

        ctx.save_for_backward(face_vertices, textures, soft_colors, faces_info, aggrs_info)
        return soft_colors

    @staticmethod
    def backward(ctx, grad_soft_colors):

        face_vertices, textures, soft_colors, faces_info, aggrs_info = ctx.saved_tensors

        grad_faces = torch.zeros_like(face_vertices,
                                      dtype=torch.float32,
                                      device=ctx.device)
        grad_textures = torch.zeros_like(textures,
                                         dtype=torch.float32,
                                         device=ctx.device)
        grad_soft_colors = grad_soft_colors.contiguous()

        grad_faces, grad_textures = \
            generalized_renderer_cuda.backward_render(
                face_vertices,
                textures,
                soft_colors,
                faces_info,
                aggrs_info,
                grad_faces,
                grad_textures,
                grad_soft_colors,
                #
                ctx.image_size,
                #
                ctx.dist_func,
                ctx.dist_scale,
                ctx.dist_squared,
                ctx.dist_shape,
                ctx.dist_shift,
                ctx.dist_eps,
                #
                ctx.aggr_alpha_func,
                ctx.aggr_alpha_t_conorm_p,
                #
                ctx.aggr_rgb_func,
                ctx.aggr_rgb_eps,
                ctx.aggr_rgb_gamma,
                #
                ctx.near,
                ctx.far,
                ctx.double_side,
                ctx.texture_type,
        )

        return (
            grad_faces, grad_textures,
            None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None
        )


def render(
    face_vertices,
    textures,
    #
    image_size=256,
    background_color=[0, 0, 0],
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
    double_side=True,
    texture_type='surface',
):
    if face_vertices.device == "cpu":
        raise TypeError('GenDR only supports CUDA Tensors.')

    return GenDRFunction.apply(
        face_vertices,
        textures,
        image_size,
        background_color,
        dist_func,
        dist_scale,
        dist_squared,
        dist_shape,
        dist_shift,
        dist_eps,
        aggr_alpha_func,
        aggr_alpha_t_conorm_p,
        aggr_rgb_func,
        aggr_rgb_eps,
        aggr_rgb_gamma,
        near,
        far,
        double_side,
        texture_type,
    )
