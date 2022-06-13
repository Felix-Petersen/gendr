# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os
from tqdm import trange
import numpy as np
import imageio
import argparse
import math

import gendr
# from results_json import ResultsJSON


def iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return (1. - intersect / union).sum()


def mse_loss(predict, target):
    return (predict - target).pow(2).sum(0).mean()


def perspective(vertices, angle=30.):
    '''
    Compute perspective distortion from a given angle
    '''
    if (vertices.ndimension() != 3):
        raise ValueError('vertices Tensor should have 3 dimensions')
    device = vertices.device
    angle = torch.tensor(angle / 180 * math.pi, dtype=torch.float32, device=device)
    width = torch.tan(angle)
    width = width.unsqueeze(-1)
    z = vertices[:, :, 2]
    x = vertices[:, :, 0] / z / width
    y = vertices[:, :, 1] / z / width
    vertices = torch.stack((x, y, z), dim=2)
    return vertices


def transform_cameras(mesh, poses, additional_poses=None):
    """
    poses: tensor of shape [N x 4]
            the first column is the distance (>0), the second is the elevation in radians, the third the azimuth in
            radians, and the fourth is the viewing angle in degree (>0).
    """

    new_vertices = mesh.vertices.clone()

    if additional_poses is not None:
        additional_eyes = gendr.functional.get_points_from_angles(additional_poses[:, 0], additional_poses[:, 1],
                                                     additional_poses[:, 2], degrees=True)
        new_vertices = gendr.functional.look_at(new_vertices, additional_eyes, only_rotate=True)

    eyes = gendr.functional.get_points_from_angles(poses[:, 0], poses[:, 1], poses[:, 2], degrees=True)
    new_vertices = gendr.functional.look_at(new_vertices, eyes)
    new_vertices = perspective(new_vertices, poses[:, 3])

    return gendr.Mesh(new_vertices, mesh.faces.clone(),
                   mesh.textures.clone() if mesh.textures is not None else None, mesh.texture_res, mesh.texture_type)


def render(mesh, poses, renderer, additional_poses=None):
    mesh_ = lighting(mesh)
    mesh_ = transform_cameras(mesh_, poses, additional_poses=additional_poses)
    return renderer(mesh_)


def make_grid(input1, input2, grid_x, grid_y, successes=None):
    input1 = input1.detach().cpu().numpy()
    input2 = input2.detach().cpu().numpy()
    img = []
    j = 0
    for y in range(grid_y):
        row = []
        for x in range(grid_x):
            # row.append(input1[j].transpose((1, 2, 0)))
            # row.append(input2[j].transpose((1, 2, 0)))
            row.append(input1[j][3])
            row.append(input1[j][0])
            if successes is not None and successes[j]:
                row[-1][0, :] = 1.
                row[-1][:, 0] = 1.
                row[-1][-1, :] = 1.
                row[-1][:, -1] = 1.
            row.append(input2[j][0])
            j += 1
        row = np.concatenate(row, 1)
        img.append(row)
    img = np.concatenate(img, 0)
    return (255*img).astype(np.uint8)


########################################################################################################################

# python gendr/experiments/opt_camera.py -sq --gif


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    parser = argparse.ArgumentParser()
    # parser.add_argument('-eid', '--experiment_id', type=int, required=True)

    parser.add_argument('--dist-func', type=str, default='logistic')
    parser.add_argument('--aggr-func', type=str, default='probabilistic')
    parser.add_argument('--dist_shape', type=float, default=0.)
    parser.add_argument('--dist_shift', type=float, default=0.)
    parser.add_argument('--t_conorm_p', type=float, default=0.)
    parser.add_argument('-sq', '--squared', action='store_true')
    parser.add_argument('--model_obj', type=str, default='teapot.obj')

    parser.add_argument('-lr', '--learning-rate', type=float, default=0.3)
    parser.add_argument('-op', '--optimizer-choice', type=str, default='adam')
    parser.add_argument('-ni', '--num-iterations', type=int, default=1_000)
    parser.add_argument('-is', '--image-size', type=int, default=64)
    parser.add_argument('-bs', '--batch-size', type=int, default=200)
    parser.add_argument('-de', '--dist-eps', type=float, default=100)
    parser.add_argument('-lo', '--losses', type=str, nargs="+", default=['iou'])

    parser.add_argument('-gif', '--gif', action='store_true')

    args = parser.parse_args()

    # assert 390_000 < args.experiment_id < 400_000, args.experiment_id
    # results = ResultsJSON(eid=args.experiment_id, path='./results/')
    # results.store_args(args)

    seed = 0

    device = 'cuda'

    batch_size = args.batch_size
    image_size = args.image_size

    mesh_location = os.path.join(data_dir, args.model_obj)

    num_steps_decaying_sigma = args.num_iterations

    optimizer_choice = args.optimizer_choice
    learning_rate = args.learning_rate

    ####################################################################################################################

    lighting = gendr.Lighting()
    diff_renderer = gendr.GenDR(
        image_size=image_size,
        dist_func=args.dist_func,
        dist_scale=None,
        dist_squared=args.squared,
        dist_shape=args.dist_shape,
        dist_shift=args.dist_shift,
        dist_eps=args.dist_eps,
        aggr_alpha_func=args.aggr_func,
        aggr_alpha_t_conorm_p=args.t_conorm_p,
        aggr_rgb_func='hard',
    )
    hard_renderer = gendr.GenDR(
        image_size=image_size,
        dist_func=0,
        dist_scale=1e-4,
        dist_squared=True,
        dist_shape=0.,
        dist_shift=0.,
        dist_eps=10,
        aggr_alpha_func=0,
        aggr_alpha_t_conorm_p=0.,
        aggr_rgb_func='hard',
    )

    mesh = gendr.Mesh.from_obj(mesh_location)
    mesh = gendr.Mesh(mesh.vertices.repeat(batch_size, 1, 1), mesh.faces.repeat(batch_size, 1, 1))

    torch.manual_seed(seed+1)
    poses_gt = torch.nn.Parameter(torch.zeros(batch_size, 4).float().to(device))
    poses_gt.data[:, 0] = 2.5 + torch.rand(batch_size) * 1.5
    poses_gt.data[:, 1] = torch.randn(batch_size) * 60
    poses_gt.data[:, 2] = torch.randn(batch_size) * 60
    poses_gt.data[:, 3] = 20.

    print('Generating goals...')
    with torch.no_grad():
        goal = render(mesh, poses_gt, hard_renderer)
        print('done.')

    ####################################################################################################################

    threshold = 5

    ####################################################################################################################

    def execute_setting_for_gif(
        initial_angle_min,
        initial_angle_max,
        loss_fn,
    ):
        grid_x = 8
        grid_y = 12

        setting = 'a{aa}-{ab}-l{l}'.format(
            aa=initial_angle_min,
            ab=initial_angle_max,
            l=loss_fn,
        )

        torch.manual_seed(seed)
        poses = torch.nn.Parameter(torch.zeros(batch_size, 4).float().to(device))
        poses.data[:, 0] = 2. + torch.rand(batch_size) * 8.
        poses.data[:, 1] = torch.randn(batch_size)
        poses.data[:, 2] = torch.randn(batch_size)
        angles = torch.sqrt(poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2)
        initial_angle = initial_angle_min + torch.rand(batch_size).to(device) * (initial_angle_max - initial_angle_min)
        poses.data[:, 1] = poses.data[:, 1] * initial_angle / angles
        poses.data[:, 2] = poses.data[:, 2] * initial_angle / angles
        poses.data[:, 3] = 10. + torch.rand(batch_size) * 20.
        if optimizer_choice == 'adam':
            optim = torch.optim.Adam([poses], learning_rate, betas=(0.5, 0.99))
        elif optimizer_choice == 'sgd':
            optim = torch.optim.SGD([poses], learning_rate)
        else:
            raise ValueError(optimizer_choice)

        writer = imageio.get_writer(os.path.join(data_dir, 'opt_camera_{}_{}.gif'.format(
            setting, args.model_obj.split('.')[0]
        )), mode='I')

        loop = trange(num_steps_decaying_sigma)
        for i, sigma in zip(loop, np.logspace(-1, -7, num_steps_decaying_sigma)):

            diff_renderer.dist_scale = sigma

            pred = render(mesh, poses, diff_renderer, additional_poses=poses_gt)

            loss = (mse_loss if loss_fn == 'mse' else iou_loss)(pred[:, 3], goal[:, 3])

            loop.set_description('Loss: %.4f, Sigma: %g' % (loss.item(), sigma))

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 20 == 0:
                successes = poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2 < threshold ** 2
                writer.append_data(make_grid(pred, goal, grid_x, grid_y, successes))

        successes = poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2 < threshold ** 2
        for _ in range(1):
            writer.append_data(make_grid(pred * 0, goal * 0, grid_x, grid_y))
        for _ in range(10):
            writer.append_data(make_grid(pred, goal, grid_x, grid_y, successes))
        for _ in range(10):
            writer.append_data(make_grid(pred * 0, goal * 0, grid_x, grid_y))

    ####################################################################################################################

    def execute_setting(
        initial_angle_min,
        initial_angle_max,
        loss_fn,
    ):
        setting = 'a{aa}-{ab}-l{l}'.format(
            aa=initial_angle_min,
            ab=initial_angle_max,
            l=loss_fn,
        )

        torch.manual_seed(seed)
        poses = torch.nn.Parameter(torch.zeros(batch_size, 4).float().to(device))
        poses.data[:, 0] = 2. + torch.rand(batch_size) * 8.
        poses.data[:, 1] = torch.randn(batch_size)
        poses.data[:, 2] = torch.randn(batch_size)
        angles = torch.sqrt(poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2)
        initial_angle = initial_angle_min + torch.rand(batch_size).to(device) * (initial_angle_max - initial_angle_min)
        poses.data[:, 1] = poses.data[:, 1] * initial_angle / angles
        poses.data[:, 2] = poses.data[:, 2] * initial_angle / angles
        poses.data[:, 3] = 10. + torch.rand(batch_size) * 20.
        if optimizer_choice == 'adam':
            optim = torch.optim.Adam([poses], learning_rate, betas=(0.5, 0.99))
        elif optimizer_choice == 'sgd':
            optim = torch.optim.SGD([poses], learning_rate)
        else:
            raise ValueError(optimizer_choice)

        loop = trange(num_steps_decaying_sigma)
        for i, sigma in zip(loop, np.logspace(-1, -7, num_steps_decaying_sigma)):

            diff_renderer.dist_scale = sigma

            pred = render(mesh, poses, diff_renderer, additional_poses=poses_gt)

            loss = (mse_loss if loss_fn == 'mse' else iou_loss)(pred[:, 3], goal[:, 3])

            loop.set_description('Loss: %.4f, Sigma: %g' % (loss.item(), sigma))

            optim.zero_grad()
            loss.backward()
            optim.step()

            # if i % 20 == 0:
            #     successes = poses.data[:, 1]**2 + poses.data[:, 2]**2 < threshold**2
            #     print({'{}_success_{}'.format(setting, threshold): successes.float().mean().item()})
            #     # results.store_results({'{}_success_{}'.format(setting, threshold): successes.float().mean().item()})
            #     # results.store_results({
            #     #     '{}_avg_error'.format(setting):
            #     #         torch.sqrt(poses.data[:, 1]**2 + poses.data[:, 2]**2).mean().item(),
            #     #     '{}_rmse'.format(setting):
            #     #         torch.sqrt((poses.data[:, 1]**2 + poses.data[:, 2]**2).mean()).item(),
            #     # })

            if loss.isnan():
                print('Stopping the loop because loss is NaN.')
                break

        successes = poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2 < threshold ** 2
        print({'{}_success_{}'.format(setting, threshold): successes.float().mean().item()})
        # results.store_final_results({'{}_success_{}'.format(setting, threshold): successes.float().mean().item()})
        # results.store_final_results({
        #     '{}_avg_error'.format(setting): torch.sqrt(poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2).mean().item(),
        #     '{}_rmse'.format(setting): torch.sqrt((poses.data[:, 1] ** 2 + poses.data[:, 2] ** 2).mean()).item(),
        # })

        # results.s3_save()

    initial_angles = [
        (15, 35),
        (35, 55),
        (55, 75),
    ]

    for initial_angle_min, initial_angle_max in initial_angles:
        for loss_fn in args.losses:
            execute_setting(
                initial_angle_min=initial_angle_min,
                initial_angle_max=initial_angle_max,
                loss_fn=loss_fn,
            )
            if args.gif:
                execute_setting_for_gif(
                    initial_angle_min=initial_angle_min,
                    initial_angle_max=initial_angle_max,
                    loss_fn=loss_fn,
                )

