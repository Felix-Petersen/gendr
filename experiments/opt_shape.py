# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
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
    return (1. - intersect / union).mean()


def mse_loss(predict, target):
    return (predict - target).pow(2).mean()


def make_grid(input1, input2, grid_x, grid_y):
    input1 = input1.detach().cpu().numpy()
    input2 = input2.detach().cpu().numpy()
    img = []
    j = 0
    for y in range(grid_y):
        row = []
        for x in range(grid_x):
            # row.append(input1[j].transpose((1, 2, 0)))
            # row.append(input2[j].transpose((1, 2, 0)))
            row.append(input1[j])
            row.append(input2[j])
            j += 1
        row = np.concatenate(row, 1)
        img.append(row)
    img = np.concatenate(img, 0)
    return (255*img).astype(np.uint8)


class Model(nn.Module):
    def __init__(self, num_vertices=1352):
        super(Model, self).__init__()

        template_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'),
                                     'sphere_{}.obj'.format(num_vertices))

        # set template mesh
        self.template_mesh = gendr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
        self.register_buffer('faces', self.template_mesh.faces)
        self.register_buffer('textures', self.template_mesh.textures)

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.template_mesh.vertices)))
        self.register_parameter('center', nn.Parameter(torch.zeros(1, 1, 3)))

        # define Laplacian and flatten geometry constraints
        self.laplacian_loss = gendr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = gendr.FlattenLoss(self.faces[0].cpu())

    def reset(self):
        self.displace.data = torch.zeros_like(self.displace.data)
        self.center.data = torch.zeros_like(self.center.data)

    def forward(self, batch_size):
        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs()))
        centroid = torch.tanh(self.center)
        vertices = torch.sigmoid(base + self.displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return gendr.Mesh(vertices.repeat(batch_size, 1, 1),
                       self.faces.repeat(batch_size, 1, 1)), laplacian_loss, flatten_loss


########################################################################################################################

# python gendr/experiments/opt_shape.py -sq --gif


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
    parser.add_argument('--model_obj', type=str, default='airplane.obj')

    parser.add_argument('-op', '--optimizer-choice', type=str, default='adam')
    parser.add_argument('-ni', '--num-iterations', type=int, default=100)
    parser.add_argument('-nv', '--num-vertices', type=int, default=642, choices=[642, 1352])
    parser.add_argument('-is', '--image-size', type=int, default=64)
    parser.add_argument('-de', '--dist-eps', type=float, default=100)
    parser.add_argument('-lo', '--loss', type=str, default='iou', choices=['mse', 'iou'])
    parser.add_argument('-lt', '--loss-threshold', type=float, default=.1)
    parser.add_argument('-cr', '--criterion', type=str, default='loss', choices=['loss', 'steps_to_threshold'])

    parser.add_argument('-gif', '--gif', action='store_true')

    args = parser.parse_args()

    # assert 390_000 < args.experiment_id < 400_000, args.experiment_id
    # results = ResultsJSON(eid=args.experiment_id, path='./results/')
    # results.store_args(args)

    seed = 0

    device = 'cuda'

    image_size = args.image_size

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
        dist_eps=1,
        aggr_alpha_func=0,
        aggr_alpha_t_conorm_p=0.,
        aggr_rgb_func='hard',
    )
    transform = gendr.LookAt(viewing_angle=15)

    all_cameras = np.load(os.path.join(data_dir, 'cameras.npy')).astype('float32')
    print('all_cameras.shape', all_cameras.shape)

    # if args.model_obj == 'airplane_from_images.npy':
    #     all_images = (np.load(os.path.join(data_dir, 'images.npy')).astype('float32') / 255.)[:, 3]
    # else:
    with torch.no_grad():
        print('Generating goals...')
        camera_distances = torch.from_numpy(all_cameras[:, 0])
        elevations = torch.from_numpy(all_cameras[:, 1])
        viewpoints = torch.from_numpy(all_cameras[:, 2])
        transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

        mesh = gendr.Mesh.from_obj(os.path.join(data_dir, args.model_obj))
        mesh = gendr.Mesh(mesh.vertices.repeat(120, 1, 1), mesh.faces.repeat(120, 1, 1))
        mesh = lighting(mesh)
        mesh = transform(mesh)
        all_images = hard_renderer(mesh)[:, 3]
        all_images = all_images.cpu().numpy()
        print('done.')
        print('all_images.shape', all_images.shape)

    ####################################################################################################################

    for views in ['24@-60', '24@-30', '24@0', '24@30', '24@60']:

        print(views)

        if views == 'all':
            pass
        elif views == '4@30':
            images = all_images[3*24:4*24]
            cameras = all_cameras[3*24:4*24]
            images = images[::6]
            cameras = cameras[::6]
        elif views == '8@30':
            images = all_images[3*24:4*24]
            cameras = all_cameras[3*24:4*24]
            images = images[::3]
            cameras = cameras[::3]
        elif views == '24@-60':
            j = 0
            images = all_images[j*24:(j+1)*24]
            cameras = all_cameras[j*24:(j+1)*24]
        elif views == '24@-30':
            j = 1
            images = all_images[j*24:(j+1)*24]
            cameras = all_cameras[j*24:(j+1)*24]
        elif views == '24@0':
            j = 2
            images = all_images[j*24:(j+1)*24]
            cameras = all_cameras[j*24:(j+1)*24]
        elif views == '24@30':
            j = 3
            images = all_images[j*24:(j+1)*24]
            cameras = all_cameras[j*24:(j+1)*24]
        elif views == '24@60':
            j = 4
            images = all_images[j*24:(j+1)*24]
            cameras = all_cameras[j*24:(j+1)*24]
        else:
            raise ValueError(args.views)

        camera_distances = torch.from_numpy(cameras[:, 0])
        elevations = torch.from_numpy(cameras[:, 1])
        viewpoints = torch.from_numpy(cameras[:, 2])
        transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

        images_gt = torch.from_numpy(images).to(device)

        model = Model(args.num_vertices).to(device)

        ################################################################################################################

        def execute_setting_for_gif(
            learning_rate,
            sigma,
        ):

            model.reset()
            if args.optimizer_choice == 'adam':
                optim = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.5, 0.95))
            elif args.optimizer_choice == 'sgd':
                optim = torch.optim.SGD(model.parameters(), learning_rate)
            else:
                raise ValueError(args.optimizer_choice)

            diff_renderer.dist_scale = sigma

            writer = imageio.get_writer(os.path.join(data_dir, 'shape_{}_{}.gif'.format(
                views, args.model_obj.split('.')[0]
            )), mode='I')
            for i in range(args.num_iterations):
                mesh, laplacian_loss, flatten_loss = model(len(cameras))

                mesh = lighting(mesh)
                mesh = transform(mesh)
                images_pred = diff_renderer(mesh)[:, 3]

                sil_loss = (mse_loss if args.loss == 'mse' else iou_loss)(images_pred, images_gt)

                loss = sil_loss + (0.03 * laplacian_loss + 0.0003 * flatten_loss)

                optim.zero_grad()
                loss.backward()
                optim.step()

                writer.append_data(make_grid(images_pred, images_gt, 4, 6))

        ################################################################################################################

        def execute_setting(
            learning_rate,
            sigma,
        ):

            steps_to_threshold = int(1e10)
            hard_loss = 1e10

            model.reset()
            if args.optimizer_choice == 'adam':
                optim = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.5, 0.95))
            elif args.optimizer_choice == 'sgd':
                optim = torch.optim.SGD(model.parameters(), learning_rate)
            else:
                raise ValueError(args.optimizer_choice)

            diff_renderer.dist_scale = sigma

            for i in range(args.num_iterations):
                mesh, laplacian_loss, flatten_loss = model(len(cameras))

                mesh = lighting(mesh)
                mesh = transform(mesh)
                images_pred = diff_renderer(mesh)[:, 3]

                sil_loss = (mse_loss if args.loss == 'mse' else iou_loss)(images_pred, images_gt)

                with torch.no_grad():
                    images_pred = hard_renderer(mesh)[:, 3]
                    hard_sil_loss = (mse_loss if args.loss == 'mse' else iou_loss)(images_pred, images_gt)
                    hard_loss = min(hard_loss, hard_sil_loss)
                    if hard_loss < args.loss_threshold:
                        steps_to_threshold = min(i, steps_to_threshold)

                loss = sil_loss + (0.03 * laplacian_loss + 0.0003 * flatten_loss)

                optim.zero_grad()
                loss.backward()
                optim.step()

            if args.criterion == 'loss':
                return hard_loss.item()
            elif args.criterion == 'steps_to_threshold':
                return steps_to_threshold
            else:
                raise ValueError(args.criterion)

        ################################################################################################################

        best_setting = [None, None, 1e10]

        assert args.optimizer_choice == 'adam', args.optimizer_choice

        for learning_rate in tqdm(np.logspace(-1.25, -1.75, 3)):
            for sigma in np.logspace(-1, -7, 7):
                res = execute_setting(learning_rate, sigma)
                if res < best_setting[2]:
                    best_setting = [learning_rate, sigma, res]

        learning_rate = best_setting[0]
        range_ = np.logspace(math.log10(best_setting[1])-1, math.log10(best_setting[1])+1, 21)
        for sigma in tqdm(range_):
            res = execute_setting(learning_rate, sigma)
            if res < best_setting[2]:
                best_setting = [learning_rate, sigma, res]

        print({
            'learning_rate_{}'.format(views): best_setting[0],
            'sigma_{}'.format(views): best_setting[1],
            args.criterion + '_{}'.format(views): best_setting[2],
        })

        # results.store_final_results({
        #     'learning_rate_{}'.format(views): best_setting[0],
        #     'sigma_{}'.format(views): best_setting[1],
        #     args.criterion + '_{}'.format(views): best_setting[2],
        # })

        if args.gif:
            execute_setting_for_gif(best_setting[0], best_setting[1])

    # results.s3_save()

