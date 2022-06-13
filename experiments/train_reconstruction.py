# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import torchvision
import numpy as np
import math
import time
import warnings
import random
import tqdm
import os


import gendr

# from results_json import ResultsJSON


torch.set_num_threads(min(8, torch.get_num_threads()))


########################################################################################################################
# Utils ################################################################################################################
########################################################################################################################

def iou(predict, target, eps=1e-6):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()


def iou_loss(predict, target):
    return 1 - iou(predict, target)


def multiview_iou_loss(predicts, targets_a, targets_b):
    loss = (iou_loss(predicts[0][:, 3], targets_a[:, 3]) +
            iou_loss(predicts[1][:, 3], targets_a[:, 3]) +
            iou_loss(predicts[2][:, 3], targets_b[:, 3]) +
            iou_loss(predicts[3][:, 3], targets_b[:, 3])) / 4
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def img_cvt(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)


def adjust_learning_rate(optimizers, learning_rate, i):
    lr = learning_rate
    if i >= 150000:
        lr *= 0.3

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def adjust_dist_scale(dist_scale, i):
    if i >= 150000:
        dist_scale *= 0.3
    return dist_scale


########################################################################################################################
# Models ###############################################################################################################
########################################################################################################################

class Encoder(torch.nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024, im_size=64):
        super(Encoder, self).__init__()
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, dim2]

        self.conv1 = torch.nn.Conv2d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=2)

        self.bn1 = torch.nn.BatchNorm2d(dim_hidden[0])
        self.bn2 = torch.nn.BatchNorm2d(dim_hidden[1])
        self.bn3 = torch.nn.BatchNorm2d(dim_hidden[2])

        self.fc1 = torch.nn.Linear(dim_hidden[2]*math.ceil(im_size/8)**2, dim_hidden[3])
        self.fc2 = torch.nn.Linear(dim_hidden[3], dim_hidden[4])
        self.fc3 = torch.nn.Linear(dim_hidden[4], dim_out)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = gendr.Mesh.from_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])  # vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])  # faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = torch.nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = torch.nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = torch.nn.Linear(dim_hidden[1], 3)
        self.fc_bias = torch.nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = torch.relu(vertices) * scale_pos - torch.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces


class Model(torch.nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        # auto-encoder
        self.encoder = Encoder(im_size=args.image_size)
        self.decoder = Decoder(filename_obj)

        # renderer
        self.transform = gendr.LookAt(viewing_angle=15)
        self.lighting = gendr.Lighting()
        self.renderer = gendr.GenDR(
            image_size=args.image_size,
            #
            dist_func=args.distribution,
            dist_scale=args.dist_scale,
            dist_squared=args.squared,
            dist_shape=args.dist_shape,
            dist_shift=args.dist_shift,
            dist_eps=args.dist_eps,
            #
            aggr_alpha_func=args.t_conorm,
            aggr_alpha_t_conorm_p=args.t_conorm_p,
            #
            aggr_rgb_func='hard',
        )

        # mesh regularizer
        self.laplacian_loss = gendr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = gendr.FlattenLoss(self.decoder.faces)

    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def set_dist_scale(self, dist_scale):
        self.renderer.dist_scale = dist_scale

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces

    def render_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b):
        # [Ia, Ib]
        images = torch.cat((image_a, image_b), dim=0)
        # [Va, Va, Vb, Vb], set viewpoints
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        self.transform.set_eyes(viewpoints)

        vertices, faces = self.reconstruct(images)
        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)

        # [Raa, Rba, Rab, Rbb], render for cross-view consistency
        mesh = gendr.Mesh(vertices, faces)
        mesh = self.lighting(mesh)
        mesh = self.transform(mesh)
        silhouettes = self.renderer(mesh)
        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.reconstruct(images)

        faces_ = gendr.functional.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = gendr.functional.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        return iou, vertices, faces

    def forward(self, images=None, viewpoints=None, voxels=None, task='train'):
        if task == 'train':
            return self.render_multiview(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == 'test':
            return self.evaluate_iou(images, voxels)


########################################################################################################################
# Dataset ##############################################################################################################
########################################################################################################################

class_ids_map = {
    '02691156': 'Airplane',
    '02828884': 'Bench',
    '02933112': 'Cabinet',
    '02958343': 'Car',
    '03001627': 'Chair',
    '03211117': 'Display',
    '03636649': 'Lamp',
    '03691459': 'Loudspeaker',
    '04090263': 'Rifle',
    '04256520': 'Sofa',
    '04379243': 'Table',
    '04401088': 'Telephone',
    '04530566': 'Watercraft',
}


class ShapeNet(object):
    url = 'https://nyc3.digitaloceanspaces.com/publicdata1/mesh_reconstruction_dataset.zip'
    md5 = 'cde0353519427992c1183232ebff74e8'

    def __init__(self, root, class_ids=None, set_name=None, download=False):

        self.root = root

        if download:
            self.download()

        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        self.class_ids_map = class_ids_map

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(list(np.load(
                os.path.join(root, 'mesh_reconstruction', '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])
            voxels.append(list(np.load(
                os.path.join(root, 'mesh_reconstruction', '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    def download(self):
        torchvision.datasets.utils.download_and_extract_archive(self.url, self.root, md5=self.md5)

    @property
    def class_ids_pair(self):
        class_names = [self.class_ids_map[i] for i in self.class_ids]
        return zip(self.class_ids, class_names)

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = gendr.functional.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 15)
        viewpoints_b = gendr.functional.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 15)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids

        for i in range((data_ids.size - 1) // batch_size + 1):
            images = torch.from_numpy(
                self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.)
            voxels = torch.from_numpy(
                self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] // 24].astype('float32'))
            yield images, voxels


########################################################################################################################
# Training #############################################################################################################
########################################################################################################################

def train():
    end = time.time()
    batch_time = AverageMeter()
    losses = AverageMeter()

    for i in tqdm.trange(1, args.num_iterations + 1):
        # adjust learning rate and dist_scale (decay after 150k iter)
        lr = adjust_learning_rate([optimizer], args.learning_rate, i)
        model.set_dist_scale(adjust_dist_scale(args.dist_scale, i))

        # load images from multi-view
        images_a, images_b, viewpoints_a, viewpoints_b = dataset_train.get_random_batch(args.batch_size)
        images_a = images_a.cuda()
        images_b = images_b.cuda()
        viewpoints_a = viewpoints_a.cuda()
        viewpoints_b = viewpoints_b.cuda()

        # soft render images
        render_images, laplacian_loss, flatten_loss = model([images_a, images_b],
                                                            [viewpoints_a, viewpoints_b],
                                                            task='train')
        laplacian_loss = laplacian_loss.mean()
        flatten_loss = flatten_loss.mean()

        # compute loss
        loss = multiview_iou_loss(render_images, images_a, images_b) + \
            args.lambda_laplacian * laplacian_loss + \
            args.lambda_flatten * flatten_loss
        losses.update(loss.data.item(), images_a.size(0))

        # compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # print
        if i % args.print_freq == 0:
            print('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Loss {loss.val:.3f}\t'
                  'lr {lr:.6f}\t'
                  'sv {sv:.6f}\t'.format(i, args.num_iterations,
                                         batch_time=batch_time, loss=losses,
                                         lr=lr, sv=model.renderer.dist_scale))

        if i % args.eval_freq == 0:
            model.eval()

            # VALIDATION
            iou_all = []
            for class_id, class_name in dataset_val.class_ids_pair:
                iou = 0

                for i, (im, vx) in enumerate(
                        dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
                    images = im

                    images = torch.autograd.Variable(images).cuda()
                    voxels = vx.numpy()

                    batch_iou, vertices, faces = model(images, voxels=voxels, task='test')
                    iou += batch_iou.sum().item()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    # print loss
                    if i % args.print_freq == 0:
                        print('Iter: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f}\t'
                              'IoU {2:.3f}\t'.format(i, ((dataset_val.num_data[class_id] * 24) // args.batch_size),
                                                     batch_iou.mean(),
                                                     batch_time=batch_time))

                iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
                iou_all.append(iou_cls)
                print('Mean Valid IoU: {:.3f} for class {}'.format(iou_cls, class_name))
            print('Mean Valid IoU: {:.3f} for all classes'.format(sum(iou_all) / len(iou_all)))
            print()

            # iou_all = iou_all + [sum(iou_all) / len(iou_all)]
            #
            # results.store_results(dict(iou_val=iou_all))

            # TEST
            iou_all = []
            for class_id, class_name in dataset_test.class_ids_pair:
                iou = 0

                for i, (im, vx) in enumerate(
                        dataset_test.get_all_batches_for_evaluation(args.batch_size, class_id)):
                    images = im

                    images = torch.autograd.Variable(images).cuda()
                    voxels = vx.numpy()

                    batch_iou, vertices, faces = model(images, voxels=voxels, task='test')
                    iou += batch_iou.sum().item()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    # print loss
                    if i % args.print_freq == 0:
                        print('Iter: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f}\t'
                              'IoU {2:.3f}\t'.format(i, ((dataset_test.num_data[class_id] * 24) // args.batch_size),
                                                     batch_iou.mean(),
                                                     batch_time=batch_time))

                iou_cls = iou / 24. / dataset_test.num_data[class_id] * 100
                iou_all.append(iou_cls)
                print('Mean Test IoU: {:.3f} for class {}'.format(iou_cls, class_name))
            print('Mean Test IoU: {:.3f} for all classes'.format(sum(iou_all) / len(iou_all)))
            print()

            # iou_all = iou_all + [sum(iou_all) / len(iou_all)]
            #
            # results.store_results(dict(iou_test=iou_all))
            #
            # results.s3_save()

            model.train()


########################################################################################################################
# Main #################################################################################################################
########################################################################################################################

# python gendr/experiments/train_reconstruction.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-eid', '--experiment_id', type=int, required=True)
    parser.add_argument('--class_ids', type=str, default='02691156,02828884,02933112,02958343,03001627,'
                                                         '03211117,03636649,03691459,04090263,04256520,'
                                                         '04379243,04401088,04530566')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-ni', '--num_iterations', type=int, default=250_000)
    parser.add_argument('--print_freq', type=int, default=1_000)
    parser.add_argument('--eval_freq', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--distribution', type=str, default='uniform')
    parser.add_argument('-sq', '--squared', action='store_true')
    parser.add_argument('--dist_scale', type=float, default=None)
    parser.add_argument('--dist_shape', type=float, default=0)
    parser.add_argument('--dist_shift', type=float, default=0)
    parser.add_argument('--dist_eps', type=float, default=300.)

    parser.add_argument('--t_conorm', type=str, default='probabilistic')
    parser.add_argument('--t_conorm_p', type=float, default=0)

    parser.add_argument('--lambda_laplacian', type=float, default=5e-3)
    parser.add_argument('--lambda_flatten', type=float, default=5e-4)
    args = parser.parse_args()

    distribution = args.distribution + ('_squares' if args.squared else '')
    t_conorm = args.t_conorm + '_{:.1f}'.format(args.t_conorm_p)

    if args.dist_scale is None:
        print('Using default `dist_scale` for {} distribution and {} T-conorm.'.format(
            args.distribution, args.t_conorm))
        distributions_with_default_scale = [
            'uniform',
            'gaussian',
            'logistic',
            'logistic_squares',
            'cauchy',
            'cauchy_squares',
            'gumbel_min',
            'gamma_rev',
            'gamma_rev_squares',
            'exponential_rev',
        ]
        assert distribution in distributions_with_default_scale, 'Default for {} distribution unknown as not ' \
                                                                 'in {}.'.format(distribution,
                                                                                 distributions_with_default_scale)
        t_conorms_with_default_scale = [
            'probabilistic_0.0',
            'einstein_0.0',
            'yager_2.0',
        ]
        assert t_conorm in t_conorms_with_default_scale, 'Default for {} t-conorm unknown as not in {}.'.format(
            t_conorm, t_conorms_with_default_scale
        )
        default_log_scales = torch.tensor(
            [[-1.5000, -1.5000, -1.5000],
             [-1.5000, -1.5000, -2.0000],
             [-2.0000, -2.0000, -2.0000],
             [-4.0000, -4.0000, -4.0000],
             [-3.5000, -3.5000, -3.0000],
             [-4.5000, -4.5000, -4.0000],
             [-2.0000, -2.5000, -2.0000],
             [-2.0000, -2.0000, -2.0000],
             [-4.0000, -4.0000, -3.5000],
             [-2.0000, -2.0000, -2.0000]]
        )
        log_scale = default_log_scales[
            distributions_with_default_scale.index(distribution),
            t_conorms_with_default_scale.index(t_conorm)
        ].item()
        dist_scale = 10**(log_scale)
        print('Retrieved default log_scale of {} corresponding to a dist_scale of {}.'.format(log_scale, dist_scale))
        args.dist_scale = dist_scale

    # assert 380_000 < args.experiment_id < 390_000, args.experiment_id
    # results = ResultsJSON(eid=args.experiment_id, path='./results/')
    # results.store_args(args)

    print(vars(args))

    if args.seed != 0:
        warnings.warn(
            'You set seed to {}. However, notice that the backward of the renderer uses a non-deterministic component. '
            'Thus, the seed can not ensure determinism.'.format(args.seed)
        )

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model = Model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/sphere_642.obj'), args=args)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.model_param(), args.learning_rate)

    dataset_train = ShapeNet('./data-shapenet', args.class_ids.split(','), 'train', download=True)
    dataset_val = ShapeNet('./data-shapenet', args.class_ids.split(','), 'val')
    dataset_test = ShapeNet('./data-shapenet', args.class_ids.split(','), 'test')

    train()
