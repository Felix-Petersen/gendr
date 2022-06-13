import torch
import torch.nn as nn

# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gendr


class AmbientLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1, 1, 1)):
        super(AmbientLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color

    def forward(self, light):
        return gendr.functional.ambient_lighting(light, self.light_intensity, self.light_color)


class DirectionalLighting(nn.Module):
    def __init__(self, light_intensity=0.5, light_color=(1, 1, 1), light_direction=(0, 1, 0)):
        super(DirectionalLighting, self).__init__()

        self.light_intensity = light_intensity
        self.light_color = light_color
        self.light_direction = light_direction

    def forward(self, light, normals):
        return gendr.functional.directional_lighting(light, normals,
                                        self.light_intensity, self.light_color,
                                        self.light_direction)


class Lighting(nn.Module):
    def __init__(self, intensity_ambient=0.5, color_ambient=[1, 1, 1],
                 intensity_directionals=0.5, color_directionals=[1, 1, 1],
                 directions=[0, 1, 0]):
        super(Lighting, self).__init__()

        self.ambient = AmbientLighting(intensity_ambient, color_ambient)
        self.directionals = nn.ModuleList([DirectionalLighting(intensity_directionals,
                                                               color_directionals,
                                                               directions)])

    def forward(self, mesh):
        if mesh.texture_type == 'surface':
            light = torch.zeros_like(mesh.faces, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.ambient(light)
            for directional in self.directionals:
                light = directional(light, mesh.surface_normals)
            new_textures = mesh.textures * light[:, :, None, :]

        elif mesh.texture_type == 'vertex':
            light = torch.zeros_like(mesh.vertices, dtype=torch.float32).to(mesh.device)
            light = light.contiguous()
            light = self.ambient(light)
            for directional in self.directionals:
                light = directional(light, mesh.vertex_normals)
            new_textures = mesh.textures * light

        vertices = mesh.vertices
        faces = mesh.faces
        texture_res = mesh.texture_res
        texture_type = mesh.texture_type
        mesh = gendr.Mesh(vertices, faces, new_textures, texture_res, texture_type)

        return mesh
