# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules = [
    CUDAExtension('gendr.cuda.load_textures', [
        'gendr/cuda/load_textures_cuda.cpp',
        'gendr/cuda/load_textures_cuda_kernel.cu',
        ]),
    CUDAExtension('gendr.cuda.create_texture_image', [
        'gendr/cuda/create_texture_image_cuda.cpp',
        'gendr/cuda/create_texture_image_cuda_kernel.cu',
        ]),
    CUDAExtension('gendr.cuda.generalized_renderer', [
        'gendr/cuda/generalized_renderer_cuda.cpp',
        'gendr/cuda/generalized_renderer_cuda_kernel.cu',
        ]),
    CUDAExtension('gendr.cuda.voxelization', [
        'gendr/cuda/voxelization_cuda.cpp',
        'gendr/cuda/voxelization_cuda_kernel.cu',
        ]),
    ]

INSTALL_REQUIREMENTS = ['numpy', 'torch>=1.9.0', 'scikit-image', 'tqdm', 'imageio']

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='gendr',
    version='v0.1.0',
    description='GenDR - The Generalized Differentiable Renderer',
    author='Felix Petersen',
    author_email='ads0475@felix-petersen.de',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Felix-Petersen/gendr',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='MIT License',
    packages=['gendr', 'gendr.cuda', 'gendr.functional'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
