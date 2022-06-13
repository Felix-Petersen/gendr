# GenDR - The Generalized Differentiable Renderer

![gendr_logo](gendr_logo.png)

Official implementation for our CVPR 2022 Paper "GenDR: A Generalized Differentiable Renderer".

Paper @ [ArXiv](https://arxiv.org/abs/2204.13845),
Video @ [Youtube](https://youtu.be/p-ZCcUWzriE).

## 💻 Installation

`gendr` can be installed via pip from PyPI with
```shell
pip install gendr
```
> ⚠️ Note that `gendr` requires CUDA, the CUDA Toolkit (for compilation), and `torch>=1.9.0` (matching the CUDA version).

Alternatively, GenDR may be installed from source, e.g., in a virtual environment like
```shell
virtualenv -p python3 .env1
. .env1/bin/activate
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install .
```
Make sure that the CUDA version of PyTorch (e.g., `cu111` for CUDA 11.1) matches the locally installed version.
However, on some machines, compiling works only with specific subversions that may be unequal to the local subversion, 
so a potential quick fix is trying different PyTorch version and CUDA subversion combinations.

## 👩‍💻 Documentation

A differentiable renderer may be defined as follows

```python
import gendr

diff_renderer = gendr.GenDR(
    image_size=256,
    dist_func='uniform',
    dist_scale=0.01,
    dist_squared=False,
    aggr_alpha_func='probabilistic',
    aggr_rgb_func='hard',
)
```

In the following, we provide the entire set of arguments of `GenDR`.
The most important parameters are marked in **bold**.
For the essential parameters `dist_func` and `aggr_alpha_func`, we give a set of options. 
For a reference, see [the paper](https://arxiv.org/abs/2204.13845).

* `image_size` **the size of the rendered image** (default: 256)
* `background_color` (default: [0, 0, 0])
* `anti_aliasing` render it at 2x the resolution and average to reduce aliasing (default: False)

* `dist_func` **the distribution used for the differentiable occlusion test** (default: uniform)
  * `hard` hard, non-differentiable rendering, Dirac delta distribution, Heaviside function (alias `heaviside`)
  * `uniform` uniform distribution
  * `cubic_hermite` Cubic-Hermite sigmoid function
  * `wigner_semicircle` Wigner Semicircle distribution
  * `gaussian` Gaussian Distribution
  * `laplace` Laplace Distribution
  * `logistic` logistic Distribution
  * `gudermannian` Gudermannian function, hyperbolic secant distribution (alias `hyperbolic_secant`)
  * `cauchy` Cauchy distribution
  * `reciprocal` reciprocal sigmoid function
  * `gumbel_max` Gumbel-max distribution
  * `gumbel_min` Gumbel-min distribution
  * `exponential` exponential distribution
  * `exponential_rev` exponential distribution (reversed / mirrored)
  * `gamma` gamma distribution
  * `gamma_rev` gamma distribution (reversed / mirrored)
  * `levy` Levy distribution
  * `levy_rev` Levy distribution (reversed / mirrored)
* `dist_scale` **the scale parameter of the distribution, tau in the paper** (default: 1e-2)
* `dist_squared` optionally, use the square-root distribution of `dist_func` (default: False)
* `dist_shape` for some distributions, we need a shape parameter (default: None)
* `dist_shift` for some distributions, we need an optional shift parameter (default: None or 0)
* `dist_eps` pixels further away than `dist_scale*dist_eps` are ignored for performance reasons (default: 1e4)

* `aggr_alpha_func` **the t-conorm used to aggregate occlusion values** (default: probabilistic)
  * `hard` to be used with `dist_func='hard'`
  * `max` maximum T-conorm
  * `probabilistic` probabilistic T-conorm
  * `einstein` Einstein sum T-conorm
  * `hamacher` Hamacher T-conorm
  * `frank` Frank T-conorm
  * `yager` Yager T-conorm
  * `aczel_alsina` Aczel-Alsina T-conorm
  * `dombi` Dombi T-conorm
  * `schweizer_sklar` Schweizer-Sklar T-conorm
* `aggr_alpha_t_conorm_p` for some t-conorms, we need a shape parameter (default: None)

* `aggr_rgb_func` (default: softmax)
* `aggr_rgb_eps` (default: 1e-3)
* `aggr_rgb_gamma` (default: 1e-3)

* `near` value for the viewing frustum (default: 1)
* `far` value for the viewing frustum (default: 100)
* `double_side` render all faces from both sides (default: False)
* `texture_type` type of texture sampling (default: surface; options: surface, vertex)


## 🧪 Experiments

### 🐼 Shape Optimization (`opt_shape.py`)

```shell
python experiments/opt_shape.py -sq --gif
```

### 📽 Camera Pose Optimization (`opt_camera.py`)

```shell
python experiments/opt_camera.py -sq --gif
```

### ✈️ Single-View 3D Reconstruction (`train_reconstruction.py`)

Optimal default parameters for `--dist_scale` are automatically used in the script for the set of distributions
and t-conorms that are benchmarked on this task in the paper.

```shell
python experiments/train_reconstruction.py --distribution uniform --t_conorm probabilistic
```

## 📖 Citing

```bibtex
@inproceedings{petersen2022gendr,
  title={{GenDR: A Generalized Differentiable Renderer}},
  author={Petersen, Felix and Goldluecke, Bastian and Borgelt, Christian and Deussen, Oliver},
  booktitle={IEEE/CVF International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```

## License

`gendr` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.

