# ๐งช Experiments

## ๐ผ Shape Optimization (`opt_shape.py`)

```shell
python experiments/opt_shape.py -sq --gif
```

## ๐ฝ Camera Pose Optimization (`opt_camera.py`)

```shell
python experiments/opt_camera.py -sq --gif
```

## โ๏ธ Single-View 3D Reconstruction (`train_reconstruction.py`)

Optimal default parameters for `--dist_scale` are automatically used in the script for the set of distributions
and t-conorms that are benchmarked on this task in the paper.

```shell
python experiments/train_reconstruction.py --distribution uniform --t_conorm probabilistic
```
