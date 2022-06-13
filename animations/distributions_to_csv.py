# Copyright (c) Felix Petersen.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import gendr.cuda.generalized_renderer as generalized_renderer_cuda
import math


def sigmoid_forward(
        function_id,
        sign,
        x,
        scale=1.,
        param1=-10.,
        param2=-10.,
):
    return generalized_renderer_cuda.sigmoid_forward(
        function_id,
        sign,
        x,
        scale,
        param1,
        param2,
    )


def sigmoid_forward_(
        function_id,
        xs,
        scale=1.,
        param1=-10.,
        param2=-10.,
):
    result = []
    for x in xs:
        result.append(sigmoid_forward(function_id, math.copysign(1, x), abs(x), scale, param1, param2))
    return result


def sigmoid_backward(
        function_id,
        sign,
        x,
        scale=1.,
        param1=-10.,
        param2=-10.,
):
    return generalized_renderer_cuda.sigmoid_backward(
        function_id,
        sign,
        x,
        scale,
        param1,
        param2,
    )


def sigmoid_backward_(
        function_id,
        xs,
        scale=1.,
        param1=-10.,
        param2=-10.,
):
    result = []
    for x in xs:
        result.append(sigmoid_backward(function_id, math.copysign(1, x), abs(x), scale, param1, param2))
    return result


sigmoid_functions = [
    ('uniform', 0),
    ('gaussian', 0),  # Gaussian
    ('logistic', 0),  # Logistic
    ('laplace', 0),  # Laplace
    ('cubic_hermite', 0),  # Cubic Hermite
    ('cauchy', 0),  # Cauchy
    ('gamma', 2.),  # Gamma w/ p=2 p2=0
    ('gamma', .5),  # Gamma w/ p=.5 p2=0
    ('gamma_rev', 2.),  # Neg-Gamma w/ p=2 p2=0
    ('gamma_rev', .5),  # Neg-Gamma w/ p=.5 p2=0
]


if __name__ == '__main__':
    xs = np.linspace(-5, 5, 201)

    results = [xs]

    func_dist_map = {
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

    for i, p in sigmoid_functions:
        # if sq:
        #     xs_ = np.sign(xs) * xs**2
        # else:
        xs_ = xs

        if i in ['uniform', 'cubic_hermite', 'wigner_semicircle']:
            xs_ = xs_ / 2

        if i in ['levy', 'levy_rev']:
            xs_ = xs_ * 3

        if i in ['levy', 'levy_rev']:
            results.append(sigmoid_forward_(func_dist_map[i], xs_, scale=2, param1=p, param2=0))
        else:
            results.append(sigmoid_forward_(func_dist_map[i], xs_, scale=1, param1=p, param2=0))

    results = np.vstack(results).T

    print(results.shape)

    np.savetxt('dist_function_values.csv', results, delimiter=",")

