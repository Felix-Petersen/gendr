/* Copyright (c) Felix Petersen.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/torch.h>

#include <vector>

#include <iostream>

// CUDA forward declarations

std::vector<at::Tensor> forward_render_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        int dist_func,
        float dist_scale,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type
);


std::vector<at::Tensor> backward_render_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,        
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        int dist_func,
        float dist_scale,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> forward_render(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        int dist_func,
        float dist_scale,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type
) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(soft_colors);

    return forward_render_cuda(
        faces,
        textures,
        faces_info,
        aggrs_info,
        soft_colors,
        image_size,
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
        texture_type
    );
}


std::vector<at::Tensor> backward_render(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        int dist_func,
        float dist_scale,
        bool dist_squared,
        float dist_shape,
        float dist_shift,
        float dist_eps,
        int aggr_alpha_func,
        float aggr_alpha_t_conorm_p,
        int aggr_rgb_func,
        float aggr_rgb_eps,
        float aggr_rgb_gamma,
        float near,
        float far,
        bool double_side,
        int texture_type
) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(soft_colors);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(grad_faces);
    CHECK_INPUT(grad_textures);
    CHECK_INPUT(grad_soft_colors);

    return backward_render_cuda(
        faces,
        textures,
        soft_colors,
        faces_info,
        aggrs_info,
        grad_faces,
        grad_textures,
        grad_soft_colors,
        image_size,
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
        texture_type
    );
}


float sigmoid_forward_float(
        int function_id,
        float sign,
        float x,
        float scale,
        float dist_shape,
        float dist_shift
);

float sigmoid_backward_float(
        int function_id,
        float sign,
        float x,
        float scale,
        float dist_shape,
        float dist_shift
);

float t_conorm_forward_float(
        int t_conorm_id,
        float a_existing,
        float b_new,
        int face_id,
        float t_conorm_p
);

float t_conorm_backward_float(
        int t_conorm_id,
        float a_all,
        float b_current,
        int number_of_faces,
        float t_conorm_p
);


PYBIND11_MODULE(generalized_renderer, m) {
    m.def("forward_render", &forward_render, "FORWARD_RENDER (CUDA)");
    m.def("backward_render", &backward_render, "BACKWARD_RENDER (CUDA)");
    m.def("sigmoid_forward", &sigmoid_forward_float);
    m.def("sigmoid_backward", &sigmoid_backward_float);
    m.def("t_conorm_forward", &t_conorm_forward_float);
    m.def("t_conorm_backward", &t_conorm_backward_float);
}
