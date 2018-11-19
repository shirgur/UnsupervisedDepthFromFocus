import torch
import torch.nn as nn
from torch.autograd import Function

import gauss_psf


class GaussPSFFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, kernel_size=7):
        with torch.no_grad():
            x = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(kernel_size, 1).float().repeat(1, kernel_size).cuda()

            y = torch.arange(kernel_size // 2,
                             -kernel_size // 2,
                             -1).view(1, kernel_size).float().repeat(kernel_size, 1).cuda()

        outputs, wsum = gauss_psf.forward(input, weights, x, y)
        ctx.save_for_backward(input, outputs, weights, wsum, x, y)

        return outputs

    @staticmethod
    def backward(ctx, grad):
        input, outputs, weights, wsum, x, y = ctx.saved_variables
        x = -x
        y = -y
        grad_input, grad_weights = gauss_psf.backward(grad.contiguous(), input, outputs, weights, wsum, x, y)
        return grad_input, grad_weights, None


class GaussPSF(nn.Module):
    def __init__(self, kernel_size, near=1, far=10, pixel_size=5.6e-6, scale=3):
        super(GaussPSF, self).__init__()
        self.kernel_size = kernel_size
        self.near = near
        self.far = far
        self.pixel_size = pixel_size
        self.scale = scale

    def forward(self, image, depth, focal_depth, apature, focal_length):

        Ap = apature.view(-1, 1, 1).expand_as(depth)
        FL = focal_length.view(-1, 1, 1).expand_as(depth)
        focal_depth = focal_depth.view(-1, 1, 1).expand_as(depth)
        Ap = FL / Ap

        real_depth = (self.far - self.near) * depth + self.near
        real_fdepth = (self.far - self.near) * focal_depth + self.near
        c = torch.abs(Ap * (FL * (real_depth - real_fdepth)) / (real_depth * (real_fdepth - FL))) / (self.pixel_size*self.scale)
        c = c.clamp(min=1, max=self.kernel_size)
        weights = c.unsqueeze(1).expand_as(image).contiguous()

        return GaussPSFFunction.apply(image, weights, self.kernel_size)
