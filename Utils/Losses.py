import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, mu=1.5):
    _1D_window = gaussian(window_size, mu).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window


def create_window_avg(window_size, channel):
    _2D_window = torch.ones(window_size, window_size).float().unsqueeze(0).unsqueeze(0) / (window_size ** 2)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel)


class AVERAGE(torch.nn.Module):
    def __init__(self, window_size=7, size_average=False):
        super(AVERAGE, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_avg(window_size, self.channel)

    def forward(self, image):
        mu = F.avg_pool2d(image, 7, 1, self.window_size // 2, count_include_pad=False)
        return mu


class Rec_Loss(nn.Module):
    def __init__(self):
        super(Rec_Loss, self).__init__()
        self.AVG = AVERAGE()
        self.SSIM = SSIM()

    def forward(self, aif_images, focused_images, rec_images, pred_depth):

        alpha = 0.85

        rec_loss = F.l1_loss(rec_images, focused_images)
        ssim_loss = (1 - self.SSIM(rec_images, focused_images)).mean()
        rec_loss = alpha * ssim_loss/2 + (1 - alpha)*rec_loss

        rec_srp = self.sharpness(rec_images).squeeze(1)
        inp_srp = self.sharpness(focused_images).squeeze(1)
        sharpness_loss = F.l1_loss(rec_srp, inp_srp)

        aif_grad = self.gradient(aif_images)
        aif_grad_x_exp = torch.exp(-aif_grad[0].abs())
        aif_grad_y_exp = torch.exp(-aif_grad[1].abs())

        dx, dy = self.gradient(pred_depth.unsqueeze(1))
        dD_x = dx.abs() * aif_grad_x_exp
        dD_y = dy.abs() * aif_grad_y_exp
        sm_loss = (dD_x + dD_y).mean()

        return rec_loss.unsqueeze(0), ssim_loss.unsqueeze(0), sm_loss.unsqueeze(0), sharpness_loss.unsqueeze(0)

    def gradient(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy

    def sharpness(self, image):
        grad = self.gradient(image)
        mu = self.AVG(image) + 1e-8
        output = - (grad[0]**2 + grad[1]**2) - torch.abs((image - mu) / mu) - torch.pow(image - mu, 2)

        return output
