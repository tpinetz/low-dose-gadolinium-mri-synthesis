import torch
import numpy as np


def get_gaussian_filter_3d(scale, ksize=9):
    x_cord = torch.arange(ksize)
    y_cord = torch.arange(ksize)
    z_cord = torch.arange(ksize)

    grid_x, grid_y, grid_z = torch.meshgrid([x_cord, y_cord, z_cord])
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    mean = (ksize - 1) / 2
    variance = scale ** 2

    gaussian_kernel = (1. / (2. * np.pi * variance) ** (3 / 2)) * torch.exp( -torch.sum((grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel.view(1, 1, ksize, ksize, ksize)
    gaussian_kernel /= gaussian_kernel.sum()

    gaussian_filter = torch.nn.Conv3d(1, 1, kernel_size=ksize, groups=1, bias=False, padding=ksize // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter
