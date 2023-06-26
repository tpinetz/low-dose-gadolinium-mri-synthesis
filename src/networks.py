import torch
import numpy as np
import abc
from enum import Enum


class Resolution(Enum):
    HALF = "halfmm"
    ONE = "onemm"


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError


class Downsample2x2x2(torch.nn.Module):
    def __init__(self, fast=False):
        super().__init__()

        self.fast = fast

        # create the convolution kernel
        np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
        np_k2 = np_k @ np_k.T
        np_k = (np_k @ np_k2.reshape(1, -1)).reshape(5, 5, 5)
        np_k /= np_k.sum()
        np_k = np.reshape(np_k, (1, 1, 5, 5, 5))
        self.register_buffer('blur', torch.from_numpy(np_k))

    def forward(self, x):
        kernel = self.blur
        pad = int(kernel.shape[-1])//2
        N, C, D, H, W = x.shape
        assert x.stride()[-1] == 1
        if self.fast:
            x = torch.nn.functional.conv3d(x.view(N*C, 1, D, H, W), kernel, stride=2, padding=pad)
        else:
            x = torch.nn.functional.pad(x.view(N*C, 1, D, H, W), (pad, pad, pad, pad, pad, pad), 'replicate')
            # compute the convolution
            x = torch.nn.functional.conv3d(x, kernel, stride=2)
        return x.view(N, C, (D+1)//2, (H+1)//2, (W+1)//2)


class AvgPool2x2x2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.avg_pool3d(x, kernel_size=3, stride=2, padding=1)


class MaxPool2x2x2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.max_pool3d(x, kernel_size=3, stride=2, padding=1)


class Interpolate2x2x2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, output_shape=None):
        return torch.nn.functional.interpolate(x, size=output_shape[2:], mode="trilinear", align_corners=False)


class Upsample2x2x2(torch.nn.Module):
    def __init__(self, fast=False):
        super().__init__()

        self.fast = fast

        # create the convolution kernel
        np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
        np_k2 = np_k @ np_k.T
        np_k = (np_k @ np_k2.reshape(1, -1)).reshape(5, 5, 5)
        np_k /= np_k.sum()
        np_k *= 8
        np_k = np.reshape(np_k, (1, 1, 5, 5, 5))
        self.register_buffer('blur', torch.from_numpy(np_k))

    def forward(self, x, output_shape=None):
        # determine the amount of padding
        if output_shape is not None:
            output_padding = (
                output_shape[2] - ((x.shape[2]-1)*2+1),
                output_shape[3] - ((x.shape[3]-1)*2+1),
                output_shape[4] - ((x.shape[4]-1)*2+1)
            )
        else:
            output_padding = 0

        kernel = self.blur
        pad = int(kernel.shape[-1])//4

        N, C, D, H, W = x.shape
        assert x.stride()[-1] == 1
        if self.fast:
            x = torch.nn.functional.conv_transpose3d(x.view(N*C, 1, D, H, W), kernel, stride=2, padding=2*pad,
                                                     output_padding=output_padding)
        else:
            x = torch.nn.functional.pad(x.view(N*C, 1, D, H, W), (pad, pad, pad, pad, pad, pad), 'replicate')
            # compute the convolution
            x = torch.nn.functional.conv_transpose3d(x, kernel, stride=2, padding=4*pad, output_padding=output_padding)
        return x.view(N, C, *output_shape[2:])


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ksize, bias=True, exclude_final_act=False, fast=False):
        super().__init__()

        self.c = torch.nn.Conv3d(in_channels, out_channels, ksize, padding=ksize//2,
                                 padding_mode='zeros' if fast else 'replicate',
                                 bias=bias)
        self.a = torch.nn.ReLU(inplace=True) if not exclude_final_act else torch.nn.Identity()

    def forward(self, x):
        return self.a(self.c(x))


class ConvDepthwiseSeperable(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ksize, bias=True, exclude_final_act=False, fast=False):
        super().__init__()

        self.c1 = torch.nn.Conv3d(in_channels, in_channels, ksize, padding=ksize//2,
                                  padding_mode='zeros' if fast else 'replicate',
                                  bias=bias, groups=in_channels)
        self.c2 = torch.nn.Conv3d(in_channels, out_channels, 1, padding=0,
                                  bias=bias, groups=1)
        self.a = torch.nn.ReLU(inplace=True) if not exclude_final_act else torch.nn.Identity()

    def forward(self, x):
        return self.a(self.c2(self.c1(x)))


class Block(torch.nn.Module):
    def __init__(self, in_channels, channels, out_channels, ksize, num_layers=2, bias=True,
                 exclude_final_act=False, depthwise_separable=False, fast=False):
        super().__init__()

        def layer(*args, **kwargs):
            if depthwise_separable:
                return ConvDepthwiseSeperable(*args, **kwargs)
            else:
                return Conv(*args, **kwargs)

        if num_layers == 1:
            self.b = layer(in_channels, out_channels, ksize, bias, exclude_final_act, fast)
        elif num_layers == 2:
            self.b = torch.nn.Sequential(layer(in_channels, channels, ksize, bias, exclude_final_act=False, fast=fast),
                                         layer(channels, out_channels, ksize, bias, exclude_final_act, fast))
        else:
            self.b = torch.nn.Sequential(
                layer(in_channels, channels, ksize, bias, exclude_final_act=False, fast=fast),
                *[layer(in_channels, channels, ksize, bias, exclude_final_act=False, fast=fast)
                  for _ in range(num_layers-2)],
                layer(channels, out_channels, ksize, bias, exclude_final_act, fast))

    def forward(self, x):
        return self.b(x)


class MultiScaleModel(Model):

    def __init__(self, config):
        super().__init__(config)

        num_in_channels = self.config["num_in_channels"]
        num_channels = self.config["num_channels"]
        num_out_channels = self.config["num_out_channels"]
        ksize = self.config["ksize"]
        depth_seperable = self.config["depthwise_separable"]
        self.fast = config["fast"] if "fast" in config else False

        self.resolutions = [0.5, 1.0]

        # feature extraction and head for each resolution
        self.extractors = torch.nn.ModuleList([
            Block(num_in_channels, num_channels, num_channels, ksize, num_layers=1, fast=self.fast),
            Block(num_in_channels, 2*num_channels, 2*num_channels, ksize, num_layers=1, fast=self.fast),
            ])
        self.mixings = torch.nn.ModuleList([
            Block(2*num_channels, num_channels, num_channels, ksize,
                  depthwise_separable=depth_seperable, fast=self.fast),
            Block(4*num_channels, 2*num_channels, num_channels, ksize,
                  depthwise_separable=depth_seperable, fast=self.fast),
            ])
        self.heads = torch.nn.ModuleList([
            Block(num_channels, num_channels, num_out_channels, ksize, num_layers=1, fast=self.fast,
                  exclude_final_act=True),
            Block(num_channels, num_channels, num_out_channels, ksize, num_layers=1, fast=self.fast,
                  exclude_final_act=True),
            ])

        # prediction network for the coarsest resolution
        self.bin = Block(2*num_channels, 2*num_channels, 2*num_channels, ksize,
                         depthwise_separable=depth_seperable, fast=self.fast)
        self.down = AvgPool2x2x2()
        self.b21 = Block(2*num_channels, 4*num_channels, 4*num_channels, ksize,
                         depthwise_separable=depth_seperable, fast=self.fast)
        self.b4 = Block(4*num_channels, 4*num_channels, 4*num_channels, ksize,
                        depthwise_separable=depth_seperable, fast=self.fast)
        self.b22 = Block(8*num_channels, 4*num_channels, 2*num_channels, ksize,
                         depthwise_separable=depth_seperable, fast=self.fast)
        self.up = Interpolate2x2x2()

    def preprocess_input(self, x, resolution):
        assert resolution in self.resolutions

        if resolution < 0.75:
            return {Resolution.HALF.value: x, Resolution.ONE.value: self.down(x)}
        else:
            return {Resolution.ONE.value: x}

    def forward(self, x, resolution):

        inputs = self.preprocess_input(x, resolution)
        outputs = {}

        # compute features on the coarsest scale
        x1 = self.extractors[1](inputs[Resolution.ONE.value])
        x1 = self.bin(x1)
        x2 = self.down(x1)
        x2 = self.b21(x2)
        x4 = self.down(x2)
        x4 = self.b4(x4)
        x4 = self.up(x4, output_shape=x2.shape)
        x2 = torch.cat([x2, x4], 1)
        if self.fast:
            del x4
        x2 = self.b22(x2)
        x2 = self.up(x2, output_shape=x1.shape)
        x1 = torch.cat([x1, x2], 1)
        if self.fast:
            del x2
        x1 = self.mixings[1](x1)
        outputs[Resolution.ONE.value] = inputs[Resolution.ONE.value][:, :1] + self.heads[1](x1)

        if Resolution.HALF.value in inputs:
            x0 = self.extractors[0](inputs[Resolution.HALF.value])
            x1 = self.up(x1, output_shape=x0.shape)
            x0 = torch.cat([x0, x1], 1)
            if self.fast:
                del x1
            x0 = self.mixings[0](x0)
            outputs[Resolution.HALF.value] = inputs[Resolution.HALF.value][:, :1] + self.heads[0](x0)

        return outputs

    def extra_repr(self):
        return f"resolutions={self.resolutions} fast={self.fast}"


class RecurrentMultiScaleModel(Model):
    def __init__(self, config):
        super().__init__(config)

        num_in_channels = self.config["num_in_channels"]
        num_channels = self.config["num_channels"]
        num_out_channels = self.config["num_out_channels"]
        ksize = self.config["ksize"]
        self.fast = config["fast"] if "fast" in config else False

        self.resolutions = [0.5, 1, 2]

        # feature extraction and head for each resolution
        self.extractor = Block(num_in_channels, num_channels, num_channels, ksize, fast=self.fast)
        self.mixing = Block(2*num_channels, num_channels, num_channels, ksize, fast=self.fast)
        self.head = Block(num_channels, num_channels, num_out_channels, ksize, fast=self.fast,
                          exclude_final_act=True)

        # prediction network for the coarsest resolution
        self.bin = Block(num_channels, 4*num_channels, 4*num_channels, ksize, fast=self.fast)
        self.down = Downsample2x2x2(fast=self.fast)
        self.b4 = Block(4*num_channels, 4*num_channels, 4*num_channels, ksize, fast=self.fast)
        self.bout = Block(4*num_channels, 4*num_channels, num_channels, ksize, fast=self.fast)
        self.up = Upsample2x2x2(fast=self.fast)

    def preprocess_input(self, x, resolution):
        assert resolution in self.resolutions

        inputs = {resolution: x}
        while resolution < 2:
            resolution = resolution * 2
            x = torch.nn.functional.avg_pool3d(x, 2)
            inputs[resolution] = x
        return inputs

    def forward(self, x, resolution):

        inputs = self.preprocess_input(x, resolution)
        outputs = {}

        # compute features on the coarsest scale
        x2 = self.extractor(inputs[2])
        x4 = self.down(self.bin(x2))
        x4 = self.b4(x4)
        x4 = self.up(x4, output_shape=x2.shape)
        x4 = self.bout(x4)
        x2 = torch.cat([x2, x4], 1)
        if self.fast:
            del x4
        x2 = self.mixing(x2)

        outputs[2] = inputs[2][:, :1] + self.head(x2)

        if 1 in inputs:
            x1 = self.extractor(inputs[1])
            x2 = self.up(x2, output_shape=x1.shape)
            x1 = torch.cat([x1, x2], 1)
            if self.fast:
                del x2
            x1 = self.mixing(x1)
            outputs[1] = inputs[1][:, :1] + self.head(x1)

        if 0.5 in inputs:
            x0 = self.extractor(inputs[0.5])
            x1 = self.up(x1, output_shape=x0.shape)
            x0 = torch.cat([x0, x1], 1)
            if self.fast:
                del x1
            x0 = self.mixing(x0)
            outputs[0.5] = inputs[0.5][:, :1] + self.head(x0)

        return outputs

    def extra_repr(self):
        return f"resolutions={self.resolutions} fast={self.fast}"


class Block2D(torch.nn.Module):
    def __init__(self, in_channels, channels, out_channels, ksize, bias=True, exclude_final_act=False,
                 fast=False):
        super().__init__()

        self.cin = torch.nn.Conv2d(in_channels, channels, ksize, padding=ksize//2,
                                   padding_mode='zeros' if fast else 'replicate',
                                   bias=bias)
        self.ain = torch.nn.ReLU(inplace=fast)
        self.cmid = torch.nn.Conv2d(channels, channels, ksize, padding=ksize//2,
                                    padding_mode='zeros' if fast else 'replicate',
                                    bias=bias)
        self.amid = torch.nn.ReLU(inplace=fast) if not exclude_final_act else torch.nn.Identity()
        self.cout = torch.nn.Conv2d(channels, out_channels, ksize, padding=ksize//2,
                                    padding_mode='zeros' if fast else 'replicate',
                                    bias=bias)
        self.aout = torch.nn.ReLU(inplace=fast) if not exclude_final_act else torch.nn.Identity()

    def forward(self, x):
        x = self.aout(self.cout(self.amid(self.cmid(self.ain(self.cin(x))))))
        # torch.cuda.empty_cache()
        return x


class Scale2DModel(Model):
    def __init__(self, config):
        super().__init__(config)

        num_in_channels = self.config["num_in_channels"]
        num_channels = self.config["num_channels"]
        num_out_channels = self.config["num_out_channels"]
        ksize = self.config["ksize"]

        self.fast = config["fast"] if "fast" in config else False

        self.bin = Block2D(num_in_channels, num_channels, num_channels, ksize, fast=self.fast)

        self.down1 = torch.nn.MaxPool2d(2)

        self.b11 = Block2D(num_channels, 2*num_channels, 2*num_channels, ksize, fast=self.fast)

        self.down2 = torch.nn.MaxPool2d(2)

        self.b21 = Block2D(2*num_channels, 4*num_channels, 4*num_channels, ksize, fast=self.fast)

        self.down3 = torch.nn.MaxPool2d(2)

        self.b31 = torch.nn.Conv2d(4*num_channels, 4*num_channels, ksize, padding=ksize//2,
                                   padding_mode='zeros' if self.fast else 'replicate', bias=True)
        self.b31_relu = torch.nn.ReLU(inplace=self.fast)

        self.up3 = torch.nn.Upsample(size=128)  

        self.b22 = Block2D(8*num_channels, 4*num_channels, 2*num_channels, ksize, fast=self.fast)

        self.up2 = torch.nn.Upsample(size=256)  

        self.b12 = Block2D(4*num_channels, 2*num_channels, num_channels, ksize, fast=self.fast)

        self.up1 = torch.nn.Upsample(512)  

        self.bout = Block2D(2 * num_channels, num_channels, num_out_channels, ksize,
                            exclude_final_act=True, fast=self.fast)

    def forward(self, x):
        x0 = self.bin(x)

        x1 = self.down1(x0)

        x1 = self.b11(x1)

        x2 = self.down2(x1)

        x2 = self.b21(x2)

        x3 = self.down3(x2)
        x3 = self.b31_relu(self.b31(x3))

        x2 = torch.cat([x2, self.up3(x3)], dim=1)
        x2 = self.b22(x2)

        x1 = torch.cat([x1, self.up2(x2)], dim=1)
        x1 = self.b12(x1)

        x0 = torch.cat([x0, self.up1(x1)], dim=1)

        return self.bout(x0)

    def extra_repr(self):
        return f"fast={self.fast}"