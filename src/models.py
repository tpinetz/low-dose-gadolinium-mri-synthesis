import torch
import numpy as np
import abc


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError


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
        self.amid = torch.nn.ReLU(inplace=fast)
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

        self.up3 = torch.nn.Upsample(size=64)  # This is fixed in Gong et al.

        self.b22 = Block2D(8*num_channels, 4*num_channels, 2*num_channels, ksize, fast=self.fast)

        self.up2 = torch.nn.Upsample(size=128)  # This is fixed in Gong et al.

        self.b12 = Block2D(4*num_channels, 2*num_channels, num_channels, ksize, fast=self.fast)

        self.up1 = torch.nn.Upsample(256)  # This is fixed in Gong et al.

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

        x2 = torch.concat([x2, self.up3(x3)], dim=1)
        x2 = self.b22(x2)

        x1 = torch.concat([x1, self.up2(x2)], dim=1)
        x1 = self.b12(x1)

        x0 = torch.concat([x0, self.up1(x1)], dim=1)

        return self.bout(x0)

    def extra_repr(self):
        return f"fast={self.fast}"
