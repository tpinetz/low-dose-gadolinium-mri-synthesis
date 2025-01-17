import torch

from networks import Resolution, Model, AvgPool2x2x2, MaxPool2x2x2, Downsample2x2x2, Interpolate2x2x2


def nonlinearity(inplace=False):
    return torch.nn.LeakyReLU(0.1, inplace=inplace)

def normalization(norm_fn: str, channels: int, num_groups: int = 1):
    if norm_fn == 'group':
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    elif norm_fn == 'batch':
        return torch.nn.BatchNorm3d(channels)

    elif norm_fn == 'instance':
        return torch.nn.InstanceNorm3d(channels)

    elif norm_fn == 'none':
        return torch.nn.Identity()


class Conv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 bias: bool = True, fast: bool = False):
        super().__init__()

        self.c = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                                 padding=kernel_size//2,
                                 padding_mode='zeros' if fast or (kernel_size == 1) else 'replicate',
                                 bias=bias)

    def forward(self, x):
        return self.c(x)


class FastShortcut(torch.nn.Module):
    """ Implementation of a 1x1x1 convolution using a Linear layer.
        Required due to missing CPU implementation of pytorch!
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # convert to channels last
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        # convert back to channels first
        return x.permute(0, 4, 1, 2, 3)


class ResnetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, cemb_channels: int = 512,
                 pre_conv: bool = False, post_conv: bool = False, norm_fn='none', use_embedding=True,
                 fast: bool = False):
        super().__init__()

        self.use_embedding = use_embedding

        if pre_conv:
            self.pre_conv = Conv(in_channels, out_channels, fast=fast)
            in_channels = out_channels
        else:
            self.pre_conv = torch.nn.Identity()

        if post_conv:
            self.post_conv = Conv(in_channels, out_channels, fast=fast)
            out_channels = in_channels
        else:
            self.post_conv = torch.nn.Identity()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = normalization(norm_fn, in_channels)
        self.act1 = nonlinearity(inplace=False)
        self.conv1 = Conv(in_channels,
                          out_channels,
                          fast=fast)
        self.actcemb = nonlinearity(inplace=False)
        self.temb_proj = torch.nn.Linear(cemb_channels,
                                         out_channels)
        self.norm2 = normalization(norm_fn, out_channels)
        self.act2 = nonlinearity(inplace=True)
        self.conv2 = Conv(out_channels,
                          out_channels,
                          fast=fast)
        if self.in_channels != self.out_channels:
            self.in_shortcut = FastShortcut(in_channels,
                                            out_channels)
        else:
            self.in_shortcut = torch.nn.Identity()

    def forward(self, x, cemb):
        x = self.pre_conv(x)
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        if self.use_embedding:
            h = h + self.temb_proj(self.actcemb(cemb))[:, :, None, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        x = self.in_shortcut(x)

        return self.post_conv(x+h)

class IdentityWithSecondArgument(torch.nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(IdentityWithSecondArgument, self).__init__()

    def forward(self, input, input2):
        return input


class LocalAttentionBlock(torch.nn.Module):
    def __init__(self, dim: int, heads: int = 1, size: int = 3):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0
        self.size = size
        self.ws = size*size*size

        self.scale = (self.dim//self.heads) ** (-.5)

        self.q = torch.nn.Linear(self.dim, self.dim)
        self.k = torch.nn.Linear(self.dim, self.dim)
        self.v = torch.nn.Linear(self.dim, self.dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _unfold(self, x):
        B, D, H, W, C = x.shape
        # pad with zeros
        pad = int(self.size//2)
        xp = x.new_zeros((B, D+2*pad, H+2*pad, W+2*pad, C))
        xp[:, pad:-pad, pad:-pad, pad:-pad, :] = x
        x = xp.unfold(1, self.size, 1)
        x = x.unfold(2, self.size, 1)
        x = x.unfold(3, self.size, 1)  # [B, D, H, W, C, S, S, S]
        x = x.permute(0, 1, 2, 3, 5, 6, 7, 4)  # [B, D, H, W, S, S, S, C]
        x = x.reshape(B, D*H*W, self.ws, self.heads, C//self.heads,).transpose(2, 3)  # [B, DHW, Nh, S**3, C/Nh]
        return x

    def forward(self, x, cemb):
        # transform to channels last
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        B, D, H, W, C = x.shape
        assert C == self.dim

        # compute query, key and value
        q = self.q(x).view(B, D*H*W, self.heads, 1, C//self.heads)  # [B, DHW, Nh, 1, C/Nh]
        k = self._unfold(self.k(x))  # [B, DHW, Nh, S**3, C/Nh]
        v = self._unfold(self.v(x))  # [B, DHW, Nh, S**3, C/Nh]
        # compute local attention
        q = q * self.scale
        attn = torch.einsum("...ij,...kj->...ik", q, k)  # q @ k.T: [B, DHW, Nh, 1, S**3]
        # TODO: maybe add relative position
        attn = self.softmax(attn)
        # compute output
        h = (attn @ v).view(B, D, H, W, C)  # [B, D, H, W, C]
        return (x + h).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]


class Discriminator(Model):

    def __init__(self, config, fast: bool = False):
        super().__init__(config)

        # conditional embedding
        in_cond_channels = self.config["in_cond_channels"]
        cond_channels = self.config["cond_channels"]
        cond_layers = self.config["cond_layers"]
        cond_linear = [torch.nn.Linear(cond_channels, cond_channels) for _ in range(cond_layers)]
        cond_nonlinear = [nonlinearity(inplace=True) for _ in range(cond_layers)]
        self.cond_emb = torch.nn.Sequential(
            # start with first linear layer
            torch.nn.Linear(in_cond_channels, cond_channels),
            # alternate a linear and nonlinear layer
            *[x for y in zip(cond_nonlinear, cond_linear) for x in y]
        )

        # network architecture
        in_channels = self.config["in_channels"]
        channels = self.config["channels"]
        out_channels = self.config["out_channels"]
        self.fast = fast
        pool_type = self.config.get("pool_type", "avg")
        norm_fn = self.config.get("norm_fn", "none")

        attn_enable = self.config.get("attn_enable", False)
        attn_heads = self.config.get("attn_heads", 4)
        attn_size = self.config.get("attn_size", 3)

        # feature extraction and head for each resolution
        self.extractor = Conv(in_channels, channels, fast=self.fast)

        # prediction network for the coarsest resolution
        self.down = Downsample2x2x2(fast=self.fast)
        if pool_type == "max":
            self.pool = MaxPool2x2x2()
        elif pool_type == "avg":
            self.pool = AvgPool2x2x2()
        else:
            raise RuntimeError(f"unsupported pool_type '{pool_type}'")


        self.b11 = torch.nn.ModuleList([
                ResnetBlock(channels, 2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                ResnetBlock(2*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                ])

        self.b21 = torch.nn.ModuleList([
                ResnetBlock(2*channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                LocalAttentionBlock(4*channels, heads=attn_heads, size=attn_size) if attn_enable
                else IdentityWithSecondArgument(),
                ResnetBlock(4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                ])


        self.b41 = torch.nn.ModuleList([
                ResnetBlock(4*channels, 8*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                LocalAttentionBlock(8*channels, heads=attn_heads, size=attn_size) if attn_enable
                else IdentityWithSecondArgument(),
                ResnetBlock(8 * channels, 4*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                ])

        self.b81 = torch.nn.ModuleList([
                ResnetBlock(4*channels, 8*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                LocalAttentionBlock(8*channels, heads=attn_heads, size=attn_size) if attn_enable
                else IdentityWithSecondArgument(),
                ResnetBlock(8*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                ])

        self.b161 = torch.nn.ModuleList([
                ResnetBlock(8*channels, 16*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                LocalAttentionBlock(16*channels, heads=attn_heads, size=attn_size) if attn_enable
                else IdentityWithSecondArgument(),
                ResnetBlock(16*channels, cemb_channels=cond_channels, norm_fn=norm_fn, fast=self.fast),
                ])

        self.final_1x1_conv = Conv(16 * channels, 1, 1)

    def forward(self, x, condition):
        inputs = x
        outputs = {}

        cemb = self.cond_emb(condition)

        # compute features on the coarsest scale
        x1 = self.extractor(inputs)
        for b in self.b11:
            x1 = b(x1, cemb)
        x2 = self.pool(x1)
        for b in self.b21:
            x2 = b(x2, cemb)
        x4 = self.pool(x2)
        for b in self.b41:
            x4 = b(x4, cemb)
        x8 = self.pool(x4)
        for b in self.b81:
            x8 = b(x8, cemb)
        x16 = self.pool(x8)
        for b in self.b161:
            x16 = b(x16, cemb)
        
        x16 = self.final_1x1_conv(x16)
        outputs = torch.mean(x16, axis=(1, 2, 3, 4))

        return outputs

    def extra_repr(self):
        return f"resolutions={self.resolutions} fast={self.fast}"
