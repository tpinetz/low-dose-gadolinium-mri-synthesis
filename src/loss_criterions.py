import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel2d(ksize, sigma) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        ksize (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(ksize_x, ksize_y)`

    Examples::

        >>> image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(ksize, tuple) or len(ksize) != 2:
        raise TypeError("ksize must be a tuple of length two. Got {}"
                        .format(ksize))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = ksize
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = gaussian(ksize_x, sigma_x)
    kernel_y: torch.Tensor = gaussian(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d


class SSIM(nn.Module):
    r"""Creates a criterion that measures the Structural Similarity (SSIM)
    index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    the loss, or the Structural dissimilarity (DSSIM) can be finally described
    as:

    .. math::

      \text{loss}(x, y) = \frac{1 - \text{SSIM}(x, y)}{2}

    Arguments:
        window_size (int): the size of the kernel.
        max_val (float): the dynamic range of the images. Default: 1.
        reduction (str, optional): Specifies the reduction to apply to the
         output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
         'mean': the sum of the output will be divided by the number of elements
         in the output, 'sum': the output will be summed. Default: 'none'.

    Returns:
        Tensor: the ssim index.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Target :math:`(B, C, H, W)`
        - Output: scale, if reduction is 'none', then :math:`(B, C, H, W)`

    Examples::

        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim = tgm.losses.SSIM(5, reduction='none')
        >>> loss = ssim(input1, input2)  # 1x4x5x5
    """

    def __init__(
            self,
            window_size: int,
            max_val: float = 1.0) -> None:
        super(SSIM, self).__init__()
        self.window_size: int = window_size
        self.max_val: float = max_val

        self.window: torch.Tensor = get_gaussian_kernel2d(
            (window_size, window_size), (1.5, 1.5))
        self.padding: int = self.compute_zero_padding(window_size)

        self.C1: float = (0.01 * self.max_val) ** 2
        self.C2: float = (0.03 * self.max_val) ** 2

    @staticmethod
    def compute_zero_padding(kernel_size: int) -> int:
        """Computes zero padding."""
        return (kernel_size - 1) // 2

    def filter2D(
            self,
            input: torch.Tensor,
            kernel: torch.Tensor,
            channel: int) -> torch.Tensor:
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img1):
            raise TypeError("Input img1 type is not a torch.Tensor. Got {}"
                            .format(type(img1)))
        if not torch.is_tensor(img2):
            raise TypeError("Input img2 type is not a torch.Tensor. Got {}"
                            .format(type(img2)))
        if not len(img1.shape) == 4:
            raise ValueError("Invalid img1 shape, we expect BxCxHxW. Got: {}"
                             .format(img1.shape))
        if not len(img2.shape) == 4:
            raise ValueError("Invalid img2 shape, we expect BxCxHxW. Got: {}"
                             .format(img2.shape))
        if not img1.shape == img2.shape:
            raise ValueError("img1 and img2 shapes must be the same. Got: {}"
                             .format(img1.shape, img2.shape))
        if not img1.device == img2.device:
            raise ValueError("img1 and img2 must be in the same device. Got: {}"
                             .format(img1.device, img2.device))
        if not img1.dtype == img2.dtype:
            raise ValueError("img1 and img2 must be in the same dtype. Got: {}"
                             .format(img1.dtype, img2.dtype))
        # prepare kernel
        b, c, h, w = img1.shape
        tmp_kernel: torch.Tensor = self.window.to(img1.device).to(img1.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # compute local mean per channel
        mu1: torch.Tensor = self.filter2D(img1, kernel, c)
        mu2: torch.Tensor = self.filter2D(img2, kernel, c)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # compute local sigma per channel
        sigma1_sq = self.filter2D(img1 * img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2 * img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1 * img2, kernel, c) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
            ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.

        return torch.mean(loss)


def ssim(
        img1: torch.Tensor,
        img2: torch.Tensor,
        window_size: int,
        max_val: float = 1.0) -> torch.Tensor:
    r"""Function that measures the Structural Similarity (SSIM) index between
    each element in the input `x` and target `y`.
    """
    return SSIM(window_size, max_val)(img1, img2)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.bl = torchvision.models.vgg16(pretrained=True).features[:15].eval()    # Block_3_Conv_3
        for p in self.bl.parameters():
            p.requires_grad = False

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        tf_inp = self.bl(input)
        tf_tg = self.bl(target)

        return ((tf_inp - tf_tg) ** 2).mean()   # Its what they write in their paper.

def PatchWiseWasserStein(img1: torch.Tensor,
                         img2: torch.Tensor,
                         window_size: int,
                         stride: int):
    B, D, H, W, C = img1.shape

    x = img1.unfold(2, window_size, stride)
    x = x.unfold(3, window_size, stride)
    x = x.unfold(4, window_size, stride)
    
    x = x.reshape(B, -1, window_size, window_size, window_size)

    y = img2.unfold(2, window_size, stride)
    y = y.unfold(3, window_size, stride)
    y = y.unfold(4, window_size, stride)

    y = y.reshape(B, -1, window_size, window_size, window_size)

    sigma_1, mu_1 = torch.std_mean(x, (-3, -2, -1), True)
    sigma_2, mu_2 = torch.std_mean(y, (-3, -2, -1), True)

    return torch.mean((mu_1 - mu_2) ** 2 + (sigma_1 - sigma_2) ** 2)


def PatchWiseWasserSteinSinkhorn(img1: torch.Tensor,
                                 img2: torch.Tensor,
                                 window_size: int,
                                 stride: int,
                                 num_iterations: int = 100,
                                 eps: float = 1.,
                                 shift: bool = True):
    if shift:
        roll1 = np.random.choice((window_size // 2) + 1)
        roll2 = np.random.choice((window_size // 2) + 1)
        roll3 = np.random.choice((window_size // 2) + 1)

        img1 = torch.roll(img1, shifts=(roll1, roll2, roll3), dims=(2, 3, 4))
        img2 = torch.roll(img2, shifts=(roll1, roll2, roll3), dims=(2, 3, 4))

    x = img1.unfold(2, window_size, stride)
    x = x.unfold(3, window_size, stride)
    x = x.unfold(4, window_size, stride)

    y = img2.unfold(2, window_size, stride)
    y = y.unfold(3, window_size, stride)
    y = y.unfold(4, window_size, stride)

    x = x.reshape(-1, window_size**3)
    y = y.reshape(-1, window_size**3)

    C_res = torch.abs(x[:, None, :] - y[:, :, None])
    with torch.no_grad():
        Q = C_res.detach()
        Q = torch.exp(-Q / torch.amax(C_res, dim=(1,2), keepdim=True) * 10)
        b = torch.ones((Q.shape[0], window_size ** 3, 1)).type_as(Q)
        T = 1
        for _ in range(num_iterations):
            K = T * Q
            a = 1 / window_size**3 / torch.matmul(K, b)
            b = 1 / window_size**3 / torch.matmul(K.transpose(2,1), a)
            T = a * K * b.transpose(2,1)

    return torch.sum((T * C_res).sum((1,2)))


def get_loss_criterion(type: str):
    if type == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif type == "l2":
        return torch.nn.MSELoss(reduction="none")
    elif type == "huber":
        return torch.nn.SmoothL1Loss(reduction="none", beta=0.1)
    else:
        raise ValueError(f"Loss criterion '{type}' unknown")


def loss_with_gradient_masking(x, y, criterion,
                               grad, g_filter,
                               lambda_all, lambda_small_structures, lambda_smooth_regions,
                               border, mask, mask_atlas=None):
    d, h, w = x.shape[2:]

    loss_small = grad
    loss_smooth = (g_filter(y) > 0.1).to(dtype=y.dtype)

    if mask_atlas is not None:
        loss_small *= mask_atlas
        loss_smooth *= mask_atlas

    # determine metric
    M = mask_atlas + \
        lambda_all * mask + \
        lambda_small_structures * loss_small + \
        lambda_smooth_regions * loss_smooth

    loss = M * criterion(x, y)
    loss = loss[:, :, border:d - border, border:h - border, border:w - border]

    return loss.mean(), loss_small, loss_smooth


def psnr_criterion(x, y, mask=None):
    diff = x-y
    diff = diff.flatten(1)
    y = y.flatten(1)
    if mask is None:
        v_max = torch.max(y, 1)[0]
        N = diff.shape[1]
    else:
        mask = mask.flatten(1)
        diff *= mask
        v_max = torch.max(mask*y, 1)[0]
        N = torch.sum(mask, dim=1)  # account for empty masks

    psnr = 20*torch.log10(v_max / torch.sqrt(torch.sum(diff**2, 1)/N))
    # sort out infinite values
    return torch.mean(psnr[torch.isfinite(psnr)])


def psnr_criterion_np(x, y, mask=None):
    """
        Same behavior as psnr_criterion, just in numpy to evalute batches of volumes
        of the form: NxCxHxW.
        returns:
            psnr: np.float32
    """
    diff = x-y
    diff = diff.reshape((diff.shape[0], -1))
    y = y.reshape((y.shape[0], -1))
    if mask is None:
        v_max = np.max(y, 1)[0]
        N = diff.shape[1]
    else:
        mask = mask.reshape((mask.shape[0], -1))
        diff *= mask
        v_max = np.max(mask*y, axis=1)[0]
        N = np.sum(mask, axis=1)  # account for empty masks

    psnr = 20*np.log10(v_max / np.sqrt(np.sum(diff**2, axis=1)/N))
    # sort out infinite values
    return np.mean(psnr[np.isfinite(psnr)])
