#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from utils.loss_utils import ssim

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def masked_psnr(img1, img2, mask):
    """Compute PSNR only over masked pixels (mask > 0.5)."""
    mask_bool = (mask > 0.5).expand_as(img1)
    if not mask_bool.any():
        return torch.tensor(0.0, device=img1.device)
    diff_sq = (img1 - img2) ** 2
    masked_mse = diff_sq[mask_bool].mean()
    return 20 * torch.log10(1.0 / torch.sqrt(masked_mse))

def mask_bbox(mask, pad=8):
    """Tight bounding box of the mask (pixels with mask > 0.5), padded by `pad`
    pixels and clamped to the image bounds.

    Args:
        mask: tensor of shape [1, 1, H, W].
        pad: padding in pixels added on every side of the tight box.
    Returns:
        (y0, y1, x0, x1) as Python ints, usable as img[..., y0:y1, x0:x1].
        Returns None if the mask is empty.
    """
    H, W = mask.shape[-2], mask.shape[-1]
    ys, xs = torch.where(mask[0, 0] > 0.5)
    if ys.numel() == 0:
        return None
    y0 = max(int(ys.min().item()) - pad, 0)
    y1 = min(int(ys.max().item()) + 1 + pad, H)
    x0 = max(int(xs.min().item()) - pad, 0)
    x1 = min(int(xs.max().item()) + 1 + pad, W)
    return y0, y1, x0, x1

def masked_ssim(img1, img2, mask, window_size=11):
    """SSIM averaged only over pixels whose entire window_size x window_size
    window lies inside the mask.

    The mask is binarized (mask > 0.5) and eroded by window_size // 2 pixels
    (min-pooling), so background pixels and object-border pixels whose SSIM
    window overlaps the background are excluded from the average. The mean is
    taken over all channels of the surviving pixels.

    Args:
        img1, img2: tensors of shape [1, C, H, W].
        mask: tensor of shape [1, 1, H, W].
    Returns:
        scalar tensor; 0.0 if no pixel survives the erosion.
    """
    ssim_map = ssim(img1, img2, window_size=window_size, size_average=False)
    mask_bin = (mask > 0.5).float()
    eroded = -F.max_pool2d(-mask_bin, window_size, stride=1, padding=window_size // 2)
    valid = (eroded > 0.5).expand_as(ssim_map)
    if not valid.any():
        return torch.tensor(0.0, device=img1.device)
    return ssim_map[valid].mean()
