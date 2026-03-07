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
