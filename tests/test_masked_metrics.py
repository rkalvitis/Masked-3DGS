"""Sanity checks for the masked evaluation metrics used by metrics.py.

Builds a synthetic 256x256 image pair where a ~40x40 object sits on a black
background. Noise is added *only inside the object*, so a correct masked metric
must report a clearly worse score than a full-image metric that is diluted by
the large, perfectly matching black background.

Run with:  pytest -s tests/test_masked_metrics.py
or:        python tests/test_masked_metrics.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.loss_utils import ssim  # noqa: E402
from utils.image_utils import masked_psnr, masked_ssim, mask_bbox  # noqa: E402


H = W = 256
Y0, Y1, X0, X1 = 100, 140, 108, 148  # 40x40 object


def make_pair(seed=0, noise_std=0.25):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, 1, H, W)
    mask[..., Y0:Y1, X0:X1] = 1.0
    gt = torch.zeros(1, 3, H, W)
    gt[..., Y0:Y1, X0:X1] = torch.rand(1, 3, Y1 - Y0, X1 - X0, generator=g)
    noise = torch.randn(1, 3, H, W, generator=g) * noise_std
    render = (gt + noise * mask).clamp(0.0, 1.0)
    # Emulate metrics.py: images multiplied by the mask, black background.
    return render * mask, gt * mask, mask


def test_mask_bbox():
    _, _, mask = make_pair()
    assert mask_bbox(mask, pad=0) == (Y0, Y1, X0, X1)
    assert mask_bbox(mask, pad=8) == (Y0 - 8, Y1 + 8, X0 - 8, X1 + 8)
    assert mask_bbox(mask, pad=1000) == (0, H, 0, W)
    assert mask_bbox(torch.zeros(1, 1, H, W)) is None


def test_masked_ssim_lower_than_full_image_ssim():
    render, gt, mask = make_pair()
    full = ssim(render, gt).item()
    masked = masked_ssim(render, gt, mask).item()
    print(f"\nSSIM  full-image: {full:.4f}   masked (eroded window): {masked:.4f}")
    # Full-image SSIM is dominated by ~97% black-on-black pixels scoring ~1.
    assert full > 0.9
    assert masked < full - 0.1, "masked SSIM should be noticeably lower than full-image SSIM"


def test_masked_ssim_empty_mask_is_guarded():
    render, gt, _ = make_pair()
    assert masked_ssim(render, gt, torch.zeros(1, 1, H, W)).item() == 0.0
    # A mask thinner than the window is fully eroded away.
    thin = torch.zeros(1, 1, H, W)
    thin[..., 120:125, 120:125] = 1.0
    assert masked_ssim(render, gt, thin).item() == 0.0


def test_masked_ssim_matches_map_mean_on_interior():
    render, gt, mask = make_pair()
    ssim_map = ssim(render, gt, size_average=False)
    assert ssim_map.shape == (1, 3, H, W)
    interior = torch.zeros(1, 1, H, W, dtype=torch.bool)
    interior[..., Y0 + 5:Y1 - 5, X0 + 5:X1 - 5] = True
    expected = ssim_map[interior.expand_as(ssim_map)].mean()
    assert torch.allclose(masked_ssim(render, gt, mask), expected)


def test_cropped_lpips_higher_than_full_image_lpips():
    pytest.importorskip("torchvision")
    from lpipsPyTorch import lpips

    render, gt, mask = make_pair()
    y0, y1, x0, x1 = mask_bbox(mask)
    with torch.no_grad():
        full = lpips(render, gt, net_type="vgg").item()
        cropped = lpips(render[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1], net_type="vgg").item()
    print(f"\nLPIPS full-image: {full:.4f}   cropped ({y1 - y0}x{x1 - x0} bbox): {cropped:.4f}")
    assert cropped > full * 3, "cropped LPIPS should be noticeably higher than full-image LPIPS"


def test_masked_metrics_are_crop_invariant():
    """metrics.py feeds every metric the same bbox crop; PSNR and masked SSIM
    must give the same value on the crop as on the full image."""
    render, gt, mask = make_pair()
    y0, y1, x0, x1 = mask_bbox(mask)
    r_c, g_c, m_c = render[..., y0:y1, x0:x1], gt[..., y0:y1, x0:x1], mask[..., y0:y1, x0:x1]
    assert torch.allclose(masked_ssim(r_c, g_c, m_c), masked_ssim(render, gt, mask), atol=1e-6)
    assert torch.allclose(masked_psnr(r_c.squeeze(0), g_c.squeeze(0), m_c.squeeze(0)),
                          masked_psnr(render.squeeze(0), gt.squeeze(0), mask.squeeze(0)), atol=1e-4)


def test_masked_psnr_matches_manual():
    render, gt, mask = make_pair()
    got = masked_psnr(render.squeeze(0), gt.squeeze(0), mask.squeeze(0))
    obj_r = render[..., Y0:Y1, X0:X1]
    obj_g = gt[..., Y0:Y1, X0:X1]
    manual = 20 * torch.log10(1.0 / torch.sqrt(((obj_r - obj_g) ** 2).mean()))
    full = 20 * torch.log10(1.0 / torch.sqrt(((render - gt) ** 2).mean()))
    print(f"\nPSNR  full-image: {full.item():.4f}   masked: {got.item():.4f}   manual object-only: {manual.item():.4f}")
    assert torch.allclose(got, manual, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "-v", __file__]))
