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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
import traceback
from tqdm import tqdm
from utils.image_utils import psnr, masked_psnr, masked_ssim, mask_bbox
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, masks_dir=None):
    """Load all views as CPU tensors. They are moved to the GPU one view at a
    time in evaluate(); loading everything onto the GPU at once OOMs at full
    resolution (46 x 12 MP renders + GTs is ~13 GB)."""
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        if masks_dir is not None and (masks_dir / fname).exists():
            mask = Image.open(masks_dir / fname)
            mask = tf.to_tensor(mask).unsqueeze(0)[:, :1, :, :]
            masks.append((mask > 0.5).float())
        else:
            masks.append(None)
        image_names.append(fname)
    return renders, gts, masks, image_names

@torch.no_grad()
def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "masks"
                if not masks_dir.exists():
                    masks_dir = None
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir)

                ssims = []
                psnrs = []
                lpipss = []
                eval_names = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    if masks[idx] is not None:
                        mask = masks[idx]
                        bbox = mask_bbox(mask)
                        if bbox is None:
                            print("  WARNING: empty mask for {}, skipping view".format(image_names[idx]))
                            continue
                        y0, y1, x0, x1 = bbox
                        # Identical input for every metric: the masked image (background
                        # set to black) cropped to the padded bounding box of the mask.
                        # Only the crop is moved to the GPU.
                        render_c = ((renders[idx] * mask)[..., y0:y1, x0:x1]).cuda()
                        gt_c = ((gts[idx] * mask)[..., y0:y1, x0:x1]).cuda()
                        mask_c = (mask[..., y0:y1, x0:x1]).cuda()
                        # SSIM: mean over crop pixels whose full 11x11 window lies inside the mask.
                        ssims.append(masked_ssim(render_c, gt_c, mask_c).cpu())
                        # PSNR: MSE over crop pixels inside the mask.
                        psnrs.append(masked_psnr(render_c.squeeze(0), gt_c.squeeze(0), mask_c.squeeze(0)).cpu())
                        # LPIPS: over the whole crop (no per-pixel form exists).
                        lpipss.append(lpips(render_c, gt_c, net_type='vgg').cpu())
                    else:
                        render = renders[idx].cuda()
                        gt = gts[idx].cuda()
                        ssims.append(ssim(render, gt).cpu())
                        psnrs.append(psnr(render, gt).cpu())
                        lpipss.append(lpips(render, gt, net_type='vgg').cpu())
                    eval_names.append(image_names[idx])

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), eval_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), eval_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), eval_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception:
            print("Unable to compute metrics for model", scene_dir)
            traceback.print_exc()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--device', type=int, default=0,
                        help="CUDA device index (lab-fork style)")
    args = parser.parse_args()
    torch.cuda.set_device(torch.device(f"cuda:{args.device}"))
    evaluate(args.model_paths)
