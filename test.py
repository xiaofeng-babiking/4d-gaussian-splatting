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

import random
import sys
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from utils.loss_utils import l1_loss, ssim, msssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams


def evaluate(
    dataset,
    pipe,
    checkpoint,
    load_iteration,
    gaussian_dim,
    time_duration,
    num_pts,
    num_pts_ratio,
    rot_4d,
    force_sh_3d,
):
    if dataset.frame_ratio > 1:
        time_duration = [
            time_duration[0] / dataset.frame_ratio,
            time_duration[1] / dataset.frame_ratio,
        ]

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=2 if pipe.eval_shfs_4d else 0,
        prefilter_var=dataset.prefilter_var,
    )

    if checkpoint:
        # Checkpoint route: skip the PLY auto-load and replay weights from a .pth.
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=None,
            shuffle=False,
            num_pts=num_pts,
            num_pts_ratio=num_pts_ratio,
            time_duration=time_duration,
        )
        model_params, ckpt_iter = torch.load(checkpoint)
        gaussians.restore(model_params, training_args=None)
        print("Loaded checkpoint from iteration {}: {}".format(ckpt_iter, checkpoint))
    else:
        # Standard route: Scene loads point_cloud/iteration_<N>/point_cloud.ply.
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=load_iteration,
            shuffle=False,
            num_pts=num_pts,
            num_pts_ratio=num_pts_ratio,
            time_duration=time_duration,
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Train-view sanity slice (5 samples) + full test set.
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    validation_configs = (
        {
            "name": "train",
            "cameras": (
                [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)]
                if len(train_cams) > 0
                else []
            ),
        },
        {"name": "test", "cameras": [test_cams[idx] for idx in range(len(test_cams))]},
    )

    with torch.no_grad():
        for config in validation_configs:
            if not config["cameras"]:
                continue
            l1_acc = 0.0
            psnr_acc = 0.0
            ssim_acc = 0.0
            msssim_acc = 0.0
            for batch_data in tqdm(
                config["cameras"], desc="Evaluating {}".format(config["name"])
            ):
                gt_image, viewpoint = batch_data
                gt_image = gt_image.cuda()
                viewpoint = viewpoint.cuda()

                render_pkg = render(viewpoint, gaussians, pipe, background)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                l1_acc += l1_loss(image, gt_image).mean().double()
                psnr_acc += psnr(image, gt_image).mean().double()
                ssim_acc += ssim(image, gt_image).mean().double()
                msssim_acc += msssim(image[None].cpu(), gt_image[None].cpu())

            n = len(config["cameras"])
            print(
                "\n[{:5s}]  L1 {:.5f}  PSNR {:.3f}  SSIM {:.4f}  MS-SSIM {:.4f}  ({} views)".format(
                    config["name"],
                    l1_acc / n,
                    psnr_acc / n,
                    ssim_acc / n,
                    msssim_acc / n,
                    n,
                )
            )

    torch.cuda.empty_cache()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser(description="Test Script for Vanilla 4DGS.")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dynerf/coffee_martini.yaml",
        help="4DGS configure file.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--start_checkpoint",
        type=str,
        default="output/N3V/coffee_martini/chkpnt30000.pth",
        help="Optional: load weights from a torch .pth checkpoint "
        "(overrides --load_iteration)",
    )
    parser.add_argument(
        "--load_iteration",
        type=int,
        default=-1,
        help="Iteration under model_path/point_cloud/iteration_<N>/ "
        "to load (-1 = latest available)",
    )

    parser.add_argument("--gaussian_dim", type=int, default=3)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[-0.5, 0.5])
    parser.add_argument("--num_pts", type=int, default=100_000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    parser.add_argument("--rot_4d", action="store_true")
    parser.add_argument("--force_sh_3d", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)

    args = parser.parse_args(sys.argv[1:])

    if args.config:
        cfg = OmegaConf.load(args.config)

        # Soft merge: silently skip keys not registered on this parser (e.g.,
        # OptimizationParams block, batch_size, exhaust_test). Lets train.py
        # configs feed test.py without modification.
        def recursive_merge(key, host):
            if isinstance(host[key], DictConfig):
                for key1 in host[key].keys():
                    recursive_merge(key1, host[key])
            else:
                if hasattr(args, key):
                    setattr(args, key, host[key])

        for k in cfg.keys():
            recursive_merge(k, cfg)

    setup_seed(args.seed)
    print("Evaluating model at: " + args.model_path)
    safe_state(args.quiet)

    evaluate(
        lp.extract(args),
        pp.extract(args),
        args.start_checkpoint,
        args.load_iteration,
        args.gaussian_dim,
        args.time_duration,
        args.num_pts,
        args.num_pts_ratio,
        args.rot_4d,
        args.force_sh_3d,
    )

    print("\nEvaluation complete.")
