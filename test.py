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

import os
import random
import sys
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.loss_utils import l1_loss, ssim, msssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams


def make_round_trip_cameras(anchor_cam, num_frames, t_lo, t_hi, look_distance=2.0):
    """Build a 360° orbital camera trajectory around the scene.

    Frame 0 starts at `anchor_cam`'s pose so the orbit visually anchors against
    a known test-set view. Subsequent frames orbit the focal point in the
    horizontal plane (world +Z up, matching N3V/Blender). Time advances
    linearly from `t_lo` to `t_hi` across the orbit, so the rendered video
    shows spatial AND temporal change combined.

    `look_distance` (default 2.0 world units) is how far ahead of the anchor
    we place the orbit center. For N3V coffee_martini this lands roughly on
    the bartender area; bump it for wider scenes, shrink for closer subjects.

    Critical: FoVx/FoVy are set to -1 to match the codebase's latent FOV-clamp
    behavior (forward.cu evaluates `tan(FoVx * 0.5)` for the projection clamp,
    and the model is trained against `tan(-0.5) = -0.546` — passing positive
    FoV shifts the render ~5 dB off). Real intrinsics still go through
    `fl_x/fl_y/cx/cy` for the projection matrix itself.
    """
    R = np.asarray(anchor_cam.R, dtype=np.float64)
    T = np.asarray(anchor_cam.T, dtype=np.float64)
    pos0 = -R @ T
    forward0 = R[:, 2]
    focal_point = pos0 + forward0 * look_distance

    z_axis = np.array([0.0, 0.0, 1.0])
    radius_vec = pos0 - focal_point
    height_offset = float(radius_vec @ z_axis)
    horiz_vec = radius_vec - height_offset * z_axis

    cameras = []
    for i in range(num_frames):
        theta = 2.0 * np.pi * i / num_frames
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        new_pos = focal_point + Rz @ horiz_vec + height_offset * z_axis

        forward = focal_point - new_pos
        forward /= np.linalg.norm(forward)
        # If camera ends up nearly above/below the focal point, the world-Z
        # cross product degenerates; fall back to +X as the up hint.
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(forward @ z_axis) > 0.999 else z_axis
        right = np.cross(forward, up_hint)
        right /= np.linalg.norm(right)
        up_actual = np.cross(right, forward)

        # OpenCV camera-local: X=right, Y=down, Z=forward.
        R_c2w = np.stack([right, -up_actual, forward], axis=1)
        T_w2c = -R_c2w.T @ new_pos

        timestamp = t_lo + (t_hi - t_lo) * (i / max(num_frames - 1, 1))

        cameras.append(
            Camera(
                colmap_id=i,
                R=R_c2w.astype(np.float32),
                T=T_w2c.astype(np.float32),
                FoVx=-1.0,
                FoVy=-1.0,
                image=torch.empty(0),
                gt_alpha_mask=None,
                image_name=f"orbit_{i:04d}",
                uid=i,
                resolution=(anchor_cam.image_width, anchor_cam.image_height),
                cx=anchor_cam.cx,
                cy=anchor_cam.cy,
                fl_x=anchor_cam.fl_x,
                fl_y=anchor_cam.fl_y,
                meta_only=True,
                data_device="cuda",
                timestamp=timestamp,
            )
        )
    return cameras


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
    add_orbit_trip=False,
    orbit_trip_frames=60,
    orbit_trip_look_distance=2.0,
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
            skip_train_cams=True,
        )
        # weights_only=False: torch 2.6+ flipped the default to True (security), which
        # refuses to unpickle the numpy scalars + Python tuple that gaussians.capture()
        # produces. The checkpoint is trusted (we wrote it via train.py).
        model_params, ckpt_iter = torch.load(checkpoint, weights_only=False)
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
            skip_train_cams=True,
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if add_orbit_trip:
        # Orbit-only mode: skip the test-cam eval entirely and render the
        # 360° trajectory to <model_path>/orbit/. Frame 0 of the orbit matches
        # the first test cam's pose, so visual sanity-check is direct.
        anchor = scene.test_cameras[1.0][0]
        t_lo, t_hi = float(time_duration[0]), float(time_duration[1])
        orbit_cams = make_round_trip_cameras(
            anchor, orbit_trip_frames, t_lo, t_hi, look_distance=orbit_trip_look_distance
        )
        orbit_dir = os.path.join(dataset.model_path, "orbit")
        os.makedirs(orbit_dir, exist_ok=True)
        print(
            f"\nOrbit-only mode: {orbit_trip_frames} frames, "
            f"t=[{t_lo:.3f}..{t_hi:.3f}], look_dist={orbit_trip_look_distance} -> {orbit_dir}"
        )
        with torch.no_grad():
            for cam in tqdm(orbit_cams, desc="Round-trip"):
                cam_cuda = cam.cuda()
                pkg = render(cam_cuda, gaussians, pipe, background)
                image = torch.clamp(pkg["render"], 0.0, 1.0)
                angle_deg = round(360.0 * cam.uid / orbit_trip_frames)
                fname = f"orbit_A{angle_deg:03d}_T{cam.timestamp:.3f}.png"
                save_image(image, os.path.join(orbit_dir, fname))
        torch.cuda.empty_cache()
        return

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
        # Pass the CameraDataset directly (not a list comprehension) — its
        # __getitem__ decodes a ~2 MP JPEG from /jfs on each access, which takes
        # ~3 s per camera. Eager materialization of all 300 test cams burns ~18 min
        # before tqdm even starts. Lazy iteration overlaps decode with rendering.
        {"name": "test", "cameras": test_cams},
    )

    with torch.no_grad():
        for config in validation_configs:
            if not config["cameras"]:
                continue
            render_dir = os.path.join(dataset.model_path, "render")
            os.makedirs(render_dir, exist_ok=True)
            l1_acc = 0.0
            psnr_acc = 0.0
            ssim_acc = 0.0
            msssim_acc = 0.0
            # Parallel-decode the 2704x2028 PNGs in worker processes so the slow
            # PIL decode + resize (~7 s each on this dataset) overlaps with GPU
            # render. Without this, eval is decode-bound at ~15 s/iter on N3V.
            loader = DataLoader(
                config["cameras"],
                batch_size=1,
                num_workers=8,
                collate_fn=lambda x: x[0],
            )
            for batch_idx, batch_data in enumerate(
                tqdm(
                    loader,
                    total=len(config["cameras"]),
                    desc="Evaluating {}".format(config["name"]),
                )
            ):
                gt_image, viewpoint = batch_data
                gt_image = gt_image.cuda()
                viewpoint = viewpoint.cuda()

                render_pkg = render(viewpoint, gaussians, pipe, background)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)

                # image_name follows N3V's "cam{NN}_{FFFF}" stem from n3v2blender.py.
                cam_part, frame_part = viewpoint.image_name.rsplit("_", 1)
                cam_idx = int(cam_part[3:])
                frame_idx = int(frame_part)
                basename = f"{config['name']}_C{cam_idx:04d}_F{frame_idx:04d}"
                save_image(image, os.path.join(render_dir, basename + ".png"))

                l1_cur = l1_loss(image, gt_image).mean().double()
                l1_acc += l1_cur

                psnr_cur = psnr(image, gt_image).mean().double()
                psnr_acc += psnr_cur

                ssim_cur = ssim(image, gt_image).mean().double()
                ssim_acc += ssim_cur

                mssim_cur = msssim(image[None].cpu(), gt_image[None].cpu())
                msssim_acc += mssim_cur

                print(
                    f"Indedx={batch_idx}, "
                    + f"L1={l1_cur:.4f}, PSNR={psnr_cur:.4f}, "
                    + f"SSIM={ssim_cur:.4f}, mSSIM={mssim_cur:.4f}."
                )

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

    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument(
        "--add_orbit_trip",
        action="store_true",
        help="Skip the test-cam evaluation and instead render a 360° orbital "
        "trajectory to <model_path>/orbit/. Useful for free-view inspection "
        "of dynamic scenes without spinning up the live viewer.",
    )
    parser.add_argument(
        "--orbit_trip_frames",
        type=int,
        default=60,
        help="Number of frames in the orbital trajectory. 60 = 6° per frame.",
    )
    parser.add_argument(
        "--orbit_trip_look_distance",
        type=float,
        default=2.0,
        help="World units ahead of the anchor camera to place the orbit center. "
        "Adjust for scene scale; ~2.0 is right for N3V coffee_martini.",
    )
    # Structural args (gaussian_dim, time_duration, num_pts, num_pts_ratio,
    # rot_4d, force_sh_3d) intentionally NOT registered: they describe the
    # tensor layout of the saved model and must come from the training-time
    # source-of-truth (the --config YAML), not from CLI where a typo would
    # silently mis-shape the parameters at load.

    args = parser.parse_args(sys.argv[1:])

    if args.config:
        cfg = OmegaConf.load(args.config)

        # Merge every leaf key onto args. Structural args (gaussian_dim,
        # time_duration, num_pts, num_pts_ratio, rot_4d, force_sh_3d) are no
        # longer registered on argparse and land here. Unknown training-only
        # keys (OptimizationParams block, batch_size, exhaust_test) also land
        # but go unread.
        def recursive_merge(key, host):
            if isinstance(host[key], DictConfig):
                for key1 in host[key].keys():
                    recursive_merge(key1, host[key])
            else:
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
        add_orbit_trip=args.add_orbit_trip,
        orbit_trip_frames=args.orbit_trip_frames,
        orbit_trip_look_distance=args.orbit_trip_look_distance,
    )

    print("\nEvaluation complete.")
