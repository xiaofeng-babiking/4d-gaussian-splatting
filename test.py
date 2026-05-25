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
from plyfile import PlyData, PlyElement

from utils.loss_utils import l1_loss, ssim, msssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams


def sample_4dgs_params_by_t(gaussians, timestamp):
    """Project a 4DGS model onto a static 3DGS parameter set at `timestamp`.

    Returns a dict of raw-parameter-space tensors keyed by the standard 3DGS
    names (xyz, features_dc, features_rest, scaling, rotation, opacity) so the
    result drops straight into a 3DGS PLY writer or into a fresh GaussianModel.

    For gaussian_dim=4 + rot_4d=True, spatial cov is the Schur complement of
    the joint 4D cov on t (Gaussian conditioning), and scaling/rotation are
    recovered via eigendecomposition + matrix->quat. For rot_4d=False, the
    joint cov is block-diagonal, so spatial geometry is time-invariant and only
    opacity gets the temporal-Gaussian gate. SH-4D coefficients (sh_degree_t>0)
    are baked at t by collapsing each cos(2pi*k*(mu_t-t)/L) temporal slab.
    """
    t = float(timestamp)

    if gaussians.gaussian_dim == 3:
        return {
            "xyz": gaussians._xyz.detach().clone(),
            "features_dc": gaussians._features_dc.detach().clone(),
            "features_rest": gaussians._features_rest.detach().clone(),
            "scaling": gaussians._scaling.detach().clone(),
            "rotation": gaussians._rotation.detach().clone(),
            "opacity": gaussians._opacity.detach().clone(),
        }

    # Opacity: multiply by the temporal Gaussian's value at t, then back to logit.
    marginal_t = gaussians.get_marginal_t(t)
    opacity_act = (gaussians.get_opacity * marginal_t).clamp(1e-6, 1.0 - 1e-6)
    opacity = torch.log(opacity_act / (1.0 - opacity_act)).detach()

    # SH features: bake the temporal basis at t when sh_degree_t > 0.
    if gaussians.max_sh_degree_t > 0:
        L = gaussians.time_duration[1] - gaussians.time_duration[0]
        dir_t = (gaussians.get_t - t).detach()  # [N, 1]
        spatial_n = (gaussians.max_sh_degree + 1) ** 2
        sh_full = torch.cat([gaussians._features_dc, gaussians._features_rest], dim=1)
        baked = sh_full[:, :spatial_n, :].clone()
        for k in range(1, gaussians.max_sh_degree_t + 1):
            mod = torch.cos(2 * torch.pi * k * dir_t / L)
            baked = baked + mod[:, :, None] * sh_full[:, k * spatial_n : (k + 1) * spatial_n, :]
        features_dc = baked[:, :1, :].detach()
        features_rest = baked[:, 1:, :].detach()
    else:
        features_dc = gaussians._features_dc.detach().clone()
        features_rest = gaussians._features_rest.detach().clone()

    if not gaussians.rot_4d:
        # Block-diagonal 4D cov: spatial geometry is time-invariant.
        return {
            "xyz": gaussians._xyz.detach().clone(),
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": gaussians._scaling.detach().clone(),
            "rotation": gaussians._rotation.detach().clone(),
            "opacity": opacity,
        }

    # rot_4d=True: condition on t to drift position and reshape spatial cov.
    cond_cov, delta_mean = gaussians.get_current_covariance_and_mean_offset(
        scaling_modifier=1.0, timestamp=t
    )
    xyz = (gaussians._xyz + delta_mean).detach()

    # Sigma_xx|t = V diag(lambda) V^T  ->  s = sqrt(lambda), R = V (det=+1).
    eigvals, eigvecs = torch.linalg.eigh(cond_cov)
    det = torch.linalg.det(eigvecs)
    sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
    eigvecs = torch.cat([eigvecs[:, :, :2], eigvecs[:, :, 2:3] * sign], dim=2)

    s = torch.sqrt(eigvals.clamp(min=1e-12))
    scaling = torch.log(s).detach()
    rotation = _matrix_to_quat_wxyz(eigvecs).detach()

    return {
        "xyz": xyz,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "scaling": scaling,
        "rotation": rotation,
        "opacity": opacity,
    }


def _matrix_to_quat_wxyz(R):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z), Shepperd's method.

    Picks one of four formulas based on which of {trace, R[0,0], R[1,1], R[2,2]}
    is largest, to avoid catastrophic cancellation when w (or any component) is
    near zero.
    """
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

    case1 = tr > 0
    rest = ~case1
    case2 = rest & (m00 >= m11) & (m00 >= m22)
    case3 = rest & ~case2 & (m11 >= m22)
    case4 = rest & ~case2 & ~case3

    s1 = torch.sqrt(tr.clamp(min=-1.0 + 1e-12) + 1.0) * 2.0
    q[case1, 0] = 0.25 * s1[case1]
    q[case1, 1] = (R[case1, 2, 1] - R[case1, 1, 2]) / s1[case1]
    q[case1, 2] = (R[case1, 0, 2] - R[case1, 2, 0]) / s1[case1]
    q[case1, 3] = (R[case1, 1, 0] - R[case1, 0, 1]) / s1[case1]

    s2 = torch.sqrt((1.0 + m00 - m11 - m22).clamp(min=1e-12)) * 2.0
    q[case2, 0] = (R[case2, 2, 1] - R[case2, 1, 2]) / s2[case2]
    q[case2, 1] = 0.25 * s2[case2]
    q[case2, 2] = (R[case2, 0, 1] + R[case2, 1, 0]) / s2[case2]
    q[case2, 3] = (R[case2, 0, 2] + R[case2, 2, 0]) / s2[case2]

    s3 = torch.sqrt((1.0 - m00 + m11 - m22).clamp(min=1e-12)) * 2.0
    q[case3, 0] = (R[case3, 0, 2] - R[case3, 2, 0]) / s3[case3]
    q[case3, 1] = (R[case3, 0, 1] + R[case3, 1, 0]) / s3[case3]
    q[case3, 2] = 0.25 * s3[case3]
    q[case3, 3] = (R[case3, 1, 2] + R[case3, 2, 1]) / s3[case3]

    s4 = torch.sqrt((1.0 - m00 - m11 + m22).clamp(min=1e-12)) * 2.0
    q[case4, 0] = (R[case4, 1, 0] - R[case4, 0, 1]) / s4[case4]
    q[case4, 1] = (R[case4, 0, 2] + R[case4, 2, 0]) / s4[case4]
    q[case4, 2] = (R[case4, 1, 2] + R[case4, 2, 1]) / s4[case4]
    q[case4, 3] = 0.25 * s4[case4]

    return q


def dump_supersplat_ply_file(path, params):
    """Write SuperSplat-format PLY from a sampled 3DGS parameter dict.

    Field order follows PlayCanvas's documented schema:
      x, y, z, scale_*, rot_*, opacity, f_dc_*, f_rest_*
    No normals. All fields are float32. Activation conventions match the
    inria/3DGS convention SuperSplat consumes: raw logit opacity, raw log-space
    scaling, raw (w, x, y, z) quaternion (consumers re-normalize).
    """
    xyz = params["xyz"].cpu().numpy()
    scaling = params["scaling"].cpu().numpy()
    rotation = params["rotation"].cpu().numpy()
    opacity = params["opacity"].cpu().numpy()
    # SH layout: [N, K, 3] -> transpose -> [N, 3, K] -> flatten gives channel-major
    # (ch0 K coeffs, ch1 K coeffs, ch2 K coeffs) matching the 3DGS reference exporter.
    features_dc = params["features_dc"].cpu().numpy().transpose(0, 2, 1).reshape(xyz.shape[0], -1)
    features_rest = params["features_rest"].cpu().numpy().transpose(0, 2, 1).reshape(xyz.shape[0], -1)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    dtype += [(f"scale_{i}", "f4") for i in range(scaling.shape[1])]
    dtype += [(f"rot_{i}", "f4") for i in range(rotation.shape[1])]
    dtype += [("opacity", "f4")]
    dtype += [(f"f_dc_{i}", "f4") for i in range(features_dc.shape[1])]
    dtype += [(f"f_rest_{i}", "f4") for i in range(features_rest.shape[1])]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate(
        [xyz, scaling, rotation, opacity, features_dc, features_rest], axis=1
    )
    elements[:] = list(map(tuple, attributes))

    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


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
                # Dump SuperSplat PLY of the 4DGS sampled at this frame's t.
                # NOTE: each PLY is ~(num_gaussians * 51) floats; can be very large
                # at full test-set cadence. Comment this block out to skip.
                params_at_t = sample_4dgs_params_by_t(gaussians, viewpoint.timestamp)
                dump_supersplat_ply_file(
                    os.path.join(render_dir, basename + ".ply"), params_at_t
                )

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
    )

    print("\nEvaluation complete.")
