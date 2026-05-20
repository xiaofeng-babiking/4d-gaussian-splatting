import argparse
import glob
import json
import math
import os
import re
import sqlite3
import subprocess
import sys

import numpy as np

# Schema below was lifted verbatim from COLMAP at commit afe04f56 (v3.14.0.dev0):
#   src/colmap/scene/database_sqlite.cc :: Create*Table()
# The rigs / rig_sensors / frames / frame_data / pose_priors tables were added in
# the 3.10 rig-refactor; before that, pose priors lived inline on the images table
# (columns prior_qw, prior_qx, ..., prior_tz). MIN_COLMAP_VERSION below gates that.
MIN_COLMAP_VERSION = (3, 10)
MAX_IMAGE_ID = 2**31 - 1

COLMAP_SCHEMA = {
    "rigs": (
        "CREATE TABLE IF NOT EXISTS rigs ("
        " rig_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
        " ref_sensor_id INTEGER NOT NULL,"
        " ref_sensor_type INTEGER NOT NULL)"
    ),
    "rig_sensors": (
        "CREATE TABLE IF NOT EXISTS rig_sensors ("
        " rig_id INTEGER NOT NULL,"
        " sensor_id INTEGER NOT NULL,"
        " sensor_type INTEGER NOT NULL,"
        " sensor_from_rig BLOB,"
        " FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE)"
    ),
    "cameras": (
        "CREATE TABLE IF NOT EXISTS cameras ("
        " camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
        " model INTEGER NOT NULL,"
        " width INTEGER NOT NULL,"
        " height INTEGER NOT NULL,"
        " params BLOB,"
        " prior_focal_length INTEGER NOT NULL)"
    ),
    "frames": (
        "CREATE TABLE IF NOT EXISTS frames ("
        " frame_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
        " rig_id INTEGER NOT NULL,"
        " FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE)"
    ),
    "frame_data": (
        "CREATE TABLE IF NOT EXISTS frame_data ("
        " frame_id INTEGER NOT NULL,"
        " data_id INTEGER NOT NULL,"
        " sensor_id INTEGER NOT NULL,"
        " sensor_type INTEGER NOT NULL,"
        " FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE)"
    ),
    "images": (
        "CREATE TABLE IF NOT EXISTS images ("
        " image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,"
        " name TEXT NOT NULL UNIQUE,"
        " camera_id INTEGER NOT NULL,"
        f" CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_IMAGE_ID}),"
        " FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"
    ),
    "pose_priors": (
        "CREATE TABLE IF NOT EXISTS pose_priors ("
        " image_id INTEGER PRIMARY KEY NOT NULL,"
        " position BLOB,"
        " coordinate_system INTEGER NOT NULL,"
        " position_covariance BLOB,"
        " FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"
    ),
    "keypoints": (
        "CREATE TABLE IF NOT EXISTS keypoints ("
        " image_id INTEGER PRIMARY KEY NOT NULL,"
        " rows INTEGER NOT NULL,"
        " cols INTEGER NOT NULL,"
        " data BLOB,"
        " FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"
    ),
    "descriptors": (
        "CREATE TABLE IF NOT EXISTS descriptors ("
        " image_id INTEGER PRIMARY KEY NOT NULL,"
        " rows INTEGER NOT NULL,"
        " cols INTEGER NOT NULL,"
        " data BLOB,"
        " FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"
    ),
    "matches": (
        "CREATE TABLE IF NOT EXISTS matches ("
        " pair_id INTEGER PRIMARY KEY NOT NULL,"
        " rows INTEGER NOT NULL,"
        " cols INTEGER NOT NULL,"
        " data BLOB)"
    ),
    "two_view_geometries": (
        "CREATE TABLE IF NOT EXISTS two_view_geometries ("
        " pair_id INTEGER PRIMARY KEY NOT NULL,"
        " rows INTEGER NOT NULL,"
        " cols INTEGER NOT NULL,"
        " data BLOB,"
        " config INTEGER NOT NULL,"
        " F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB)"
    ),
}

COLMAP_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS rig_ref_sensor_assignment ON rigs(ref_sensor_id, ref_sensor_type)",
    "CREATE UNIQUE INDEX IF NOT EXISTS rig_sensor_assignment ON rig_sensors(sensor_id, sensor_type)",
    "CREATE UNIQUE INDEX IF NOT EXISTS frame_sensor_assignment ON frame_data(data_id, sensor_type)",
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)",
]


def _parse_version(v):
    """Parse e.g. '3.14.0.dev0' -> (3, 14). Returns (0, 0) for None/unparseable."""
    if not v:
        return (0, 0)
    m = re.match(r"(\d+)\.(\d+)", v)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def get_colmap_version():
    """Return the COLMAP version string (e.g. "3.14.0.dev0"), or None if the binary
    is not callable. Parses the header line that COLMAP prints to stdout/stderr:
        COLMAP 3.14.0.dev0 -- Structure-from-Motion and Multi-View Stereo
    """
    try:
        out = subprocess.run(
            ["colmap", "--help"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in (out.stdout + out.stderr).splitlines():
        m = re.match(r"COLMAP (\S+)", line)
        if m:
            return m.group(1)
    return None


def check_colmap_compat():
    """Abort if COLMAP isn't callable; warn if it predates our embedded schema."""
    version = get_colmap_version()
    if version is None:
        sys.exit("FATAL: `colmap` not on PATH. Install COLMAP or add it to PATH.")
    if _parse_version(version) < MIN_COLMAP_VERSION:
        print(
            f"WARNING: COLMAP {version} predates the rigs/frames refactor "
            f"(introduced in {MIN_COLMAP_VERSION[0]}.{MIN_COLMAP_VERSION[1]}). "
            f"The embedded schema in COLMAP_SCHEMA reflects the post-3.10 layout; "
            f"COLMAP itself still owns the live schema, so updates are safe."
        )
    return version


def array_to_blob(array):
    # `ndarray.tostring()` was removed in NumPy 2.0; `tobytes()` is the modern equivalent.
    return array.tobytes()


def blob_to_array(blob, dtype, shape=(-1,)):
    # `np.fromstring` was removed in NumPy 2.0; `np.frombuffer` is the modern equivalent.
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    """SQLite wrapper for COLMAP databases.

    In this script's pipeline the schema is materialised by `colmap feature_extractor`,
    so `update_camera` is the only path actually exercised. `create_tables` is provided
    for callers that need a standalone DB; it mirrors the schema embedded above
    (COLMAP_SCHEMA / COLMAP_INDEXES).
    """

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def create_tables(self):
        for sql in COLMAP_SCHEMA.values():
            self.execute(sql)
        for sql in COLMAP_INDEXES:
            self.execute(sql)
        self.commit()

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id),
        )
        return cursor.lastrowid


def camTodatabase(txtfile, database_path):
    camModelDict = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "FULL_OPENCV": 5,
        "SIMPLE_RADIAL_FISHEYE": 6,
        "RADIAL_FISHEYE": 7,
        "OPENCV_FISHEYE": 8,
        "FOV": 9,
        "THIN_PRISM_FISHEYE": 10,
    }

    if not os.path.exists(database_path):
        print("ERROR: database path doesn't exist -- please check database.db.")
        return

    db = COLMAPDatabase.connect(database_path)

    idList, modelList, widthList, heightList, paramsList = [], [], [], [], []
    with open(txtfile, "r") as cam:
        for line in cam.readlines():
            if line.startswith("#"):
                continue
            strLists = line.split()
            cameraId = int(strLists[0])
            cameraModel = camModelDict[strLists[1]]
            width = int(strLists[2])
            height = int(strLists[3])
            params = np.array(strLists[4:12]).astype(np.float64)
            idList.append(cameraId)
            modelList.append(cameraModel)
            widthList.append(width)
            heightList.append(height)
            paramsList.append(params)
            db.update_camera(cameraModel, width, height, params, cameraId)

    db.commit()

    rows = db.execute("SELECT * FROM cameras")
    for i in range(len(idList)):
        camera_id, model, width, height, params, _ = next(rows)
        params = blob_to_array(params, np.float64)
        assert camera_id == idList[i]
        assert (
            model == modelList[i] and width == widthList[i] and height == heightList[i]
        )
        assert np.allclose(params, paramsList[i])

    db.close()


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


# returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
def closest_point_2_lines(oa, da, ob, db):
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def build_transforms(args_path):
    """Compute train/test transforms from poses_bounds.npy + image listing. Idempotent."""
    images = [
        f[len(args_path) :]
        for f in sorted(glob.glob(os.path.join(args_path, "images", "*")))
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    cams = sorted(set([im[7:12] for im in images]))

    poses_bounds = np.load(os.path.join(args_path, "poses_bounds.npy"))
    N = poses_bounds.shape[0]
    print(
        f"[INFO] loaded {len(images)} images from {len(cams)} videos, {N} poses_bounds as {poses_bounds.shape}"
    )
    assert N == len(cams)

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
    H, W, fl = poses[0, :, -1]
    print(f"[INFO] H = {H}, W = {W}, fl = {fl}")

    # inversion of https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
    poses = np.concatenate(
        [poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1
    )
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses = np.concatenate([poses, last_row], axis=1)  # (N, 4, 4)

    # colmap2nerf-style axis fixups
    poses[:, 0:3, 1] *= -1
    poses[:, 0:3, 2] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1

    up = poses[:, 0:3, 1].sum(0)
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    poses = R @ poses

    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in range(N):
        mf = poses[i, :3, :]
        for j in range(i + 1, N):
            mg = poses[j, :3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(f"[INFO] totp = {totp}")
    poses[:, :3, 3] -= totp

    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    poses[:, :3, 3] *= 4.0 / avglen
    print(f"[INFO] average radius = {avglen}")

    train_frames, test_frames = [], []
    for i in range(N):
        cam_frames = [
            {
                "file_path": im.lstrip("/").split(".")[0],
                "transform_matrix": poses[i].tolist(),
                "time": int(im.lstrip("/").split(".")[0][-4:]) / 30.0,
            }
            for im in images
            if cams[i] in im
        ]
        if i == 0:
            test_frames += cam_frames
        else:
            train_frames += cam_frames

    common = {"w": W, "h": H, "fl_x": fl, "fl_y": fl, "cx": W // 2, "cy": H // 2}
    return {**common, "frames": train_frames}, {**common, "frames": test_frames}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="", help="input path to the video")
    args = parser.parse_args()

    if args.path[-1] != "/":
        args.path += "/"

    # ---- compatibility / sanity checks ----
    colmap_version = check_colmap_compat()
    print(f"[INFO] COLMAP {colmap_version}")
    print(f"[INFO] NumPy {np.__version__}")

    # ---- paths (everything lives directly under args.path) ----
    images_path = os.path.join(args.path, "images")
    train_json = os.path.join(args.path, "transforms_train.json")
    test_json = os.path.join(args.path, "transforms_test.json")
    points_ply = os.path.join(args.path, "points3d.ply")
    db_path = os.path.join(args.path, "database.db")
    sparse_in = os.path.join(args.path, "sparse_in")   # manual model w/ known poses
    sparse_out = os.path.join(args.path, "sparse")     # triangulated model
    dense_dir = os.path.join(args.path, "dense")       # MVS workspace
    cameras_txt = os.path.join(sparse_in, "cameras.txt")
    images_txt = os.path.join(sparse_in, "images.txt")
    points3d_txt = os.path.join(sparse_in, "points3D.txt")
    image_list_txt = os.path.join(sparse_in, "image_list.txt")

    # ---- step 1: extract frames from .mp4 with ffmpeg ----
    if not os.path.exists(images_path) or not os.listdir(images_path):
        os.makedirs(images_path, exist_ok=True)
        videos = [
            os.path.join(args.path, v)
            for v in os.listdir(args.path)
            if v.endswith(".mp4")
        ]
        for video in videos:
            cam_name = video.split("/")[-1].split(".")[-2]
            do_system(
                f"ffmpeg -i {video} -start_number 0 {images_path}/{cam_name}_%04d.png"
            )
    else:
        print(f"[SKIP] ffmpeg extraction: {images_path} already populated")

    # ---- step 2: train/test transforms (cached as JSON) ----
    if os.path.exists(train_json) and os.path.exists(test_json):
        print(f"[SKIP] transforms_*.json already exist")
        with open(train_json) as f:
            train_transforms = json.load(f)
        with open(test_json) as f:
            test_transforms = json.load(f)
    else:
        train_transforms, test_transforms = build_transforms(args.path)
        print(f"[INFO] write {train_json} and {test_json}")
        with open(train_json, "w") as f:
            json.dump(train_transforms, f, indent=2)
        with open(test_json, "w") as f:
            json.dump(test_transforms, f, indent=2)

    # ---- step 3: stop if final point cloud already exists ----
    if os.path.exists(points_ply):
        print(f"[SKIP] COLMAP pipeline: {points_ply} already exists")
        sys.exit(0)

    # ---- step 4: prepare the manual sparse model (cameras.txt / images.txt) ----
    W = int(train_transforms["w"])
    H = int(train_transforms["h"])
    cx, cy = train_transforms["cx"], train_transforms["cy"]
    fx, fy = train_transforms["fl_x"], train_transforms["fl_y"]
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )

    os.makedirs(sparse_in, exist_ok=True)

    fname2pose = {}
    for frame in train_transforms["frames"]:
        if frame["time"] == 0:
            fname = frame["file_path"].split("/")[-1] + ".png"
            fname2pose[fname] = np.array(frame["transform_matrix"]) @ blender2opencv

    if not os.path.exists(cameras_txt):
        with open(cameras_txt, "w") as f:
            f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}")

    # `--image_list_path` filters which images COLMAP scans from `images/` —
    # avoids the symlink / tmp-folder dance and avoids feature-extracting every
    # frame of every video.
    if not os.path.exists(image_list_txt):
        with open(image_list_txt, "w") as f:
            for fname in fname2pose:
                f.write(f"{fname}\n")

    if not os.path.exists(images_txt):
        with open(images_txt, "w") as f:
            for idx, (fname, pose) in enumerate(fname2pose.items(), start=1):
                R = np.linalg.inv(pose[:3, :3])
                T = -np.matmul(R, pose[:3, 3])
                q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
                q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
                q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
                q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
                f.write(f"{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n")

    if not os.path.exists(points3d_txt):
        open(points3d_txt, "w").close()

    # ---- step 5: SfM (feature + match), reading directly from data_root/images ----
    if not os.path.exists(db_path):
        # --ImageReader.single_camera=1 → one shared camera (and one rig) across
        # all images, matching the single PINHOLE entry in cameras.txt. Otherwise
        # COLMAP 3.10+ creates one camera/rig per image and Reconstruction::Load()
        # in point_triangulator rejects the rig mismatch against our text model.
        do_system(
            f"colmap feature_extractor "
            f"--database_path {db_path} "
            f"--image_path {images_path} "
            f"--image_list_path {image_list_txt} "
            f"--ImageReader.single_camera 1"
        )
        camTodatabase(cameras_txt, db_path)
        do_system(f"colmap exhaustive_matcher --database_path {db_path}")
    else:
        print(f"[SKIP] feature_extractor + matcher: {db_path} exists")

    # ---- step 6: triangulate with known poses ----
    if not os.path.exists(sparse_out) or not os.listdir(sparse_out):
        os.makedirs(sparse_out, exist_ok=True)
        do_system(
            f"colmap point_triangulator "
            f"--database_path {db_path} "
            f"--image_path {images_path} "
            f"--input_path {sparse_in} "
            f"--output_path {sparse_out}"
        )
        do_system(
            f"colmap model_converter "
            f"--input_path {sparse_out} "
            f"--output_path {sparse_out} "
            f"--output_type TXT"
        )
    else:
        print(f"[SKIP] point_triangulator: {sparse_out} populated")

    # ---- step 7: MVS (undistort + patch_match + fusion) ----
    if not os.path.exists(dense_dir) or not os.listdir(dense_dir):
        os.makedirs(dense_dir, exist_ok=True)
        do_system(
            f"colmap image_undistorter "
            f"--image_path {images_path} "
            f"--input_path {sparse_out} "
            f"--output_path {dense_dir}"
        )
        do_system(f"colmap patch_match_stereo --workspace_path {dense_dir}")
    else:
        print(f"[SKIP] image_undistorter + patch_match_stereo: {dense_dir} populated")

    do_system(
        f"colmap stereo_fusion "
        f"--workspace_path {dense_dir} "
        f"--output_path {points_ply}"
    )

    vis_path = points_ply + ".vis"
    if os.path.exists(vis_path):
        os.remove(vis_path)

    print(f"[INFO] Initial point cloud is saved in {points_ply}.")
