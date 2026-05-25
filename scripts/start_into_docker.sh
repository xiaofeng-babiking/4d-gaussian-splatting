#!/bin/bash

# Run the container as the host user so writes to /workspace (bind-mounted from the
# host) work natively — no chown gymnastics after uv sync / colmap output / etc.
# --group-add babiking keeps /home/babiking readable (where uv and the colmap
# install live, owned by babiking with 750 perms).
# HOME=/tmp because the host uid has no entry in /etc/passwd → no real home dir.
docker run -it --rm \
--name ironman \
--network host \
--shm-size 8G \
--user $(id -u):$(id -g) \
--env HOST_USER_ID=$(id -u) \
--env HOST_GROUP_ID=$(id -g) \
--env HOME=/tmp \
--env USER=$(whoami) \
--env TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions \
--env PYTHONHOME=/workspace/.venv \
--privileged \
-v $(pwd):/workspace \
-v /jfs:/jfs \
babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04 /bin/bash

# This image variant already bakes in:
#   /usr/local/bin/colmap -> /home/babiking/install/colmap/bin/colmap
# so no post-start `docker exec -u 0 ... ln -sf ...` step is needed any more.

# Resuming the uv env inside the new container:
#   source /workspace/.venv/bin/activate
#   # — or invoke directly: /workspace/.venv/bin/python train.py ...
# The .venv lives on the host bind-mount, so it survives `docker rm` and is
# immediately ready in any container that mounts the project at /workspace.
# JIT-built CUDA extensions (diff_gaussian_rasterization) are cached in
# /workspace/.torch_extensions/ via the env var above — also bind-mounted, so
# the first-run ~60 s compile happens only once per host, not per container.

# Optional X11 flags above (forwards DISPLAY at container start; goes stale on re-SSH):
# --env DISPLAY=$DISPLAY \
# -v $HOME/.Xauthority:/home/babiking/.Xauthority \
# -v /tmp/.X11-unix:/tmp/.X11-unix \

# Launch COLMAP GUI from the host (replace :10 with your current $DISPLAY number):
#   # 1. Inject host's X cookie into babiking's .Xauthority inside the container.
#   xauth -f ~/.Xauthority nlist :10 | docker exec -i -u 0 ironman \
#     bash -c 'xauth -f /home/babiking/.Xauthority nmerge - && chown babiking:babiking /home/babiking/.Xauthority'
#   # 2. Spawn the GUI detached; window opens on the host's X server.
#   docker exec -d ironman bash -c \
#     "DISPLAY=:10 XAUTHORITY=/home/babiking/.Xauthority /home/babiking/install/colmap/bin/colmap gui"
#   # Kill later: docker exec ironman pkill -f 'colmap gui'

# docker commit -a "babiking" -m "install 3rd-party dependencies" ironman babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04-gsplat1.5.2