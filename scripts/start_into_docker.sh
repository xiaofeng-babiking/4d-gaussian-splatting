#!/bin/bash

docker run -it --rm \
--name ironman \
--network host \
--shm-size 8G \
--user babiking:babiking \
--env HOST_USER_ID=$(id -u) \
--env HOST_GROUP_ID=$(id -g) \
--privileged \
-v $(pwd):/workspace \
-v /jfs:/jfs \
babiking/ubuntu:12.8.1-cudnn-devel-ubuntu24.04-gsplat1.5.2 /bin/bash

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