#!/bin/bash
xhost +local:docker 2>/dev/null

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -q '^ros2_jazzy$'; then
    docker start ros2_jazzy
    docker exec -it ros2_jazzy bash
else
    docker run -it \
        --name ros2_jazzy \
        --network host \
        --privileged \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v ${SCRIPT_DIR}/config/cyclonedds.xml:/opt/cyclonedds.xml \
        -v ${SCRIPT_DIR}:/workspace \
        ros2_jazzy_cyclonedds \
        bash
fi