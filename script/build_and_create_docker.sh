#!/bin/bash
echo "Staring the build process, this might take a while..."

# Set the x86_64 platform
docker container rm pao_container

docker build --progress=plain -t pao_container_image -f Dockerfile .

echo "Creating container..."
docker create \
    --shm-size=2048m \
    --privileged \
    -v ${PWD}/workdir:/app:rw \
    --name pao_container pao_container_image

echo "Container created! Run the 'scripts/start_container.sh' script to start the container"