#!/usr/bin/env bash

# Determine absolute path to the script's directory and its parent
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJECT_ROOT="$SCRIPTPATH/.."

# Build docker container
$SCRIPTPATH/build.sh

# Create new docker volume for test
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
VOLUME_NAME="medgemma-finetuned-test-$VOLUME_SUFFIX"
docker volume create $VOLUME_NAME

# Run docker container with the test image mounted as input
MEM_LIMIT="30g"
docker run --rm\
        --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="bridge" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v "$SCRIPTPATH/test:/input" \
        -v "$VOLUME_NAME:/output" \
        medgemma-finetuned

read -rn1
