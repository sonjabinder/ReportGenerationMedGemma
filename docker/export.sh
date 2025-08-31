#!/usr/bin/env bash

# Determine absolute path to the script's directory and its parent
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJECT_ROOT="$SCRIPTPATH/.."

# Build docker container
$SCRIPTPATH/build.sh

# Export docker container
docker save medgemma-finetuned | gzip -c > "$SCRIPTPATH/export/medgemma-finetuned.tar.gz"
