#!/usr/bin/env bash

# Determine absolute path to the script's directory and its parent
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJECT_ROOT="$SCRIPTPATH/.."

# Build docker container
docker build -f "$PROJECT_ROOT/Dockerfile" -t medgemma-finetuned "$PROJECT_ROOT"
