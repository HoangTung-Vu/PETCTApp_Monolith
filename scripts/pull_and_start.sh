#!/bin/bash
# pull_and_start.sh — DEPRECATED
#
# The nnU-Net engine image is not distributed on DockerHub because model weights
# are not publicly available. To run the application, use the root launcher instead:
#
#   ./start.sh        (Linux)
#   .\start.bat       (Windows)
#
# The launcher builds the Docker image locally from source (weights must be placed
# under AI_engines/engine_nnunet_old_ver/weights/ beforehand — see README.md).

echo "This script is no longer used."
echo "Use ./start.sh (Linux) or .\\start.bat (Windows) to build and run the engine locally."
echo "See README.md for weights setup instructions."
exit 1
