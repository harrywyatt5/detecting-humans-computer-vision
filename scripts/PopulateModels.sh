#!/bin/bash
set -e

# Run this script before running a build for the first time. You should only need to run it once, unless
# you decide to delete sam3-models (which contain the .onnx blueprint files)

USER="jamjamjon"
RELEASE_TAG="sam3"
GITHUB_DOWNLOAD_URL="https://github.com/${USER}/assets/releases/download/${RELEASE_TAG}"
INSTALL_LOC="sam3-onnx"
MODELS=("vision-encoder-fp16.onnx" "text-encoder-fp16.onnx" "geo-encoder-mask-decoder-fp16.onnx")

function check_models_installed {
    for model in "${MODELS[@]}"
    do
        if [[! -e "${INSTALL_LOC}/${model}" ]]; then
            return [0]
        fi
    done

    echo "All models already exist in sam3-models directory. Doing nothing..." >&2
    exit 0
}

# Preprocessing
if [[ -d "${INSTALL_LOC}" ]]; then
    check_models_installed
else 
    mkdir sam3-models
fi

# Download
for MODEL in "${MODELS[@]}"
do
    wget "${GITHUB_DOWNLOAD_URL}/${MODEL}" -O "${INSTALL_LOC}/${MODEL}" >/dev/null 2>&1
done
