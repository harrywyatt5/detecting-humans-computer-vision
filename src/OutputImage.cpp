#include "OutputImage.h"

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

OutputImage::OutputImage(int x, int y) : imageX(x), imageY(y), data(nullptr), isInUse(false) {
    auto result = cudaMalloc((void**)&data, x * y * 3 * sizeof(uint8_t));
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to allocate output buffer. Reason: ") + cudaGetErrorString(result));
    }

    auto memsetResult = cudaMemset((void*)&data, 0, x * y * 3 * sizeof(uint8_t));
    if (memsetResult != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to populate output buffer. Reason: ") + cudaGetErrorString(memsetResult));
    }
}

int OutputImage::getX() const {
    return imageX;
}

int OutputImage::getY() const {
    return imageY;
}

uint8_t* OutputImage::getData() {
    return data;
}

const uint8_t* OutputImage::getData() const {
    return data;
}

cv::cuda::GpuMat OutputImage::getDataAsGpuMat() {
    return cv::cuda::GpuMat(
        y,
        x,
        CV_8UC1,
        data,
        x * 3
    );
}