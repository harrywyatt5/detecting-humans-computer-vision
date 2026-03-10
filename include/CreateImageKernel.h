#pragma once
#include <opencv2/core/cuda.hpp>

void launchCreateImage(
    const cv::cuda::GpuMat& maskInput,
    cv::cuda::GpuMat& outputImage,
    cv::cuda::Stream& stream,
    const float mixPercentage
);

void launchCreateMask(
    cv::cuda::GpuMat& maskOutput,
    cv::cuda::Stream& stream,
    const float* masksStart,
    const uint8_t* masksInclude,
    const int masks
);
