#pragma once
#include <opencv2/core/cuda.hpp>

void launchCreateImage(
    cv::cuda::GpuMat& output,
    cv::cuda::Stream& stream,
    const float* masksStart,
    const uint8_t* masksInclude,
    const int masks
);
