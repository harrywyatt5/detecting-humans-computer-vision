#pragma once
#include <opencv2/core/cuda.hpp>

void launchNormaliseImage(const cv::cuda::GpuMat& mat, cv::cuda::Stream& stream, float* dst);
