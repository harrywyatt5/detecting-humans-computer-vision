#include "MappedMask.h"
#include <opencv2/opencv.hpp>

void launchCreateTrackedMask(
    cv::cuda::GpuMat& maskOutput,
    cv::cuda::Stream& stream,
    const float* masksStart,
    const MappedMask* mappedMaskStart,
    const int masks
);
