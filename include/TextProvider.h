#pragma once

#include <opencv2/opencv.hpp>

class TextProvider {
public:
    virtual cv::cuda::GpuMat& getTextForNumber(int num) = 0;
    virtual ~TextProvider() = default;
};
