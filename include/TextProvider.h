#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

class TextProvider {
public:
    virtual std::shared_ptr<const cv::cuda::GpuMat> getTextForNumber(int num) const = 0;
    virtual bool hasTextForNumber(int number) const = 0;
    virtual ~TextProvider() = default;
};
