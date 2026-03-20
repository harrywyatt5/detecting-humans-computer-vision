#pragma once

#include "TextConfig.h"
#include "TextProvider.h"
#include <opencv2/opencv.hpp>
#include <vector>

class TextProviderImpl : public TextProvider {
private:
    int frameX;
    int frameY;
    int deviceId;
    TextConfig textConfig;
    std::vector<cv::cuda::GpuMat> numbers;
    void createNumberTemplates(int count);
public:
    TextProviderImpl(int count, int devId, int frameX, int frameY, const TextConfig& config);
    const std::vector<cv::cuda::GpuMat> getTemplateNumberList() const;
    const cv::cuda::GpuMat& getTemplateNumber() const;
    int getFrameX() const;
    int getFrameY() const;
    int getDeviceId() const;
};
