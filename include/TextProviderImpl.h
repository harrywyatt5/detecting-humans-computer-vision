#pragma once

#include "TextConfig.h"
#include "TextProvider.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class TextProviderImpl : public TextProvider {
private:
    int frameX;
    int frameY;
    int deviceId;
    TextConfig textConfig;
    std::vector<std::shared_ptr<cv::cuda::GpuMat>> numbers;
    void createNumberTemplates(int count);
public:
    TextProviderImpl(int countExclusive, int devId, int frameX, int frameY, const TextConfig& config);
    bool hasTextForNumber(int number) const override;
    std::shared_ptr<const cv::cuda::GpuMat> getTextForNumber(int number) const override;
    int getFrameX() const;
    int getFrameY() const;
    int getDeviceId() const;
};
