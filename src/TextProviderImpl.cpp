#include "TextProviderImpl.h"

#include "TextConfig.h"
#include <vector>
#include <stdexcept>
#include <memory>
#include <cstdint>
#include <opencv2/opencv.hpp>

TextProviderImpl::TextProviderImpl(int countExclusive, int devId, int x, int y, const TextConfig& config) 
    : frameX(x), frameY(y), deviceId(devId), textConfig(config)
{
    cv::cuda::setDevice(deviceId);
    createNumberTemplates(countExclusive);
}

void TextProviderImpl::createNumberTemplates(int count) {
    // First step, generate our templates. We know how big we want them to be but
    // not how big the text actually should be, so we do some quick maths to do this
    for (int i = 0; i < count; ++i) {
        std::string text = std::to_string(i);

        int baseLine = 0;  // Don't use this but need to pass something to OpenCV anyway...
        cv::Size textSize = cv::getTextSize(text, textConfig.getFontFace(), 1.0, textConfig.getThickness(), &baseLine);

        // Find the side we gotta divide by the most on and use that to scale our text
        double textScale = std::min((double)frameY / (double)textSize.height, (double)frameX / (double)textSize.width);

        cv::Mat newImage(frameY, frameX, CV_8UC4, cv::Scalar(0, 0, 0, 255));
        cv::putText(
            newImage,
            text,
            cv::Point(0, frameY),
            textConfig.getFontFace(),
            textScale,
            textConfig.getColour(),
            textConfig.getThickness()
        );

        auto newImageOnGpu = std::make_shared<cv::cuda::GpuMat>();
        // We block here rather than getting the CudaDevicesSingleton GPU stream.
        // If we didn't, the CPU would leave the loop and newImage would be freed :(
        newImageOnGpu->upload(newImage);
        numbers.push_back(newImageOnGpu);
    }
}

std::shared_ptr<const cv::cuda::GpuMat> TextProviderImpl::getTextForNumber(int number) const {
    if (number < 0 || (unsigned int)number > numbers.size()) {
        throw std::runtime_error("Invalid number to receive template for");
    }

    return numbers[number];
}

bool TextProviderImpl::hasTextForNumber(int number) const {
    return number >= 0 && (unsigned int)number < numbers.size(); 
}

int TextProviderImpl::getFrameX() const {
    return frameX;
}

int TextProviderImpl::getFrameY() const {
    return frameY;
}

int TextProviderImpl::getDeviceId() const {
    return deviceId;
}
