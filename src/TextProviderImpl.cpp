#include "TextProviderImpl.h"

#include "TextConfig.h"
#include <vector>
#include <opencv2/opencv.hpp>

TextProviderImpl::TextProviderImpl(int count, int devId, int x, int y, const TextConfig& config) 
    : frameX(x), frameY(y), deviceId(devId), textConfig(config)
{
    cv::cuda::setDevice(deviceId);
    createNumberTemplates(count);
}

void TextProviderImpl::createNumberTemplates(int count) {
    // First step, generate our templates. We know how big we want them to be but
    // not how big the text actually should be, so we do some quick maths to do this
    for (int i = 0; i < count; ++i) {
        std::string text = std::to_string(i);

        int baseLine = 0;  // Don't use this but need to pass something to OpenCV anyway...
        cv::Size textSize = cv::getTextSize(text, textConfig.getFontFace(), 1.0, textConfig.getThickness(), &baseLine);

        // Find the side we gotta divide by the most on and use that to scale our text
        double textScale = 1.0 / std::max((double)frameY / (double)textSize.height, (double)frameX / (double)textSize.width);

        cv::Mat newImage(frameY, frameX, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        cv::putText(
            newImage,
            text,
            cv::Point(0, 0),
            textConfig.getFontFace(),
            textScale,
            textConfig.getColour(),
            textConfig.getThickness()
        );

        cv::cuda::GpuMat newImageOnGpu;
        // We block here rather than getting the CudaDevicesSingleton GPU stream.
        // If we didn't, the CPU would leave the loop and newImage would be freed :(
        newImageOnGpu.upload(newImage);
    }
}
