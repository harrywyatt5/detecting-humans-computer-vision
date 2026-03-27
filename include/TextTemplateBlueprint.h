#pragma once

#include "TextProvider.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <ByteTrack/Rect.h>
#include <cuda_runtime.h>

struct GPUTextTemplateBlueprint {
    int topLeftX;
    int topLeftY;
    cv::cuda::PtrStepSz<const uchar4> textTemplate;
};

class TextTemplateBlueprint {
private:
    int topLeftXCoord;
    int topLeftYCoord;
    std::shared_ptr<const cv::cuda::GpuMat> textTemplate; 
public:
    TextTemplateBlueprint() : topLeftXCoord(0), topLeftYCoord(0), textTemplate(nullptr) {}
    TextTemplateBlueprint(int x, int y, std::shared_ptr<const cv::cuda::GpuMat> templateText) : topLeftXCoord(x), topLeftYCoord(y), textTemplate(templateText) {}

    int getTopLeftXCoord() const;
    int getTopLeftYCoord() const;
    std::shared_ptr<const cv::cuda::GpuMat> getTemplatePtr() const;
    operator GPUTextTemplateBlueprint() const {
        return GPUTextTemplateBlueprint{topLeftXCoord, topLeftYCoord, *textTemplate};
    }

    static TextTemplateBlueprint createBlueprintFromRect(
        int id,
        const byte_track::Rect<float>& boundingBox,
        int finalWidth,
        int finalHeight,
        const TextProvider* textProvider
    );
};
