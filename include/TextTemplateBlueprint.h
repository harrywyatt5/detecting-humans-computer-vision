#pragma once

#include <memory>
#include <opencv2/opencv.hpp>

class TextTemplateBlueprint {
private:
    int topLeftXCoord;
    int topLeftYCoord;
    std::shared_ptr<const cv::cuda::GpuMat> textTemplate; 
public:
    TextTemplateBlueprint(int x, int y, std::shared_ptr<const cv::cuda::GpuMat> templateText);
};

// Details: We will wanna get the top left coordinate for a frame and the new label
// We will probably just grab both these pieces of information from our Strack list in the processor
