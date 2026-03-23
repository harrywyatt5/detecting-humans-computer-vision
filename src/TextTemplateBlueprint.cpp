#include "TextTemplateBlueprint.h"

#include "TextProvider.h"
#include <memory>
#include <opencv2/opencv.hpp>
#include <ByteTrack/Rect.h>
#include <iostream>

int TextTemplateBlueprint::getTopLeftXCoord() const {
    return topLeftXCoord;
}

int TextTemplateBlueprint::getTopLeftYCoord() const {
    return topLeftYCoord;
}

std::shared_ptr<const cv::cuda::GpuMat> TextTemplateBlueprint::getTemplatePtr() const {
    return textTemplate;
}

TextTemplateBlueprint TextTemplateBlueprint::createBlueprintFromRect(
    int id,
    const byte_track::Rect<float>& rect,
    int intermediateWidth,
    int intermediateHeight,
    int finalWidth,
    int finalHeight,
    const TextProvider* textProvider
) {
    float xScale = (float)finalWidth / (float)intermediateWidth;
    float yScale = (float)finalHeight / (float)intermediateHeight;

    auto textTemplate = textProvider->getTextForNumber(id);

    int x = (rect.tl_x() * xScale) - (float)textTemplate->cols;
    int y = (rect.tl_y() * yScale) - (float)textTemplate->rows;

    // If we're out of bounds, try appending the template to the end of the bounding box
    if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) {
        x = rect.br_x() * xScale;
        y = rect.br_y() * yScale;

        if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) std::cerr << "Label for bounding box might be out of bounds\n";
    }

    return TextTemplateBlueprint(x, y, textTemplate);
}
