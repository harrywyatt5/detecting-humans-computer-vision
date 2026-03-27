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
    int finalWidth,
    int finalHeight,
    const TextProvider* textProvider
) {
    auto textTemplate = textProvider->getTextForNumber(id);

    int x = (rect.tl_x() * finalWidth) + ((float)textTemplate->cols / 2.0f);
    int y = (rect.tl_y() * finalHeight) - (float)textTemplate->rows;

    // If we're out of bounds, try appending the template to the bottom of the bounding box
    if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) {
        x = rect.br_x() * finalWidth + (float)textTemplate->cols / 2.0f;
        y = rect.br_y() * finalHeight;

        if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) std::cerr << "Label for bounding box might be out of bounds\n";
    }

    return TextTemplateBlueprint(x, y, textTemplate);
}
