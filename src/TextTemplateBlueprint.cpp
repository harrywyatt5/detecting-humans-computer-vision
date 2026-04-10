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

    int x = rect.tl_x() * finalWidth;
    int y = (rect.tl_y() * finalHeight) - (float)textTemplate->rows;

    // If we're out of bounds, try appending the template to the bottom of the bounding box
    if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) {
        x = rect.br_x() * finalWidth;
        y = rect.br_y() * finalHeight;

        // If we can't put in on the bottom either, then let's try and put it in the middle of the bounding box
        if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) {
            x = ((rect.tl_x() + rect.br_x()) / 2) * finalWidth;
            y = ((rect.tl_y() + rect.br_y()) / 2) * finalHeight;

            if (x < 0 || x > finalWidth || y < 0 || y > finalHeight) std::cerr << "Label for bounding box might be out of bounds\n";
        }
    }

    return TextTemplateBlueprint(x, y, textTemplate);
}
