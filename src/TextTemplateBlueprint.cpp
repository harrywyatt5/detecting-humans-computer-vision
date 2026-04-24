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

    int textW = textTemplate->cols;
    int textH = textTemplate->rows;

    int x = rect.tl_x() * finalWidth;
    int y = (rect.tl_y() * finalHeight) - textH;

    // If we're out of bounds (accounting for the text's width/height), try appending the template to the bottom of the bounding box
    if (x < 0 || x + textW > finalWidth || y < 0 || y + textH > finalHeight) {
        // Aligned to the left, appended to the bottom
        x = rect.tl_x() * finalWidth; 
        y = rect.br_y() * finalHeight;

        // If we can't put it on the bottom either, then let's try and put it in the middle of the bounding box
        if (x < 0 || x + textW > finalWidth || y < 0 || y + textH > finalHeight) {
            x = ((rect.tl_x() + rect.br_x()) / 2) * finalWidth - (textW / 2);
            y = ((rect.tl_y() + rect.br_y()) / 2) * finalHeight - (textH / 2);

            if (x < 0 || x + textW > finalWidth || y < 0 || y + textH > finalHeight) {
                std::cerr << "Label for bounding box out of bounds. Clamping to frame.\n";
                // Failsafe: Force the label to stay within the frame boundaries
                x = std::max(0, std::min(x, finalWidth - textW));
                y = std::max(0, std::min(y, finalHeight - textH));
            }
        }
    }

    return TextTemplateBlueprint(x, y, textTemplate);
}
