#include "TextConfig.h"

cv::HersheyFonts TextConfig::getFontFace() const {
    return fontFace;
}

cv::Scalar TextConfig::getColour() const {
    return colour;
}

int TextConfig::getThickness() const {
    return thickness;
}

double TextConfig::getFontScale() const {
    return fontScale;
}
