#pragma once

#include <opencv2/imgproc.hpp>

class TextConfig {
private:
    cv::HersheyFonts fontFace;
    cv::Scalar colour;
    int thickness;
    double fontScale;
public:
    TextConfig(cv::HersheyFonts face, cv::Scalar c, int thick, double scale) 
        : fontFace(face), colour(c), thickness(thick), fontScale(scale) {}

    cv::HersheyFonts getFontFace() const;
    cv::Scalar getColour() const;
    int getThickness() const;
    double getFontScale() const;
};
