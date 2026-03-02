#pragma once

#include "AbstractSession.h"
#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include <opencv2/opencv.hpp>
#include <memory>

class PersistentSam3Model {
private:
    bool hasGeneratedTextEncodings;
    std::unique_ptr<TextEncoderSession> textEncoderSession;
    std::unique_ptr<AbstractSession> visionEncoderSession;
    std::unique_ptr<AbstractSession> decoder;
public:
    PersistentSam3Model(std::unique_ptr<AbstractSession> textEncoder, std::unique_ptr<AbstractSession> visionEncoder, std::unique_ptr<AbstractSession> decoder);
    void mountAndCalculatePrompt(const LanguageToken& token);
    cv::Mat detect(const cv::Mat& inputImage);
};
