#pragma once

#include "AbstractSession.h"
#include "LanguageToken.h"
#include "Sam3ModelState.h"
#include "TextEncoderSession.h"
#include <opencv2/opencv.hpp>
#include <memory>

class PersistentSam3Model {
private:
    Sam3ModelState state;
    std::unique_ptr<TextEncoderSession> textEncoderSession;
    std::unique_ptr<AbstractSession> visionEncoderSession;
    std::unique_ptr<AbstractSession> decoder;
public:
    void mountAndCalculatePrompt(const LanguageToken& token);
    void warmupEngine(const cv::Mat& imageToUse, int iterations);
    cv::Mat detect();
};
