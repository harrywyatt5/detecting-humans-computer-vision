#include "PersistentSam3Model.h"

#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>

PersistentSam3Model::PersistentSam3Model(
    std::unique_ptr<AbstractSession> textEncoder,
    std::unique_ptr<AbstractSession> visionEncoder,
    std::unique_ptr<AbstractSession> decoder
) : textEncoderSession(nullptr), visionEncoderSession(std::move(visionEncoder)), decoder(std::move(decoder)), hasGeneratedTextEncodings(false) {
    if (TextEncoderSession* s = dynamic_cast<TextEncoderSession*>(textEncoder.get())) {
        textEncoder.release();
        textEncoderSession = std::unique_ptr<TextEncoderSession>(s);
    } else {
        throw std::runtime_error("Making PersistentSam3Model failed. Passed value is not of type TextEncoderSession");
    }
}

void PersistentSam3Model::mountAndCalculatePrompt(const LanguageToken &token) {
  textEncoderSession->initialiseSession(token);

  // Once the token is actually in the buffer (above), we can run the session.
  // This will populate the tensors in the TextEncoderSession so when they're
  // referenced downstream they will have the correct values
  textEncoderSession->run();
}

cv::Mat PersistentSam3Model::detect(const cv::Mat& inputImage) {
    return cv::Mat();
}
