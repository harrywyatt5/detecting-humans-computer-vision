#include "PersistentSam3Model.h"

#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include "MaskDecoderSession.h"
#include "AbstractSession.h"
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>

PersistentSam3Model::PersistentSam3Model(
    std::unique_ptr<AbstractSession> textEncoder,
    std::unique_ptr<AbstractSession> visionEncoder,
    std::unique_ptr<AbstractSession> decoder
) {
    if (TextEncoderSession* s = dynamic_cast<TextEncoderSession*>(textEncoder.get())) {
        textEncoder.release();
        textEncoderSession = std::unique_ptr<TextEncoderSession>(s);
    } else {
        throw std::runtime_error("First argument must be TextEncoderSession");
    }

    if (VisionEncoderSession* s = dynamic_cast<VisionEncoderSession*>(visionEncoder.get())) {
        visionEncoder.release();
        visionEncoderSession = std::unique_ptr<VisionEncoderSession>(s);
    } else {
        throw std::runtime_error("Second argument must be VisionEncoderSession");
    }

    if (MaskDecoderSession* s = dynamic_cast<MaskDecoderSession*>(decoder.get())) {
        decoder.release();
        decoder = std::unique_ptr<MaskDecoderSession>(s);
    } else {
        throw std::runtime_error("Third argument must be MaskDecoderSession");
    }
}

void PersistentSam3Model::mountAndCalculatePrompt(LanguageToken &token) {
  textEncoderSession->initialiseSession(token);

  // Once the token is actually in the buffer (above), we can run the session.
  // This will populate the tensors in the TextEncoderSession so when they're
  // referenced downstream they will have the correct values
  textEncoderSession->run();
}

cv::Mat PersistentSam3Model::detect(const cv::Mat& inputImage) {
    return cv::Mat();
}
