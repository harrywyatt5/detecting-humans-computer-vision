#pragma once

#include "AbstractSession.h"
#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include "MaskDecoderSession.h"
#include "ImageProvider.h"
#include <opencv2/opencv.hpp>
#include <memory>

class PersistentSam3Model {
private:
    bool hasGeneratedTextEncodings;
    std::unique_ptr<TextEncoderSession> textEncoderSession;
    std::unique_ptr<VisionEncoderSession> visionEncoderSession;
    std::unique_ptr<MaskDecoderSession> decoder;

    void throwIfNoTextEncodings() const;
public:
    PersistentSam3Model(
        std::unique_ptr<TextEncoderSession> textEncoder,
        std::unique_ptr<VisionEncoderSession> visionEncoder,
        std::unique_ptr<MaskDecoderSession> decoder
    ) : textEncoderSession(std::move(textEncoder)), visionEncoderSession(std::move(visionEncoder)), decoder(std::move(decoder)) {}

    void mountAndCalculatePrompt(std::shared_ptr<LanguageToken> token);
    void detect(std::shared_ptr<ImageProvider> imageProvider);
};
