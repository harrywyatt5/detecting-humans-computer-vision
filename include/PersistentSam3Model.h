#pragma once

#include "AbstractSession.h"
#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include "MaskDecoderSession.h"
#include "ImageProvider.h"
#include "OutputProcessor.h"
#include "Sam3Context.h"
#include <memory>
#include <vector>

class PersistentSam3Model {
private:
    bool hasGeneratedTextEncodings;
    bool hasGeneratedOutput;
    std::unique_ptr<TextEncoderSession> textEncoderSession;
    std::unique_ptr<VisionEncoderSession> visionEncoderSession;
    std::unique_ptr<MaskDecoderSession> decoder;
    std::vector<std::shared_ptr<OutputProcessor>> outputProcessors;

    void throwIfNoTextEncodings() const;
    void throwIfNoOutput() const;
public:
    PersistentSam3Model(
        std::unique_ptr<TextEncoderSession> textEncoder,
        std::unique_ptr<VisionEncoderSession> visionEncoder,
        std::unique_ptr<MaskDecoderSession> decoder
    ) : textEncoderSession(std::move(textEncoder)),
        visionEncoderSession(std::move(visionEncoder)),
        decoder(std::move(decoder)),
        hasGeneratedTextEncodings(false),
        hasGeneratedOutput(false) {}

    void mountAndCalculatePrompt(std::shared_ptr<LanguageToken> token);
    void detect(std::shared_ptr<ImageProvider> imageProvider);
    void registerOutputProcessor(std::shared_ptr<OutputProcessor> outputProcessor);
    void processOutput();

    static PersistentSam3Model createSam3Model(const Sam3Context& context);
};
