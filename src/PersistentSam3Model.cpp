#include "PersistentSam3Model.h"

#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include "PersistentImageInput.h"
#include "MaskDecoderSession.h"
#include "Sam3Context.h"
#include "AbstractSession.h"
#include "TextEncoderInitialiser.h"
#include "MaskDecoderInitialiser.h"
#include "VisionEncoderInitialiser.h"
#include "TextEncoderSessionFactory.h"
#include "VisionEncoderSessionFactory.h"
#include "MaskDecoderSessionFactory.h"
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

void PersistentSam3Model::mountAndCalculatePrompt(std::shared_ptr<LanguageToken> token) {
    TextEncoderInitialiser textInit(token);
    textInit.initialiseSession(*textEncoderSession);
    MaskDecoderInitialiser maskInit(token);
    maskInit.initialiseSession(*decoder);

    textEncoderSession->run();
    hasGeneratedTextEncodings = true;
}

void PersistentSam3Model::detect(std::shared_ptr<ImageProvider> imageProvider) {
    throwIfNoTextEncodings();

    // Load image into Vision tensor and run it
    VisionEncoderInitialiser visionInit(imageProvider);
    visionInit.initialiseSession(*visionEncoderSession);
    visionEncoderSession->run();

    // Run decoder
    decoder->run();
    hasGeneratedOutput = true;
}

void PersistentSam3Model::registerOutputProcessor(std::shared_ptr<OutputProcessor> outputProcessor) {
  outputProcessors.push_back(outputProcessor);
}

void PersistentSam3Model::processOutput() {
  throwIfNoOutput();

  if (outputProcessors.size() == 0) {
    throw std::runtime_error("No OutputProcessors were attached to this instance...");
  }

  for (auto& op : outputProcessors) {
    op->processOutput(*decoder->getPredicateMasks(), *decoder->getPredicateBoxes(), *decoder->getPredicateLogits(), *decoder->getPredicateLogic());
  }
}

void PersistentSam3Model::throwIfNoTextEncodings() const {
  if (!hasGeneratedTextEncodings) {
    throw std::runtime_error("Cannot use this function when text encodings have not been generated!");
  }
}

void PersistentSam3Model::throwIfNoOutput() const {
  if (!hasGeneratedOutput) {
    throw std::runtime_error("Run detect before processing the output!");
  }
}

PersistentSam3Model PersistentSam3Model::createSam3Model(const Sam3Context& context) {
  auto textEncoderSession = TextEncoderSessionFactory().createSession(context);
  auto visionEncoderSession = VisionEncoderSessionFactory().createSession(context);
  auto decoderSession = MaskDecoderSessionFactory().createSession(context, *textEncoderSession, *visionEncoderSession);
  return PersistentSam3Model(
    std::move(textEncoderSession),
    std::move(visionEncoderSession),
    std::move(decoderSession)
  );
}
