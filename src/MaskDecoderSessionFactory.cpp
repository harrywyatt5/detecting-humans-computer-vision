#include "MaskDecoderSessionFactory.h"

#include "MaskDecoderSession.h"
#include "CudaTensor.h"
#include "Sam3Context.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include <memory>
#include <vector>
#include <cstdint>

std::unique_ptr<MaskDecoderSession> MaskDecoderSessionFactory::createSession(const Sam3Context& samContext, TextEncoderSession& textEncoder, VisionEncoderSession& visionEncoder) const {
    // Inputs
    auto fpnFeat0 = visionEncoder.getFpnFeat0Tensor();
    auto fpnFeat1 = visionEncoder.getFpnFeat1Tensor();
    auto fpnFeat2 = visionEncoder.getFpnFeat2Tensor();
    auto fpnPos2 = visionEncoder.getFpnPos2Tensor();

    auto textFeatures = textEncoder.getTextFeaturesTensor();
    auto textMasks = textEncoder.getTextMaskTensor();

    // We don't want to pass bounding boxes to track, so we block out these values
    auto inputBoxes = CudaTensor<float>::createCudaTensor({1, 1, 4}, samContext);
    inputBoxes->copyToBuffer(std::vector<float>(4, 0.0f));
    auto inputBoxLabels = CudaTensor<int64_t>::createCudaTensor({1, 1}, samContext);
    inputBoxLabels->copyToBuffer(std::vector<int64_t>(1, 0));

    
}
