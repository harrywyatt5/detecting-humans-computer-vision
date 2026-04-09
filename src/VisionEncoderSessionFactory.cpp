#include "VisionEncoderSessionFactory.h"

#include "AbstractSession.h"
#include "VisionEncoderSession.h"
#include "Sam3Context.h"
#include "Binding.h"
#include "BindingBlueprint.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <cmath>

std::vector<float> VisionEncoderSessionFactory::createPositionVector(int height, int width, int channels) const {
    // Honesty this code is a little bit of black magic
    // Mostly just translated from here: https://github.com/harrywyatt5/usls/blob/main/scripts/sam3-image/export_v2.py#L13-L51
    std::vector<float> positionVector(channels * height * width, 0.0f);
    int numFeats = channels / 2;
    float temp = 1000.0f;
    float scale = 2.0f * M_PI;
    float eps = 1e-6;

    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int i = c * (height * width) + y * width + x;

                // The first half of the channels are to determine the Y position
                // and the rest are for the x position
                bool handlingY = c < numFeats;

                int relativeChannel = handlingY ? c : (c - numFeats);
                float dimensionTemp = std::pow(temp, (2.0f * (relativeChannel / 2)) / numFeats);

                // Calculate normalised value
                float embedValue;
                if (handlingY) {
                    embedValue = (y + 1) / (height + eps) * scale;
                } else {
                    embedValue = (x + 1) / (width + eps) * scale;
                }
                float normalisedValue = embedValue / dimensionTemp;

                // We apply a sine wave to even channels and cosine to odd
                positionVector[i] = relativeChannel % 2 == 0 ? std::sin(normalisedValue) : std::cos(normalisedValue);
            }
        }
    }

    return positionVector;
}

std::unique_ptr<VisionEncoderSession> VisionEncoderSessionFactory::createSession(int intermediateHeight, int intermediateWidth, const Sam3Context& samContext) const {
    std::shared_ptr<CudaTensor<float>> imageTensor = CudaTensor<float>::createCudaTensor({1, 3, intermediateHeight, intermediateWidth}, samContext);

    // divide by 3.57
    std::shared_ptr<CudaTensor<float>> fpnFeat0Tensor = CudaTensor<float>::createCudaTensor({1, 256, (intermediateHeight * 2) / 7, (intermediateWidth * 2) / 7}, samContext);
    std::shared_ptr<CudaTensor<float>> fpnFeat1Tensor = CudaTensor<float>::createCudaTensor({1, 256, intermediateHeight / 7, intermediateWidth / 7}, samContext);
    std::shared_ptr<CudaTensor<float>> fpnFeat2Tensor = CudaTensor<float>::createCudaTensor({1, 256, intermediateHeight / 14, intermediateWidth / 14}, samContext);
    std::shared_ptr<CudaTensor<float>> fpnPos2Tensor = CudaTensor<float>::createCudaTensor({1, 256, intermediateHeight / 14, intermediateWidth / 14}, samContext);

    auto session = std::make_unique<Ort::Session>(samContext.getEnvironment(), samContext.getVisionEncoderPath().c_str(), samContext.getEncoderSessionOptions());
    BindingBlueprint visionEncodingBindings;

    visionEncodingBindings.addBinding(Binding("images", imageTensor, Binding::BindingType::INPUT));
    visionEncodingBindings.addBinding(Binding("fpn_feat_0", fpnFeat0Tensor, Binding::BindingType::OUTPUT));
    visionEncodingBindings.addBinding(Binding("fpn_feat_1", fpnFeat1Tensor, Binding::BindingType::OUTPUT));
    visionEncodingBindings.addBinding(Binding("fpn_feat_2", fpnFeat2Tensor, Binding::BindingType::OUTPUT));
    // When we used the default exports, we actually had a fpn_pos_2 here. We don't generate this anymore and instead manually
    // calculate the static values on the CPU
    // visionEncodingBindings.addBinding(Binding("fpn_pos_2", fpnPos2Tensor, Binding::BindingType::OUTPUT));
    fpnPos2Tensor->copyToBuffer(createPositionVector(intermediateHeight / 14, intermediateWidth / 14, 256));

    // When the unique pointers are moved into VisionEncoderSession, they will be upgraded to shared_ptr so
    // they can be shared with other objects
    return std::make_unique<VisionEncoderSession>(
        std::move(session),
        std::move(visionEncodingBindings),
        std::move(imageTensor),
        std::move(fpnFeat0Tensor),
        std::move(fpnFeat1Tensor),
        std::move(fpnFeat2Tensor),
        std::move(fpnPos2Tensor)
    );
}
