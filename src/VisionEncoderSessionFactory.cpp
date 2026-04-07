#include "VisionEncoderSessionFactory.h"

#include "AbstractSession.h"
#include "VisionEncoderSession.h"
#include "Sam3Context.h"
#include "Binding.h"
#include "BindingBlueprint.h"
#include <onnxruntime_cxx_api.h>
#include <memory>

std::unique_ptr<VisionEncoderSession> VisionEncoderSessionFactory::createSession(const Sam3Context& samContext) const {
    std::shared_ptr<CudaTensor<float>> imageTensor = CudaTensor<float>::createCudaTensor({1, 3, 1008, 1008}, samContext);

    std::shared_ptr<CudaTensor<float>> fpnFeat0Tensor = CudaTensor<float>::createCudaTensor({1, 256, 288, 288}, samContext);
    std::shared_ptr<CudaTensor<float>> fpnFeat1Tensor = CudaTensor<float>::createCudaTensor({1, 256, 144, 144}, samContext);
    std::shared_ptr<CudaTensor<float>> fpnFeat2Tensor = CudaTensor<float>::createCudaTensor({1, 256, 72, 72}, samContext);
    std::shared_ptr<CudaTensor<float>> fpnPos2Tensor = CudaTensor<float>::createCudaTensor({1, 256, 72, 72}, samContext);

    auto session = std::make_unique<Ort::Session>(samContext.getEnvironment(), samContext.getVisionEncoderPath().c_str(), samContext.getEncoderSessionOptions());
    BindingBlueprint visionEncodingBindings;

    visionEncodingBindings.addBinding(Binding("images", imageTensor, Binding::BindingType::INPUT));
    visionEncodingBindings.addBinding(Binding("fpn_feat_0", fpnFeat0Tensor, Binding::BindingType::OUTPUT));
    visionEncodingBindings.addBinding(Binding("fpn_feat_1", fpnFeat1Tensor, Binding::BindingType::OUTPUT));
    visionEncodingBindings.addBinding(Binding("fpn_feat_2", fpnFeat2Tensor, Binding::BindingType::OUTPUT));
    visionEncodingBindings.addBinding(Binding("fpn_pos_2", fpnPos2Tensor, Binding::BindingType::OUTPUT));

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
