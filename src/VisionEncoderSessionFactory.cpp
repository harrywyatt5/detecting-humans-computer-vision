#include "VisionEncoderSessionFactory.h"

#include "AbstractSession.h"
#include "VisionEncoderSession.h"
#include "Sam3Context.h"
#include <onnxruntime_cxx_api.h>
#include <memory>

std::unique_ptr<AbstractSession> VisionEncoderSessionFactory::createSession(const Sam3Context& samContext) const {
    auto imageTensor = CudaTensor<float>::createCudaTensor({1, 3, 1008, 1008}, samContext);

    auto fpnFeat0Tensor = CudaTensor<float>::createCudaTensor({1, 256, 288, 288}, samContext);
    auto fpnFeat1Tensor = CudaTensor<float>::createCudaTensor({1, 256, 144, 144}, samContext);
    auto fpnFeat2Tensor = CudaTensor<float>::createCudaTensor({1, 256, 72, 72}, samContext);
    auto fpnPos2Tensor = CudaTensor<float>::createCudaTensor({1, 256, 72, 72}, samContext);

    auto session = std::make_unique<Ort::Session>(samContext.getEnvironment(), samContext.getVisionEncoderPath().c_str(), samContext.getSessionOptions());
    Ort::IoBinding visionEncodingBindings{*session};

    visionEncodingBindings.BindInput("images", imageTensor->getTensor());
    visionEncodingBindings.BindOutput("fpn_feat_0", fpnFeat0Tensor->getTensor());
    visionEncodingBindings.BindOutput("fpn_feat_1", fpnFeat1Tensor->getTensor());
    visionEncodingBindings.BindOutput("fpn_feat_2", fpnFeat2Tensor->getTensor());
    visionEncodingBindings.BindOutput("fpn_pos_2", fpnPos2Tensor->getTensor());

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
