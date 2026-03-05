#include "TextEncoderSession.h"

#include "LanguageToken.h"
#include "CudaTensor.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>

void TextEncoderSession::initialiseSession(LanguageToken& token) {
    // This fills in the int64_t to find our target
    token.populateTensorsWithToken(*inputIdsTensor, *attentionMaskTensor);
    this->isInitialised = true;
}

void TextEncoderSession::run() {
    thowIfNotInitialised();
    this->session->Run(Ort::RunOptions{nullptr}, bindings);

    // 2. TRT Boolean Orphan Bypass:
    // Read the input IDs we just sent to the network
    std::vector<int64_t> idsCPU(32, 0);
    cudaMemcpy(idsCPU.data(), inputIdsTensor->getStartPtr(), 32 * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // 3. Rebuild the correct boolean mask on the CPU
    std::vector<uint8_t> manualMask(32, 0);
    std::cout << "Forcing Text Mask: [ ";
    for (int i = 0; i < 32; ++i) {
        // If it's a real token (not 0), set mask to 1
        manualMask[i] = (idsCPU[i] != 0) ? 1 : 0;
        std::cout << static_cast<int>(manualMask[i]) << " ";
    }
    std::cout << "]" << std::endl;

    // 4. Force this perfect mask straight into the Decoder's input pointer!
    cudaMemcpy(textMaskTensor->getStartPtr(), manualMask.data(), 32 * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

std::vector<Ort::Value> TextEncoderSession::runWithResult() {
    run();
    return bindings.GetOutputValues();
}

std::shared_ptr<CudaTensor<float>> TextEncoderSession::getTextFeaturesTensor() {
    return textFeaturesTensor;
}

std::shared_ptr<CudaTensor<uint8_t>> TextEncoderSession::getTextMaskTensor() {
    return textMaskTensor;
}
