#include "PersistentSam3Model.h"

#include "LanguageToken.h"
#include "TextEncoderSession.h"
#include "VisionEncoderSession.h"
#include "PersistentImageInput.h"
#include "MaskDecoderSession.h"
#include "AbstractSession.h"
#include <memory>
#include <stdexcept>
#include <opencv2/opencv.hpp>

void PersistentSam3Model::mountAndCalculatePrompt(LanguageToken &token) {
  textEncoderSession->initialiseSession(token);

  // Once the token is actually in the buffer (above), we can run the session.
  // This will populate the tensors in the TextEncoderSession so when they're
  // referenced downstream they will have the correct values
  textEncoderSession->run();
  hasGeneratedTextEncodings = true;
}

void PersistentSam3Model::detect(PersistentImageInput& image) {
    throwIfNoTextEncodings();

    // Load image into Vision tensor and run it
    visionEncoderSession->initialiseSession(image);
    visionEncoderSession->run();

    // Load decoder and find masks
    auto returnTensors = decoder->runWithResult();

// DEBUG: Optimized Best-Mask Extraction with Shape Verification
    try {
        int masksIdx = -1;
        int logitsIdx = -1;

        // 1. Dynamically find the correct indices based on tensor shapes
        for (size_t i = 0; i < returnTensors.size(); ++i) {
            auto shapeInfo = returnTensors[i].GetTensorTypeAndShapeInfo();
            auto shape = shapeInfo.GetShape(); // Returns std::vector<int64_t>
            
            // Check for pred_masks: float32[batch, 200, 288, 288]
            if (shape.size() == 4 && shape[2] == 288 && shape[3] == 288) {
                masksIdx = i;
            } 
            // Check for pred_logits: float32[batch, 200]
            else if (shape.size() == 2 && shape[1] == 200) {
                logitsIdx = i;
            }
        }

        // 2. Verify we actually found the required tensors
        if (masksIdx == -1 || logitsIdx == -1) {
            throw std::runtime_error("Validation Failed: Could not find tensors matching the expected shapes for masks or logits.");
        }

        // 3. Extract shape and data using the verified indices
        auto shapeInfo = returnTensors[masksIdx].GetTensorTypeAndShapeInfo();
        auto shape = shapeInfo.GetShape();
        
        const int numMasks = shape[1]; 
        const int maskH = shape[2];
        const int maskW = shape[3];

        const float* masksData = returnTensors[masksIdx].GetTensorData<float>();
        const float* logitsData = returnTensors[logitsIdx].GetTensorData<float>();

        // 4. Find the index of the best mask
        int bestIdx = 0;
        float maxLogit = logitsData[0];
        for (int i = 1; i < numMasks; ++i) {
            if (logitsData[i] > maxLogit) {
                maxLogit = logitsData[i];
                bestIdx = i;
            }
        }

        std::cout << "[DEBUG] Total Masks: " << numMasks << " | Best Mask Index: " << bestIdx << " | Confidence Logit: " << maxLogit << std::endl;

        // 5. Wrap the raw float data of ONLY the best mask into a cv::Mat
        // Note: const_cast is safe here because we only read from it
        float* bestMaskPtr = const_cast<float*>(masksData + (bestIdx * maskH * maskW));
        cv::Mat maskMat(maskH, maskW, CV_32F, bestMaskPtr);

        // 6. Threshold using OpenCV's fast operator (Mask values > 0 are foreground)
        cv::Mat binaryMaskSmall = maskMat > 0.0f;

        // 7. Resize back to 1920x1080
        cv::Mat finalImage;
        cv::resize(binaryMaskSmall, finalImage, cv::Size(1920, 1080), 0, 0, cv::INTER_NEAREST);

        // 8. Save to disk
        cv::imwrite("mask.jpg", finalImage);
        std::cout << "[DEBUG] Saved mask.jpg successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[DEBUG ERROR] Failed to process mask: " << e.what() << std::endl;
    }
    // END DEBUG
}

void PersistentSam3Model::throwIfNoTextEncodings() const {
  if (!hasGeneratedTextEncodings) {
    throw std::runtime_error("Cannot use this function when text encodings have not been generated!");
  }
}
