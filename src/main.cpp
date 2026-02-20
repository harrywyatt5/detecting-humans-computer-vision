#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include "TensorImage.h"
#include "LanguageTensor.h"
#include "OnnxSessionPartial.h"
#include "TensorRTProviderBuilder.h"
#include "Sam3Model.h"

int main() {
	// Initialise the session parameters
	auto imageEncoderIn = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"image"}
	);
	auto imageEncoderOut = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"}
	);
	auto languageEncoderIn = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"tokens"}
	);
	auto languageEncoderOut = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"text_attention_mask", "text_memory"}
	);
	auto decoderIn = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{
			"original_height", "original_width",
			"vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
			"language_mask", "language_features",
			"box_coords", "box_labels", "box_masks"
		}
	);
	auto decoderOut = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"boxes", "scores"}
	);
	OnnxSessionPartial imgEncoderSession("sam3-onnx/sam3_image_encoder_fp16.onnx", imageEncoderIn, imageEncoderOut);
	OnnxSessionPartial langEncoderSession("sam3-onnx/sam3_language_encoder_fp16.onnx", languageEncoderIn, languageEncoderOut);
	OnnxSessionPartial decoderSession("sam3-onnx/sam3_decoder_fp16.onnx", decoderIn, decoderOut);
	// END: Initialisation of sessions

	// Initialise the engine
	std::cout << "Intialising the engine" << std::endl;
	TensorRTProviderBuilder builder;
	Sam3Model model(imgEncoderSession, langEncoderSession, decoderSession, builder);
	std::cout << "Ready!" << std::endl;

	// Initialise data that needs to be encoded and 
	// TODO: the image reader needs to be optimised so it's single copy and also keeps the result in VRAM
	TensorImage img = TensorImage::createImageFromFile("img.jpg");
	LanguageTensor language = LanguageTensor::loadFromFile("language.token");
	auto imgTensors = model.runImageEncoder(img);
	// TODO: the language tensor should be turned into a single bind with IOBind
	// TODO: outputs should be kept on the GPU!
	auto textTensors = model.runTextEncoder(language);

	// 1. Prepare dummy box prompt inputs
    float box_coords_data[] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<int64_t> box_coords_shape = {1, 1, 4};

    int64_t box_labels_data[] = {1};
    std::vector<int64_t> box_labels_shape = {1, 1};

    bool box_masks_data[] = {true}; // Python uses True when no box prompt is provided
    std::vector<int64_t> box_masks_shape = {1, 1};

    // Replace 1008 with img.getHeight()/img.getWidth() if your TensorImage wrapper implements them
	
    int64_t orig_height = img.getOriginalHeight();
    int64_t orig_width = img.getOriginalWidth();
	std::cout << "Original Image Dimensions: " << orig_width << "x" << orig_height << std::endl;
    std::vector<int64_t> scalar_shape = {1};

	auto memoryInfo = model.getMemoryInfo_Unsafe();

	// 2. Assemble inputs in the exact order expected by decoderIn
    std::vector<Ort::Value> decoderInputs;
    // Using nullptr and 0 creates a true 0-dimensional scalar tensor
    decoderInputs.push_back(Ort::Value::CreateTensor<int64_t>(*memoryInfo, &orig_height, 1, nullptr, 0));
    decoderInputs.push_back(Ort::Value::CreateTensor<int64_t>(*memoryInfo, &orig_width, 1, nullptr, 0));	

    // Append image encoder outputs 
    for (auto& tensor : imgTensors) {
        decoderInputs.push_back(std::move(tensor));
    }

    // Append text encoder outputs
    for (auto& tensor : textTensors) {
        decoderInputs.push_back(std::move(tensor));
    }

    // Append the box prompt tensors
    decoderInputs.push_back(Ort::Value::CreateTensor<float>(*memoryInfo, box_coords_data, 4, box_coords_shape.data(), box_coords_shape.size()));
    decoderInputs.push_back(Ort::Value::CreateTensor<int64_t>(*memoryInfo, box_labels_data, 1, box_labels_shape.data(), box_labels_shape.size()));
    decoderInputs.push_back(Ort::Value::CreateTensor<bool>(*memoryInfo, box_masks_data, 1, box_masks_shape.data(), box_masks_shape.size()));

    // 3. Run the decoder (Assuming your Sam3Model has a runDecoder method matching the others)
    // If not, you may need to call something like `decoderSession.Run(std::move(decoderInputs))`
    auto decoderOutputs = model.runDecoder(std::move(decoderInputs));

	// 4. Calculate and output the number of boxes
    auto type_info = decoderOutputs[0].GetTensorTypeAndShapeInfo();
    auto shape = type_info.GetShape();
    
    std::cout << "Successfully ran the decoder!" << std::endl;
	std::cout << "Box dimensions are " << std::endl;
	for (const auto& item : shape) {
		std::cout << item << std::endl;
	}

	// TODO: refactor all of this into classes and stuff
	return 0;
}

