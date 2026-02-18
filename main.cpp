#include <memory>
#include <vector>
#include <string>

#include "TensorImage.h"
#include "OnnxSessionPartial.h"
#include "TensorRTProviderBuilder.h"
#include "Sam3Model.h"

int main() {
	// Initialise the session parameters
	auto imageEncoderIn = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"image"}
	);
	auto imageEncoderOut = std::make_shared<std::vector<std::string>>(
		std::initializer_list<std::string>{"vision_poc_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"}
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
	OnnxSessionPartial imgEncoderSession("sam3-onnx/sam3_image_encoder.onnx", imageEncoderIn, imageEncoderOut);
	OnnxSessionPartial langEncoderSession("sam3-onnx/sam3_language_encoder.onnx", languageEncoderIn, languageEncoderOut);
	OnnxSessionPartial decoderSession("sam3-onnx/sam3_decoder.onnx", decoderIn, decoderOut);

	// END: Initialisation of sessions

	std::unique_ptr<TensorImage> t_img = TensorImage::createImageFromFile("img.jpg");
	TensorRTProviderBuilder builder;
	
	Sam3Model model(imgEncoderSession, langEncoderSession, decoderSession, builder);
	return 0;
}

