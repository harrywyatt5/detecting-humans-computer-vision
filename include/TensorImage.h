#pragma once

#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class TensorImage {
private:
	uint8_t* imageData;
	int originalHeight;
	int originalWidth;
	
	static int targetSize;
	static std::vector<int64_t> imageSize;
	static void resizeImageHelper(const cv::Mat& inputImage, cv::Mat& outputImage);
	static void reorderImageHelper(const cv::Mat& inputImage, uint8_t* outputVector);
public:
	TensorImage(uint8_t* data, int w, int h) : imageData(data), originalHeight(h), originalWidth(w) {};
	int getOriginalHeight() const { 
		return this->originalHeight; 
	};
	int getOriginalWidth() const {
		return this->originalWidth;
	};
	uint8_t* data() {
		return this->imageData;
	};
	int size() const;
	Ort::Value getInitialisedTensor(Ort::MemoryInfo& memoryInfo);
	~TensorImage();
	
	static std::vector<int64_t>& getImageDimensions() {
		return imageSize;
	};
	static std::unique_ptr<TensorImage> createImageFromFile(const std::string& fileName);
};
