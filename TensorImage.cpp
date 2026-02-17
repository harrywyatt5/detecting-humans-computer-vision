#include "TensorImage.h"

int TensorImage::targetSize = 1008;
std::vector<int64_t> TensorImage::imageSize = {3, 1008, 1008};

std::unique_ptr<TensorImage> TensorImage::createImageFromFile(const std::string& name) {
	// The image needs to be read into memory and scaled to 1008x1008. We then
	// need to reorder the bytes so all red bytes are ordered first, then green, then blue
	cv::Mat image = cv::imread(name);
	cv::Mat resizedImage;
	int width = image.size().width;
	int height = image.size().height;

	resizeImageHelper(image, resizedImage);
	
	uint8_t* imageData = new uint8_t[TensorImage::targetSize * TensorImage::targetSize * 3];
	TensorImage::reorderImageHelper(resizedImage, imageData);
	return std::make_unique<TensorImage>(imageData, width, height);
}

void TensorImage::resizeImageHelper(const cv::Mat& inputImage, cv::Mat& outputImage) {
	// We have to resize the image
	auto imgSize = inputImage.size();
	float largestSide = imgSize.width >= imgSize.height ? (float)imgSize.width : (float)imgSize.height;	
	float resizePercentage = (float)TensorImage::targetSize / largestSide;
	
	int newHeight = imgSize.height * resizePercentage;
	int newWidth = imgSize.width * resizePercentage;
	int padOnHeight = TensorImage::targetSize - newHeight;
	int padOnWidth = TensorImage::targetSize - newWidth;

	cv::Mat resizedIntermediate;
	cv::resize(inputImage, resizedIntermediate, cv::Size(newWidth, newHeight));

	cv::copyMakeBorder(resizedIntermediate, outputImage, 0, padOnHeight, 0, padOnWidth, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

int TensorImage::size() const {
	return TensorImage::targetSize * TensorImage::targetSize * 3;
}

void TensorImage::reorderImageHelper(const cv::Mat& inputImage, uint8_t* outputVector) {	
	std::vector<cv::Mat> colourChannels(3);
	cv::split(inputImage, colourChannels);
	int flatSize = TensorImage::targetSize * TensorImage::targetSize;

	std::memcpy(outputVector, colourChannels[2].data, flatSize);
	std::memcpy(outputVector + flatSize, colourChannels[1].data, flatSize);
	std::memcpy(outputVector + (flatSize * 2), colourChannels[0].data, flatSize);
}

Ort::Value TensorImage::getInitialisedTensor(Ort::MemoryInfo& memoryInfo) {
	return Ort::Value::CreateTensor(memoryInfo, this->data(), this->size(), getImageDimensions().data(), getImageDimensions().size()); 
}

TensorImage::~TensorImage() {
	delete[] this->imageData; 
}

