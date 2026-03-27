#include "PersistentImageInput.h"

#include "CudaTensor.h"
#include "NormaliseImageKernel.h"
#include "CudaDevicesSingleton.h"
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_view.hpp>
#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <optional>
#include <iostream>
#include <string>

PersistentImageInput::PersistentImageInput(
    int imageX,
    int imageY,
    int resizeX,
    int resizeY,
    int cudaDeviceId
) : x(imageX), y(imageY), resizedX(resizeX), resizedY(resizeY), pinnedStaticPtr(nullptr), hasUploadedImage(false) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(cudaDeviceId);
    gpuImage = std::make_shared<GpuImage>(cv::cuda::GpuMat(cv::Size(imageX, imageY), CV_8UC3), cudaDeviceId);
    resizedImage = cv::cuda::GpuMat(cv::Size(resizeX, resizeY), CV_8UC3);

    allocatePinnedMem();
}

void PersistentImageInput::allocatePinnedMem() {
    auto result = cudaMallocHost((void**)&pinnedStaticPtr, x * y * 3 * sizeof(uint8_t));

    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to allocate pinned host memory. Reason: ") + cudaGetErrorString(result));
    }
}

void PersistentImageInput::uploadImageFromDisk(const std::string& path) {
    // TODO: optimise this code so that it uses malloc'd data -> pinned static -> gpu
    auto img = cv::imread(path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image at " + path);
    }
    if (img.cols != x || img.rows != y) {
        throw std::runtime_error("Image does not match size allocated to this object!");
    }

    cv::Mat convertedImg;
    cv::cvtColor(img, convertedImg, cv::COLOR_BGR2RGB);
    gpuImage->uploadCpuImage(convertedImg);

    // Resize on the gpu as it can be parallelised
    cv::cuda::resize(gpuImage->getConstGpuMat(), resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR, cudaDevice->getOpenCVCudaStream());
    hasUploadedImage = true;
}

void PersistentImageInput::copyToPinnedMemory(const uint8_t* source, int step) {
    for (int row = 0; row < y; ++row) {
        const uint8_t* srcRowPtr = source + row * step;
        uint8_t* destination = pinnedStaticPtr + row * (x * 3);
        std::memcpy(destination, srcRowPtr, x * 3 * sizeof(uint8_t));
    }
}

void PersistentImageInput::uploadImageFromSensorMsg(const sensor_msgs::msg::Image& image, const std::optional<cv::ColorConversionCodes> conversion) {
    // Ensure the message has the same size as we're expecting
    if (image.height != (unsigned int)y || image.width != (unsigned int)x) {
        throw std::runtime_error("Image does not match size allocated to this object!");
    }

    copyToPinnedMemory(image.data.data(), image.step);
    
    const cv::Mat cpuImage(
        y,
        x,
        CV_8UC3,
        pinnedStaticPtr,
        x * 3
    );

    // If the colour format needs to be converted, it shouldn't do though!
    if (conversion.has_value()) {
        // Super expensive, when we are looking at the incoming images we should definitely warn if this is
        // the case!
        cv::cuda::GpuMat temp;
        temp.upload(cpuImage, cudaDevice->getOpenCVCudaStream());
        cv::cuda::cvtColor(temp, gpuImage->getMutableGpuMat(), conversion.value(), 0, cudaDevice->getOpenCVCudaStream());
    } else {
        gpuImage->uploadCpuImage(cpuImage);
    }

    cv::cuda::resize(gpuImage->getConstGpuMat(), resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR, cudaDevice->getOpenCVCudaStream());
    hasUploadedImage = true;
}

void PersistentImageInput::copyImageFromNitros(const nitros::NitrosImageView& image, const std::optional<cv::ColorConversionCodes> conversion) {
    // Ensure the message has the same size as we're expecting
    if (image.GetHeight() != (unsigned int)y || image.GetWidth() != (unsigned int)x) {
        throw std::runtime_error("Image does not match size allocated to this object!");
    }

    const cv::cuda::GpuMat incomingImg(
        y,
        x,
        CV_8UC3,
        const_cast<unsigned char*>(image.GetGpuData()),
        image.GetStride()
    );

    // Force colour conversion if necessary
    if (conversion.has_value()) {
        cv::cuda::cvtColor(incomingImg, gpuImage->getMutableGpuMat(), conversion.value(), 0, cudaDevice->getOpenCVCudaStream());
    } else {
        gpuImage->copyFrom(incomingImg);
    }

    cv::cuda::resize(gpuImage->getConstGpuMat(), resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR, cudaDevice->getOpenCVCudaStream());
    hasUploadedImage = true;
}

void PersistentImageInput::writeImageToCudaTensor(CudaTensor<float>& tensor) {
    if (!hasUploadedImage) {
        throw std::runtime_error("Cannot write image to tensor. An image has not been uploaded yet!");
    }

    auto tensorShape = tensor.getTensorShape();
    if (tensorShape.size() != 4 || tensorShape[0] != 1 || tensorShape[1] != 3 || tensorShape[2] != resizedY || tensorShape[3] != resizedX) {
        throw std::runtime_error("Tensor is not the correct shape to insert an image into. Aborting...");
    }

    tensor.setCudaDeviceToTensor();
    launchNormaliseImage(resizedImage, cudaDevice->getOpenCVCudaStream(), tensor.getStartPtr());
}

std::shared_ptr<GpuImage> PersistentImageInput::getMutableGpuImage() {
    return gpuImage;
}

std::shared_ptr<const GpuImage> PersistentImageInput::getConstGpuImage() const {
    return gpuImage;
}

int PersistentImageInput::getOriginalX() const {
    return x;
}

int PersistentImageInput::getOriginalY() const {
    return y;
}

PersistentImageInput::~PersistentImageInput() {
    if (pinnedStaticPtr == nullptr) {
        return;
    }

    auto result = cudaFreeHost((void*)pinnedStaticPtr);
    if (result != cudaSuccess) {
        std::cerr << "An error occurred when trying to free pinned memory. Reason: " << cudaGetErrorString(result) << std::endl;
    }

    pinnedStaticPtr = nullptr;
}

PersistentImageInput::PersistentImageInput(PersistentImageInput&& other) noexcept
    : x(other.x), y(other.y), resizedX(other.resizedX), resizedY(other.resizedY),
      gpuImage(std::move(other.gpuImage)),
      resizedImage(std::move(other.resizedImage)),
      cudaDevice(std::move(other.cudaDevice)),
      pinnedStaticPtr(other.pinnedStaticPtr),
      hasUploadedImage(other.hasUploadedImage)
{
    other.pinnedStaticPtr = nullptr;
}
