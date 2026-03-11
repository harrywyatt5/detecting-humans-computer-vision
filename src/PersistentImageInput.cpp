#include "PersistentImageInput.h"

#include "CudaTensor.h"
#include "NormaliseImageKernel.h"
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>
#include <optional>
#include <string>

PersistentImageInput::PersistentImageInput(
    int imageX,
    int imageY,
    int resizeX,
    int resizeY,
    int cudaDeviceId
) : x(imageX), y(imageY), resizedX(resizeX), resizedY(resizeY), hasUploadedImage(false) {
    cv::cuda::setDevice(cudaDeviceId);
    gpuImage = std::make_shared<cv::cuda::GpuMat>(cv::Size(imageX, imageY), CV_8UC3);
    resizedImage = cv::cuda::GpuMat(cv::Size(resizeX, resizeY), CV_8UC3);
}

void PersistentImageInput::uploadImageFromDisk(const std::string& path) {
    auto img = cv::imread(path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image at " + path);
    }
    if (img.cols != x || img.rows != y) {
        throw std::runtime_error("Image does not match size allocated to this object!");
    }

    cv::Mat convertedImg;
    cv::cvtColor(img, convertedImg, cv::COLOR_BGR2RGB);
    gpuImage->upload(convertedImg, stream);

    // Resize on the gpu as it can be parallelised
    cv::cuda::resize(*gpuImage, resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR, stream);
    hasUploadedImage = true;
}

void PersistentImageInput::uploadImageFromSensorMsg(const sensor_msgs::msg::Image& image, const std::optional<cv::ColorConversionCodes> conversion) {
    // Ensure the message has the same size as we're expecting
    if (image.height != y || image.width != x) {
        throw std::runtime_error("Image does not match size allocated to this object!");
    }
    
    const cv::Mat cpuImage(
        image.height,
        image.width,
        CV_8UC3,
        const_cast<uint8_t*>(image.data.data()),
        image.step
    );

    // If the colour format needs to be converted, it shouldn't do though!
    if (conversion.has_value()) {
        // Super expensive, when we are looking at the incoming images we should definitely warn if this is
        // the case!
        cv::cuda::GpuMat temp;
        temp.upload(cpuImage, stream);
        cv::cuda::cvtColor(temp, *gpuImage, conversion.value(), 0, stream);
    } else {
        gpuImage->upload(cpuImage, stream);
    }

    cv::cuda::resize(*gpuImage, resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR, stream);
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
    launchNormaliseImage(resizedImage, stream, tensor.getStartPtr());

    stream.waitForCompletion();
    // Throw if the kernel crashes for some reason
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Could not process image: ") + cudaGetErrorString(err));
    }
}

std::shared_ptr<cv::cuda::GpuMat> PersistentImageInput::getMutableGpuImage() {
    return gpuImage;
}

int PersistentImageInput::getOriginalX() const {
    return x;
}

int PersistentImageInput::getOriginalY() const {
    return y;
}
