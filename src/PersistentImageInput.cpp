#include "PersistentImageInput.h"

#include "CudaTensor.h"
#include "NormaliseImageKernel.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <stdexcept>
#include <string>

PersistentImageInput::PersistentImageInput(
    int imageX,
    int imageY,
    int resizeX,
    int resizeY,
    int cudaDeviceId
) : x(imageX), y(imageY), resizedX(resizeX), resizedY(resizeY), hasUploadedImage(false) {
    cv::cuda::setDevice(cudaDeviceId);
    gpuImage = cv::cuda::GpuMat(cv::Size(imageX, imageY), CV_8UC3);
    resizedImage = cv::cuda::GpuMat(cv::Size(resizeX, resizeY), CV_8UC3);
}

void PersistentImageInput::uploadImageFromDisk(const std::string& path) {
    auto img = cv::imread(path);
    cv::Mat convertedImg;
    cv::cvtColor(img, convertedImg, cv::COLOR_BGR2RGB);
    gpuImage.upload(convertedImg);

    // Resize on the gpu as it can be parallelised
    cv::cuda::resize(gpuImage, resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR);
    hasUploadedImage = true;
}

void PersistentImageInput::writeImageToTensor(CudaTensor<float>& tensor) {
    if (!hasUploadedImage) {
        throw std::runtime_error("PersistentImageInput has not been written to yet!");
    }

    auto tensorShape = tensor.getTensorShape();
    if (tensorShape.size() != 4 || tensorShape[0] != 1 || tensorShape[1] != 3 || tensorShape[2] != resizedX || tensorShape[3] != resizedY) {
        throw std::runtime_error("Tensor is not the correct shape to insert an image into. Aborting...");
    }

    tensor.setCudaDeviceToTensor();
    launchNormaliseImage(resizedImage, stream, tensor.getStartPtr());

    stream.waitForCompletion();
}
