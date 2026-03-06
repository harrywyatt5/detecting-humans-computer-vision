#include "PersistentImageInput.h"
#include "CudaTensor.h"
#include "NormaliseImageKernel.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
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
    if (img.empty()) throw std::runtime_error("Failed to load image at " + path);

    cv::Mat convertedImg;
    cv::cvtColor(img, convertedImg, cv::COLOR_BGR2RGB);
    gpuImage.upload(convertedImg, stream);

    // Resize on the gpu as it can be parallelised
    cv::cuda::resize(gpuImage, resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR, stream);
    hasUploadedImage = true;
}

void PersistentImageInput::writeImageToCudaTensor(CudaTensor<float>& tensor) {
    auto tensorShape = tensor.getTensorShape();
    if (tensorShape.size() != 4 || tensorShape[0] != 1 || tensorShape[1] != 3 || tensorShape[2] != resizedX || tensorShape[3] != resizedY) {
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