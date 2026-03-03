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
    gpuImage.upload(img);

    // Resize on the gpu as it can be parallelised
    cv::cuda::resize(gpuImage, resizedImage, cv::Size(resizedX, resizedY), 0, 0, cv::INTER_LINEAR);
}

void PersistentImageInput::writeImageToTensor(CudaTensor<float>& tensor) {
    auto tensorShape = tensor.getTensorShape();
    if (tensorShape.size() != 3 || tensorShape[0] != 3 || tensorShape[1] != resizedX || tensorShape[2] != resizedY) {
        throw std::runtime_error("Tensor is not the correct shape to insert an image into. Aborting...");
    }

    tensor.setCudaDeviceToTensor();
    launchNormaliseImage(resizedImage, stream, tensor.getStartPtr());

    stream.waitForCompletion();
}
