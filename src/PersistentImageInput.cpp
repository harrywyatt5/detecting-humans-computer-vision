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
    // We will do all the work on the CPU for this test to guarantee it matches
    auto img = cv::imread(path);
    if (img.empty()) throw std::runtime_error("Failed to load image at " + path);

    // Use the exact same OpenCV function from your working code!
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(resizedX, resizedY), cv::Scalar(), true, false, CV_32F);

    // Save the blob to our class so we can write it later
    // The blob is already in NCHW format and scaled 0-1!
    this->cpuBlob = blob.clone(); 
    this->hasUploadedImage = true;
}

void PersistentImageInput::writeImageToCudaTensor(CudaTensor<float>& tensor) {
    if (!hasUploadedImage) throw std::runtime_error("No image uploaded!");

    // Check size
    size_t numBytes = 1 * 3 * resizedX * resizedY * sizeof(float);
    if (tensor.getSizeInBytes() != numBytes) throw std::runtime_error("Tensor size mismatch!");

    // Copy the perfectly formatted NCHW blob directly to the ONNX GPU Tensor
    tensor.setCudaDeviceToTensor();
    cudaMemcpy(tensor.getStartPtr(), cpuBlob.ptr<float>(), numBytes, cudaMemcpyHostToDevice);
    
    // Note: We don't need launchNormaliseImage for this test!
}

