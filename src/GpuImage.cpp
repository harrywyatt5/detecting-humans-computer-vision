#include "GpuImage.h"

#include "CudaDevicesSingleton.h"
#include "CudaDevice.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <isaac_ros_nitros_image_type/nitros_image.hpp>
#include <isaac_ros_nitros_image_type/nitros_image_builder.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp/rclcpp.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <iostream>

GpuImage::GpuImage(cv::cuda::GpuMat mat, int devId) : gpuBuffer(nullptr), internalData(std::move(mat)) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
}

GpuImage::GpuImage(int width, int height, int devId) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
    cudaDevice->switchCudaDevice();

    auto allocError = cudaMalloc((void**)&gpuBuffer, 3 * width * height * sizeof(uint8_t));
    if (allocError != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to allocate to CUDA device. Reason: ") + cudaGetErrorString(allocError));
    }

    auto memsetError = cudaMemset((void*)&gpuBuffer, 0, 3 * width * height * sizeof(uint8_t));
    if (memsetError != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to memset GpuImage on CUDA. Reason: ") + cudaGetErrorString(memsetError));
    }
    
    internalData = cv::cuda::GpuMat(
        height,
        width,
        CV_8UC3,
        (void*)gpuBuffer,
        width * 3
    );
}

void GpuImage::uploadCpuImage(const cv::Mat& cpuImage) {
    throwIfDimensionMismatch(cpuImage.cols, cpuImage.rows);
    cudaDevice->switchCudaDevice();
    internalData.upload(cpuImage, cudaDevice->getOpenCVCudaStream());
}

void GpuImage::copyFrom(const cv::cuda::GpuMat& gpuImage) {
    throwIfDimensionMismatch(gpuImage.cols, gpuImage.rows);
    cudaDevice->switchCudaDevice();
    gpuImage.copyTo(internalData, cudaDevice->getOpenCVCudaStream());
}

void GpuImage::toNewColourTarget(cv::ColorConversionCodes code) {
    cudaDevice->switchCudaDevice();
    // It's super easy if we are using a GpuMat managed buffer. If we aren't, it's a bit harder unfortunately
    // but we essentially replicate the logic of the epic swap command
    if (gpuBuffer == nullptr) {
        cv::cuda::GpuMat tempMat;
        cv::cuda::cvtColor(internalData, tempMat, code, 0, cudaDevice->getOpenCVCudaStream());

        internalData.swap(tempMat);
    } else {
        uint8_t* newBuffer;
        auto allocError = cudaMallocAsync((void**)&newBuffer, 3 * internalData.cols * internalData.rows * sizeof(uint8_t), cudaDevice->getCudaStream());
        
        if (allocError != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to allocate CUDA bytes to perform colour conversion. Reason: ") + cudaGetErrorString(allocError));
        }

        auto tempMat = cv::cuda::GpuMat(
            internalData.rows,
            internalData.cols,
            CV_8UC3,
            newBuffer,
            internalData.cols * 3
        );
        cv::cuda::cvtColor(internalData, tempMat, code, 0, cudaDevice->getOpenCVCudaStream());

        // Free memory in this object before replacing the pointer with our new buffer
        freeMemory();
        gpuBuffer = newBuffer;
        internalData = std::move(tempMat);
    }
}

void GpuImage::download(cv::Mat& target) const {
    cudaDevice->switchCudaDevice();
    internalData.download(target, cudaDevice->getOpenCVCudaStream());
}

int GpuImage::getHeight() const {
    return internalData.rows;
}

int GpuImage::getWidth() const {
    return internalData.cols;
}

const cv::cuda::GpuMat& GpuImage::getConstGpuMat() const {
    return internalData;
}

cv::cuda::GpuMat& GpuImage::getMutableGpuMat() {
    return internalData;
}

std::unique_ptr<sensor_msgs::msg::Image> GpuImage::createRos2ImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) const {
    cudaDevice->switchCudaDevice();
    auto msg = std::make_unique<sensor_msgs::msg::Image>();

    msg->header.stamp = broadcastTime;
    msg->header.frame_id = frameName;
    msg->height = internalData.rows;
    msg->width = internalData.cols;
    msg->step = internalData.cols * 3; // Each pixel has three values. Don't have to care about padding because download strips it
    msg->encoding = sensor_msgs::image_encodings::RGB8;

    msg->data.resize(msg->height * msg->step);

    cv::Mat cpuImage(
        internalData.rows,
        internalData.cols,
        CV_8UC3,
        msg->data.data()
    );
    internalData.download(cpuImage, cudaDevice->getOpenCVCudaStream());

    return msg;
}

nitros::NitrosImage GpuImage::createNitrosImageMessage(const std::string& frameName, rclcpp::Time broadcastTime) const {
    cudaDevice->switchCudaDevice();

    std_msgs::msg::Header header;
    header.stamp = broadcastTime;
    header.frame_id = frameName;

    uint32_t stepWithoutPadding = internalData.cols * 3;

    uint8_t* gpuOutput;
    auto result = cudaMalloc((void**)&gpuOutput, internalData.rows * stepWithoutPadding * sizeof(uint8_t));
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Could not create buffer for final message. Reason: ") + cudaGetErrorString(result));
    }

    cv::cuda::GpuMat tempWrapper(
        internalData.rows,
        internalData.cols,
        CV_8UC3,
        gpuOutput,
        stepWithoutPadding
    );
    internalData.copyTo(tempWrapper, cudaDevice->getOpenCVCudaStream());
    // We might create the nitros image and send it out to subscribers before we've finished writing to it
    cudaDevice->waitForCompletion();

    return nitros::NitrosImageBuilder()
            .WithDimensions(internalData.rows, internalData.cols)
            .WithEncoding(sensor_msgs::image_encodings::RGB8)
            .WithHeader(header)
            .WithGpuData(gpuOutput)
            .Build();
}

void GpuImage::throwIfDimensionMismatch(int cols, int rows) const {
    if (internalData.cols != cols || internalData.rows != rows) {
        throw std::runtime_error("Failed to perform operation to GpuImage. Input image must have the same width and height as the target GpuImage");
    }
}

void GpuImage::freeMemory() {
    if (gpuBuffer != nullptr) {
        auto result = cudaFreeAsync(gpuBuffer, cudaDevice->getCudaStream());
        gpuBuffer = nullptr;

        if (result != cudaSuccess) {
            std::cerr << "Could not free CUDA memory in GpuImage. The application may be leaking memory. Reason: " << cudaGetErrorString(result) << "\n";
        }
    }
}

GpuImage::GpuImage(GpuImage&& other) noexcept : gpuBuffer(other.gpuBuffer), internalData(std::move(other.internalData)), cudaDevice(other.cudaDevice) {
    other.gpuBuffer = nullptr;
    other.cudaDevice = nullptr;
}

GpuImage& GpuImage::operator=(GpuImage&& other) noexcept   {
    // Avoid issues if we are moving the object to itself
    if (this == &other) {
        return *this;
    }

    // Free memory will set the pointer to nullptr, so we have to capture it before
    uint8_t* tempBuffer = other.gpuBuffer;
    other.freeMemory();
    gpuBuffer = tempBuffer;
    internalData = std::move(other.internalData);
    cudaDevice = other.cudaDevice;

    other.cudaDevice = nullptr;

    return *this;
}

GpuImage::~GpuImage() {
    freeMemory();
}
