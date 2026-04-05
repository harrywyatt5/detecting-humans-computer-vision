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

GpuImage::GpuImage(cv::cuda::GpuMat mat, int devId) : internalData(std::move(mat)) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
}

GpuImage::GpuImage(int x, int y, int devId) {
    cudaDevice = CudaDevicesSingleton::getInstance()->getForId(devId);
    internalData = cv::cuda::GpuMat(cv::Size(x, y), CV_8UC3);
}

void GpuImage::uploadCpuImage(const cv::Mat& cpuImage) {
    cudaDevice->switchCudaDevice();
    internalData.upload(cpuImage, cudaDevice->getOpenCVCudaStream());
}

void GpuImage::copyFrom(const cv::cuda::GpuMat& gpuImage) {
    cudaDevice->switchCudaDevice();
    gpuImage.copyTo(internalData, cudaDevice->getOpenCVCudaStream());
}

void GpuImage::toNewColourTarget(cv::ColorConversionCodes code) {
    cudaDevice->switchCudaDevice();

    cv::cuda::GpuMat tempMat;
    cv::cuda::cvtColor(internalData, tempMat, code, 0, cudaDevice->getOpenCVCudaStream());

    internalData.swap(tempMat);
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
    auto result = cudaMallocAsync((void**)&gpuOutput, internalData.rows * stepWithoutPadding * sizeof(uint8_t), cudaDevice->getCudaStream());
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
