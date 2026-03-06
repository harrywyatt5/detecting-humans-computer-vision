#pragma once

#include "CudaTensor.h"
#include "ImageProvider.h"
#include <vector>
#include <cstdint>
#include <string>
#include <opencv2/opencv.hpp>

class PersistentImageInput : public ImageProvider {
private:
    cv::cuda::GpuMat resizedImage;
    cv::cuda::GpuMat gpuImage;
    cv::Mat cpuBlob;
    cv::cuda::Stream stream;
    int x;
    int y;
    int resizedX;
    int resizedY;

    bool hasUploadedImage;
public:
    PersistentImageInput(int imageX, int imageY, int resizedX, int resizedY, int deviceId);
    void uploadImageFromDisk(const std::string& path);
    // TODO: implement this!
    void uploadImageFromSensorMsg() {};
    void writeImageToCudaTensor(CudaTensor<float>& tensor) override;
};
