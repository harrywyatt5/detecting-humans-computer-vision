#pragma once

#include "CudaTensor.h"
#include <vector>
#include <cstdint>
#include <string>
#include <opencv2/opencv.hpp>

class PersistentImageInput {
private:
    cv::cuda::GpuMat resizedImage;
    cv::cuda::GpuMat gpuImage;
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
    void writeImageToTensor(CudaTensor<float>& tensor);
};
