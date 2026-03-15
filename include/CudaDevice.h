#pragma once

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

class CudaDevice {
private:
    int deviceId;
    cudaStream_t stream;
    cv::cuda::Stream openCVStream;
public:
    CudaDevice(int id);

    int getDeviceId() const;
    void switchCudaDevice() const;
    void waitForCompletion();
    cudaStream_t& getCudaStream();
    cv::cuda::Stream& getOpenCVCudaStream();
};
