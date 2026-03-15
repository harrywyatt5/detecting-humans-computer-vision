#include "CudaDevice.h"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdexcept>
    
CudaDevice::CudaDevice(int id) : deviceId(id) {
    switchCudaDevice();
    cudaStreamCreate(&stream);
    openCVStream = cv::cuda::StreamAccessor::wrapStream(stream);
}

int CudaDevice::getDeviceId() const {
    return deviceId;
}

void CudaDevice::switchCudaDevice() const {
    auto result = cudaSetDevice(deviceId);

    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Could not switch to cuda device. Reason: ") + cudaGetErrorString(result));
    }
}

void CudaDevice::waitForCompletion() {
    cudaStreamSynchronize(stream);
}

cudaStream_t& CudaDevice::getCudaStream() {
    return stream;
}

cv::cuda::Stream& CudaDevice::getOpenCVCudaStream() {
    return openCVStream;
}
