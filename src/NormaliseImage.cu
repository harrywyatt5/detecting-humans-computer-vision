#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

__global__ void normaliseImage(const cv::cuda::PtrStepSz<uchar3> src, float* dst) {
    // Because we parallelise across the image, we can actually use thread information to find out where we are
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= src.cols || y >= src.rows) {
        return;
    }

    float r = ((src(y, x).x / 255.0f) - 0.485f) / 0.229f;
    float g = ((src(y, x).y / 255.0f) - 0.456f) / 0.224f;
    float b = ((src(y, x).z / 255.0f) - 0.406f) / 0.225f;

    // We wanna format the image as RRR... GGG... BBB... in a flat array
    dst[]
}

void launchNormaliseImage(const cv::cuda::GpuMat& mat, cv::cuda::Stream& stream, float* dst) {
    dim3 block(16, 16);
    dim3 grid((mat.cols + block.x - 1) / block.x, (mat.rows + block.y - 1) / block.y);

    cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
    normaliseImage<<<block, grid, 0, cudaStream>>>(mat, dst);
}