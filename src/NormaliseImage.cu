#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

__global__ void normaliseImage(const cv::cuda::PtrStepSz<uchar3> src, float* dst, int x, int y) {

}

void launchNormaliseImage() {
    
}