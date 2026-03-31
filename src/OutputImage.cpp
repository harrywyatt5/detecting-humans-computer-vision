#include "OutputImage.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

OutputImage::OutputImage(int x, int y) :  {
    auto result = cudaMalloc((void**)&data, x * y * 3 * sizeof(uint8_t));

    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to allocate output buffer. Reason: ") + cudaGetErrorString(result));
    }
}