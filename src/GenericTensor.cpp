#include "GenericTensor.h"

#include <vector>
#include <cstdint>

template<typename T>
size_t GenericTensor<T>::getTensorCountFromShape(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (auto i = 0; i < shape.size(); ++i) {
        count *= shape[i];
    }

    return count;
}
