#include <memory>

#include "TensorImage.h"

int main() {
	std::unique_ptr<TensorImage> t_img = TensorImage::createImageFromFile("img.jpg");
	return 0;
}

