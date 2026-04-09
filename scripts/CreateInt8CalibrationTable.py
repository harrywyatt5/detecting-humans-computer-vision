#!/usr/bin/env python3
import argparse
import os
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt

class Calibrator(tensorrt.IInt8EntropyCalibrator2):
    def __init__(self, images_dir, intermediate_size):
        super().__init__()
        img_list = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, f) for f in img_list]
        self.batch_size = 1
        self.index = 0
        self.size = len(img_list)
        self.intermediate_size = intermediate_size

        self.device_mem = cuda.mem_alloc(3 * intermediate_size * intermediate_size * 4) # x * y * 3 channels * 4 bytes for float32

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= self.size:
            return None
        
        path = self.images[self.index]
        print(f"Current processing {path}", flush=True)

        img = cv2.imread(path)
        img = cv2.resize(img, (self.intermediate_size, self.intermediate_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img = np.ascontiguousarray(img)

        # Copy the image into the gpu
        cuda.memcpy_htod(self.device_mem, img)
        self.index += 1

        return [int(self.device_mem)]

    def read_calibration_cache(self):
        # Throw if calibration cache already exists
        if os.path.exists("calibration.cache"):
            raise Exception("Calibration file already exists")
        
        return None
    
    def write_calibration_cache(self, cache):
        with open("calibration.cache", "wb") as f:
            f.write(cache)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the folder containing the images")
    parser.add_argument("-e", "--encoder", required=True, help="The path to the ONNX Image encoder you wish to quantize")
    parser.add_argument("-s", "--image_size", type=int, default=1008, help="Default size for SAM3 intermediary")

    return parser

def main():
    args = create_arg_parser().parse_args()

    if not os.path.exists(args.path):
        raise Exception("Folder does not exist")

    if not os.path.exists(args.encoder):
        raise Exception("Model does not exist")
    
    image_size = args.image_size
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)
    builder = tensorrt.Builder(logger)
    network = builder.create_network(1 << int(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = tensorrt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    with open(args.encoder, "rb") as f:
        parser.parse(f.read())
    
    profile = builder.create_optimization_profile()
    profile.set_shape("images", (1, 3, image_size, image_size), (1, 3, image_size, image_size), (1, 3, image_size, image_size))
    config.add_optimization_profile(profile)
    config.set_flag(tensorrt.BuilderFlag.INT8)
    config.int8_calibrator = Calibrator(args.path, image_size)

    print("Starting to build int8 cache")
    builder.build_serialized_network(network, config)
    print("Finished!")


if __name__ == "__main__":
    main()
