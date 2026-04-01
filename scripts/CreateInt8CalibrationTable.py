#!/usr/bin/env python3
import argparse
import os
import numpy as np
import cv2
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table

class Reader(CalibrationDataReader):
    def __init__(self, images_dir):
        super().__init__()
        self.images = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        self.iter = iter(self.images)
        self.input_name = "images"
        self.input_shape = np.array([1, 3, 1008, 1008])

    def get_next(self):
        path = next(self.iter, None)
        if path is None:
            return None
        
        img = cv2.imread(path)
        img = cv2.resize(img, (1008, 1008), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0

        img = np.transpose(img, (2, 0, 1))
        input_data = np.expand_dims(img, axis=0).astype(np.float32)
        return {self.input_name: input_data}

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the folder containing the images")
    parser.add_argument("-e", "--encoder", required=True, help="The path to the ONNX Image encoder you wish to quantize")

    return parser

def main():
    args = create_arg_parser().parse_args()

    if not os.path.exists(args.path):
        raise Exception("Folder does not exist")

    if not os.path.exists(args.encoder):
        raise Exception("Model does not exist")
    
    calibrator = create_calibrator(
        model=args.encoder,
        op_types_to_calibrate=[],
        augmented_model_path="temp.onnx"
    )
    calibrator.set_execution_providers(["CPUExecutionProvider"])

    data_reader = Reader(args.path)
    calibrator.collect_data(data_reader=data_reader)
    write_calibration_table(calibrator.compute_data(), dir=".", file_name="calibration.flatbuffers")

    # Clean up temp file
    if os.path.exists("temp.onnx"):
        os.remove("temp.onnx")

if __name__ == "__main__":
    main()
