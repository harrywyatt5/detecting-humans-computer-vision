#!/usr/bin/env python3
import argparse
from osam._models.yoloworld.clip import tokenize
import numpy as np

TARGET_FILE = "language.token"

def save_token(file_name, byte_payload):
    with open(file_name, "wb") as f:
        f.write(byte_payload)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--token",
        type=str,
        help="String to tokenize",
        required=True
    )
    args = parser.parse_args()

    return args

def main():
    str_to_tokenise = parse_args().token
    binary_str = tokenize(texts=[str_to_tokenise], context_length=32).tobytes() 
    save_token(TARGET_FILE, binary_str)


if __name__ == "__main__":
    main()

