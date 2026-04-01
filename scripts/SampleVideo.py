#/usr/bin/env python3
import cv2
import os
import argparse
import math

def create_parser():
    parser = argparse.ArgumentParser(description="Extract frames into jpg format")
    parser.add_argument("video", help="Path to mp4 file")
    parser.add_argument("-o", "--output", default="frames", help="Folder to save frames into")
    parser.add_argument("-n", "--number", type=int, required=True, help="Number of frames to sample from video")

    return parser

def main():
    args = create_parser().parse_args()

    video_path = args.video
    video_base_name = os.path.basename(video_path)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Video could not be loaded")
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frequency = math.floor(total_frames / args.number)
    for i in range(0, total_frames, frequency):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video_capture.read()
        if ret:
            out = os.path.join(args.output, f"{video_base_name}_{i}.jpg")
            cv2.imwrite(out, frame)
        else:
            print(f"Failed to save frame {i}")
    
    video_capture.release()



if __name__ == "__main__":
    main()
