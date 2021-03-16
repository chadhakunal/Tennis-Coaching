import cv2
import os
import numpy as np
import PIL
import torch
import time
from pathlib import Path


def get_frames(input_file, output_dir):
    start = time.time()
    vid = cv2.VideoCapture(input_file)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frameTotal = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(frameHeight, 'x', frameWidth, 'Image size')
    # print(frameFoVideoFramesurCC,  "CC", type(frameFourCC))
    count = 0
    frameArray = np.zeros((frameHeight, frameWidth, 3 ,frameTotal), dtype=np.uint8)
    # frameArray = np
    
    print(frameArray.shape[3], "Initial Frame Total")
    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == False:
            print(time.time()-start, "(s) to complete video to frame")
            break
        # if count == 5: # for troubleshooting
        #     break
        if count > frameTotal:
            frame = np.expand_dims(frame, axis=-1)
            frameArray = np.append(frameArray, frame, axis=-1)
            continue
        frameArray[:,:,:,count] = frame
        count +=1
    vid.release()
    
    # print(frameArray[:,:,:,:count].shape, "Actual Shape")
    print(fps, "FPS")
    # number of frames using CAP_PROP_FRAME_COUNT is only an estimate, count keeps track of actual frames
    print(count, 'Actaul Total Frames')
    for i in range(count):
        print("Saving Frame", i)
        filename = os.path.splitext(Path(input_file).name)
        output_file = os.path.join(output_dir, filename[0] + f"_{i}.jpg")
        print(output_file)
        cv2.imwrite(output_file, frameArray[:,:,:,i])  
    return(frameArray[:,:,:,:count], fps)

if __name__=="__main__":
    input_file = "Input/1.avi"
    output_dir = "Input/Frames"
    get_frames(input_file, output_dir)