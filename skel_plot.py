import time
import sys
import os
import PIL
import cv2
from PIL import ImageFont
import matplotlib.pylab as plt
import numpy as np
import json
import torch
import pandas as pd

sys.path.append("./Pose_Service/")

from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian
from Pose_Service.VideoFrames import VideoFrames


from Pose_Service.margipose.data.skeleton import CanonicalSkeletonDesc
from Pose_Service.margipose.data_specs import ImageSpecs
from Pose_Service.margipose.models import load_model
from Pose_Service.margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d, plot_skeleton_on_axes, angleBetween

CPU = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
init_algorithms(deterministic=True)
torch.set_grad_enabled(False)
torch.no_grad()
seed_all(12345)

MODEL_PATH = "./Pose_Service/pretrained/margipose-mpi3d.pth"
start = time.time()
model = load_model(MODEL_PATH).to(CPU).eval()
end = time.time()
print(end-start, "(s) to load Model")


INPUT_FILE_1 = "Pose_Service/Input/2.avi"
# INPUT_FILE_2 = "Pose_Service/Input/2.avi"


filename = os.path.basename(INPUT_FILE_1)
filename_noext = os.path.splitext(filename)[0]
(frameArray_1, fps_1) = VideoFrames.ExtractFrames(INPUT_FILE_1)
frameArray_1 = np.asarray(frameArray_1, dtype=np.uint8)

# filename = os.path.basename(INPUT_FILE_2)
# filename_noext = os.path.splitext(filename)[0]
# (frameArray_2, fps_2) = VideoFrames.ExtractFrames(INPUT_FILE_2)
# frameArray_2 = np.asarray(frameArray_2, dtype=np.uint8)

def process_image(image, input_specs):
    try:
        image: PIL.Image.Image = PIL.Image.open(image, 'r')
    except:
        pass
    if image.width != image.height:
        cropSize = min(image.width, image.height)
        image = image.crop((image.width/2 - cropSize/2, image.height/2 - cropSize/2,
                    image.width/2 + cropSize/2, image.height/2 + cropSize/2))
    if image.width < 256:
        image = image.resize((256, 256), PIL.Image.ANTIALIAS)
    image.thumbnail((input_specs.width, input_specs.height))
    return image

def get_coordinates(img):
    input_specs: ImageSpecs = model.data_specs.input_specs
    image = process_image(img, input_specs)
    input_image = input_specs.convert(image).to(CPU, torch.float32)
    output = model(input_image[None, ...])[0]
    norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)
    coords = norm_skel3d.cpu().numpy()
    
    fig = plt.figure(1)
    plt_3d: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
    plot_skeleton_on_axes3d(norm_skel3d, CanonicalSkeletonDesc, plt_3d, invert=True)
    # plt.show()

    #saving all outputs as image files with corresponding filename
    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer._renderer, np.uint8)[:,:,:3]
    # fig_img = fig_img[:,:,:3] 
    plt.close(fig)

    return coords, fig_img
    
def getPose(frameArray):
    skel3DArray = np.zeros((frameArray.shape[3], 17, 3), dtype=np.float)
    skel3DArray_plot = np.zeros((480, 640, 3, frameArray.shape[3]), dtype=np.uint8)
    for i in range(frameArray.shape[3]):
        print("Frame", i, "/", frameArray.shape[3])
        img = PIL.Image.fromarray(frameArray[:,:,:,i][..., ::-1])
        coords, skel3DArray_plot[:,:,:,i] = get_coordinates(img)
        skel3DArray[i, :, :] = coords
    return skel3DArray, skel3DArray_plot


skel3DArray, skel_plot = getPose(frameArray_1)
# skel3DArray2, skel_plot2 = getPose(frameArray_2)

outskel3D = cv2.VideoWriter('./skel_2.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps_1, (skel_plot.shape[1],skel_plot.shape[0]))

for i in range(skel_plot.shape[3]):
    outskel3D.write(skel_plot[:,:,:,i])

outskel3D.release()


