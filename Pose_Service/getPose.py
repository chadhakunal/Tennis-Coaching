import time
import sys
import os
import PIL
from PIL import ImageFont
import matplotlib.pylab as plt
import numpy as np
import json
import torch
from mpl_toolkits.mplot3d import Axes3D
from pose3d_utils.coords import ensure_cartesian
from VideoFrames import VideoFrames


from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.data_specs import ImageSpecs
from margipose.models import load_model
from margipose.utils import seed_all, init_algorithms, plot_skeleton_on_axes3d, plot_skeleton_on_axes, angleBetween


CPU = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
init_algorithms(deterministic=True)
torch.set_grad_enabled(False)
torch.no_grad()
seed_all(12345)

MODEL_PATH = "./pretrained/margipose-mpi3d.pth"
INPUT_FILE = "./Input/1.avi"

start = time.time()
model = load_model(MODEL_PATH).to(CPU).eval()
end = time.time()
print(end-start, "(s) to load Model")

def process_image(image):
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
    image = process_image(img)
    input_image = input_specs.convert(image).to(CPU, torch.float32)
    output = model(input_image[None, ...])[0]
    norm_skel3d = ensure_cartesian(output.to(CPU, torch.float64), d=3)
    coords = norm_skel3d.cpu().numpy()
    return coords
    
def getPose(frameArray):
    skel3DArray = np.zeros((17, 3, frameArray.shape[3]), dtype=np.float)
    for i in range(frameArray.shape[3]):
        img = PIL.Image.fromarray(frameArray[:,:,:,i][..., ::-1])
        coords = get_coordinates(img)
        skel3DArray[:, :, i] = coords
    return skel3DArray
    
    
if __name__=="__main__":
    filename = os.path.basename(INPUT_FILE)
    filename_noext = os.path.splitext(filename)[0]
    (frameArray, fps) = VideoFrames.ExtractFrames(INPUT_FILE)
    frameArray = np.asarray(frameArray, dtype=np.uint8)
    skel3DArray = getPose(frameArray)    
    print(skel3DArray)