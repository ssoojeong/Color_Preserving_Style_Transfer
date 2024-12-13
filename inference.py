import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

import cv2
import os
from tqdm import tqdm

from module.utils import *
from module.vgg import *

device = ("cuda" if torch.cuda.is_available() else "cpu")

#######
configs = {
    # Hyperparameters
    'MAX_IMAGE_SIZE' : 512,

    # Optimizer
    'OPTIMIZER' : 'adam', #or 'lbfgs'
    'ADAM_LR' : 10,
    'CONTENT_WEIGHT' : 5e0,
    'STYLE_WEIGHT' : 1e2,
    'TV_WEIGHT' : 1e-3,
    'NUM_ITER' : 500,
    'SHOW_ITER' : 100,

    # Image Files
    'INIT_IMAGE' : 'random', # or 'content'
    'PRESERVE_COLOR' : 'True', # 'False'
    'PIXEL_CLIP' : 'True', # or False - Clipping produces better images
    'CONTENT_PATH' : 'dataset/effect',
    'STYLE_PATH' : 'dataset/reference',
}
for key, value in configs.items():
    globals()[key] = value  # 전역 네임스페이스에 추가

"""
PRETRAINED VGG MODELS 
GITHUB REPO: https://github.com/jcjohnson/pytorch-vgg
VGG 19: https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth
VGG 16: https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth
"""
VGG19_PATH = 'models/vgg19-d01eb7cb.pth'
SAVE_PATH = 'outputs'
POOL = 'max'
#######

#######
# Load VGG19 Skeleton
vgg = models.vgg19(pretrained=False)

# Load pretrained weights
vgg.load_state_dict(torch.load(VGG19_PATH), strict=False)

#vgg.features = pool_(vgg.features, POOL)
model = copy.deepcopy(vgg.features)
model.to(device)

# Turn-off unnecessary gradient tracking
for param in model.parameters():
    param.requires_grad = False
#######


# Load Images
for style_file in tqdm(sorted(os.listdir(STYLE_PATH)), desc="Processing Styles"):
    style_path = os.path.join(STYLE_PATH, style_file)
    
    for content_file in tqdm(sorted(os.listdir(CONTENT_PATH)), desc=f"Processing Contents for {style_file}", leave=False):
        print(content_file)
        content_path = os.path.join(CONTENT_PATH, content_file)
      
        content_img = load_image(content_path)
        style_img = load_image(style_path)
        
        content_name = content_path.split('/')[-1].split('.')[0]
        style_name = style_path.split('/')[-1].split('.')[0]
        save_path = os.path.join(SAVE_PATH, 'reference_' + style_name)
        # os.makedirs(save_path, exist_ok=True)
        
        # Convert Images to Tensor 
        content_tensor = itot(content_img, MAX_IMAGE_SIZE).to(device)
        style_tensor = itot(style_img, MAX_IMAGE_SIZE).to(device)

        # Stylize!
        stylize(
            iteration=NUM_ITER,
            model=model,
            content_tensor=content_tensor,
            style_tensor=style_tensor,
            device=device,
            save_path=save_path,
            content_name=content_name,
            **configs
        )