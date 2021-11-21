import os
import sys
import json
import argparse
import configparser
import multiprocessing
from datetime import datetime
import pytz
import math
import matplotlib.pylab as plt
#plt.use('Agg')
sys.path.append(os.getcwd())
import random
import numpy as np
import shutil
from glob import glob
import skimage.io as io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.models import resnet50


class Predictor(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True, progress=False).eval()
        self.resnet50 = nn.Sequential(*list(resnet.children())[:-1])
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            x = self.resnet50(x)
            x = x.reshape(x.shape[0], -1)
            return x


device = 'cuda:1'
root = '/home/tokuoka/git/seminiferous_tubule_stage_classification/'
dist = os.path.join(root, 'results', 'inference_ResNet_ImageNet')
os.makedirs(dist, exist_ok='True')
os.makedirs(os.path.join(dist, 'features'), exist_ok='True')

resnet = Predictor().to(device)
path = np.sort(glob(os.path.join(root, 'datasets', 'images', '*.tif')))

for p in path:
    image = torch.Tensor(np.expand_dims(io.imread(p).transpose(2, 0, 1), axis=0)).to(device)
    #label = os.path.basename(p)[p.rfind('_')+1:p.rfind('-')]
    pred = resnet(image).to('cpu').detach()[0]
    filename = os.path.basename(p[:p.rfind('.')])
    np.save(os.path.join(dist, 'features', filename), pred)
