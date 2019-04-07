# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 14:23:44 2017

@author: Rohit Raj
"""
#Obeject Detection

#Impoting the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
import ssd import build_ssd
import imageio

#Defining the function that will do the detections
def detect(frame, net, transform):
    height, width = frame.shape[ :2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])