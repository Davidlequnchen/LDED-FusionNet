# -*- coding: utf-8 -*-
"""
Created on Wednesday 11 October 2023
---------------------------------------------------------------------
-- Author: Chen Lequn
---------------------------------------------------------------------
FusionNet model
(1) AudioVisualFusionCNN: Image and Audio are on separate branch CNN, then fused together
   --> also the so-called "feature-level" fusion
(2) EarlyFusion: Image and Audio are fused directly in the initial layer. 
   --> also known as "data-level" fusion
"""
import os
import sys
from typing import Callable, Dict
from torchviz import make_dot
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import PIL.Image
import torch
import torch.utils.data
import pandas as pd
import torchaudio
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torchsummary import summary

import torch.nn as nn
import torch

# Retaining the VGG style model for the visual part
class VisionCNN(nn.Module):
    def __init__(self):
        super(VisionCNN, self).__init__()
        self.features = self._make_layers()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x

    def _make_layers(self):
        layers = [
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)
        ]
        return nn.Sequential(*layers)

class AudioFeatures(nn.Module):
    def __init__(self):
        super(AudioFeatures, self).__init__()
        # Placeholder for 13 key acoustic features
        self.features = nn.Linear(13, 13)

    def forward(self, x):
        return self.features(x)

class HybridAudioVisualFusion(nn.Module):
    def __init__(self):
        super(HybridAudioVisualFusion, self).__init__()
        self.vision_cnn = VisionCNN()
        self.audio_features = AudioFeatures()
        
        self.fc1 = nn.Linear(512 + 13, 32) # 512 from VisionCNN and 13 from AudioFeatures
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3) # Output 3 categories
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, audio):
        image_features = self.vision_cnn(image)
        audio_feat = self.audio_features(audio)
        
        fused = torch.cat((image_features, audio_feat), dim=1)
        
        x = nn.ReLU()(self.fc1(fused))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x


if __name__ == "__main__":
    model = HybridAudioVisualFusion()
    image = torch.randn(1, 1, 32, 32)
    audio = torch.randn(1, 1, 32, 18)  # (32,7) for 40ms, (32,18) for 100 ms
    predictions = model(image, audio)
    print (predictions)
    summary(model.cuda(), [(1, 32, 32), (1, 32, 18)]) # image and audio dual inputs
    make_dot(predictions.mean(), params=dict(model.named_parameters()))