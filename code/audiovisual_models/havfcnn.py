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
    def __init__(self, num_features):
        super(AudioFeatures, self).__init__()
        self.features = nn.Linear(num_features, num_features)

    def forward(self, x):
        return self.features(x)

class HybridAudioVisualFusion(nn.Module):
    def __init__(self, num_classes=4, num_audio_features=13):
        super(HybridAudioVisualFusion, self).__init__()
        self.vision_cnn = VisionCNN()
        self.audio_features = AudioFeatures(num_audio_features)
        
        self.fc1 = nn.Linear(512 + num_audio_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, audio):
        image_features = self.vision_cnn(image)
        audio_feat = self.audio_features(audio)
        image_features = image_features.view(image_features.size(0), -1)
        audio_feat = audio_feat.view(audio_feat.size(0), -1)
        fused = torch.cat((image_features, audio_feat), dim=1)
    
        x = nn.ReLU()(self.fc1(fused))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x


if __name__ == "__main__":
    num_audio_features = 13
    model = HybridAudioVisualFusion(num_audio_features=num_audio_features)
    image = torch.randn(1, 1, 32, 32)
    audio = torch.randn(1, num_audio_features) 
    predictions = model(image, audio)
    print(predictions)
    print(model)
    # summary(model.cuda(), [(1, 32, 32), (num_audio_features,)])

