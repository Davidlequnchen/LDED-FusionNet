# -*- coding: utf-8 -*-
"""
Created on Wednesday 11 Jan 2023
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


# class EarlyFusionCNN(nn.Module):
#     def __init__(self):
#         super(EarlyFusionCNN, self).__init__()
        
#         self.cnn = nn.Sequential(
#             nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)

#         )
        
#         self.fc = nn.Linear(128*6*6, 10)

#     def forward(self, image, audio):
#         # reshape the image data
#         image = image.view(-1, 1, 640, 480) # 640, 480 corresponds to the input dimension of the image
#         # reshape the audio data
#         audio = audio.view(-1, 1, 1470) # 1470 corresponds to the input dimension of the audio
#         # Combine the audio and image data along the channel dimension
#         combined = torch.cat((image, audio), dim=1)
#         # Pass the combined data through the CNN
#         features = self.cnn(combined)
#         features = features.view(features.size(0), -1)
#         out = self.fc(features)
#         return out


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1 # 3 for RGB, change to 1 for grey-scale image and audio
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class AudioVisualFusionCNN(nn.Module):
    def __init__(self):
        super(AudioVisualFusionCNN, self).__init__()
        # CNN architecture (Vision + Audio)
        self.image_stream = nn.Sequential(
            VGG('VGG16')
        )

        self.audio_stream = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten = nn.Flatten()
         # first element can get via summary; total 5 output class (laser-off, laser-start, defect-free, cracks, keyhole pores)
        self.linear = nn.Linear(1280, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, audio):
        image_features = self.image_stream(image)
        audio_features = self.audio_stream(audio)
        image_features = image_features.view(image_features.size(0), -1)
        audio_features = audio_features.view(audio_features.size(0), -1)
        fused_features = torch.cat((image_features, audio_features), dim=1)
        fused_features = self.flatten(fused_features)
        logits = self.linear(fused_features)
        predictions = self.softmax(logits)
        return predictions

if __name__ == "__main__":
    model = AudioVisualFusionCNN()
    image = torch.randn(1, 1, 32, 32)
    audio = torch.randn(1, 1, 20, 8)
    predictions = model(image, audio)
    print (predictions)
    summary(model.cuda(), [(1, 32, 32), (1, 20, 8)]) # image and audio dual inputs
    make_dot(predictions.mean(), params=dict(model.named_parameters()))