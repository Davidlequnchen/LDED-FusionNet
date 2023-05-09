# -*- coding: utf-8 -*-
"""
Created on 9 May 2023
---------------------------------------------------------------------
-- Author: Chen Lequn
---------------------------------------------------------------------
FusionNet model
(1) AudioVisualFusionCNN: Image and Audio are on separate branch CNN, then fused together
   --> also the so-called "feature-level" fusion

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
        self.classifier = nn.Linear(512, 4)

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



class AudioVisualFusionCNNLSTM(nn.Module):
    def __init__(self):
        super(AudioVisualFusionCNNLSTM, self).__init__()
        # CNN architecture (Vision + Audio)
        self.image_stream = nn.Sequential(
            VGG('VGG11')
        )

        self.audio_stream = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=1, batch_first=True)  # 1664 for 100ms; 1280 for 40ms
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256, 4)  # Update the input size to match the LSTM's hidden_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, audio):
        batch_size, timesteps, _, _ = audio.size()

        image_features = self.image_stream(image)
        audio_features = self.audio_stream(audio)

        # Reshape the features to be compatible with LSTM input
        image_features = image_features.view(batch_size, timesteps, -1)
        audio_features = audio_features.view(batch_size, timesteps, -1)

        # Concatenate image and audio features
        fused_features = torch.cat((image_features, audio_features), dim=2)

        # Pass the fused features through the LSTM layer
        lstm_out, _ = self.lstm(fused_features)

        # Take the last LSTM output as input for the final linear layer
        logits = self.linear(lstm_out[:, -1, :])
        predictions = self.softmax(logits)
        return predictions



if __name__ == "__main__":
    model = AudioVisualFusionCNNLSTM()
    image = torch.randn(1, 1, 32, 32)
    audio = torch.randn(1, 1, 32, 7)  # (32,7) for 40ms, (32,18) for 100 ms
    predictions = model(image, audio)
    print (predictions)
    summary(model.cuda(), [(1, 32, 32), (1, 32, 7)]) # image and audio dual inputs
    make_dot(predictions.mean(), params=dict(model.named_parameters()))