# -*- coding: utf-8 -*-
"""
Created on Wednesday 11 Jan 2023
---------------------------------------------------------------------
-- Author: Chen Lequn
---------------------------------------------------------------------
FusionNet model
(1) LateFusion: Image and Audio are on separate branch CNN, then fused together
   --> also the so-called "feature-level" fusion
(2) EarlyFusion: Image and Audio are fused directly in the initial layer. 
   --> also known as "data-level" fusion
"""

import torch
import torch.nn as nn

from torch import nn, optim
from torch.nn import functional as F
from torchsummary import summary



class EarlyFusionCNN(nn.Module):
    def __init__(self):
        super(EarlyFusionCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        
        self.fc = nn.Linear(128*6*6, 10)

    def forward(self, image, audio):
        # reshape the image data
        image = image.view(-1, 1, 640, 480) # 640, 480 corresponds to the input dimension of the image
        # reshape the audio data
        audio = audio.view(-1, 1, 1470) # 1470 corresponds to the input dimension of the audio
        # Combine the audio and image data along the channel dimension
        combined = torch.cat((image, audio), dim=1)
        # Pass the combined data through the CNN
        features = self.cnn(combined)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return out



class LateFusionNet(nn.Module):
    def __init__(self):
        super(LateFusionNet, self).__init__()
        # CNN architecture (Vision + Audio)
        self.image_stream = nn.Sequential(
            # nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
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
        self.linear = nn.Linear(3328, 5)
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
    model = LateFusionNet()
    image = torch.randn(1, 1, 64, 48)
    audio = torch.randn(1, 1, 32, 6)
    # predictions = model(image, audio)
    summary(model.cuda(), [(1, 64, 48), (1, 32, 6)]) # image and audio