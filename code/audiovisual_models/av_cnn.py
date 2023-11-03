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
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torchvision.models as models


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


def get_visual_backbone(architecture: str):
    """
    Returns the desired visual backbone model.
    
    Args:
    - architecture (str): The name of the desired architecture. (e.g., 'VGG11', 'ResNet50')
    
    Returns:
    - model (nn.Module): The corresponding model.
    """
    if "VGG" in architecture:
        return VGG(architecture)
    elif "ResNet18" == architecture:
        return resnet18(pretrained=False)
    elif "ResNet34" == architecture:
        return resnet34(pretrained=False)
    elif "ResNet50" == architecture:
        return resnet50(pretrained=False)
    elif "ResNet101" == architecture:
        return resnet101(pretrained=False)
    elif "ResNet152" == architecture:
        return resnet152(pretrained=False)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


class AudioBackbone(nn.Module):
    def __init__(self, backbone_type, num_classes=4, num_handcrafted_features=None):
        super(AudioBackbone, self).__init__()
        
        # Select the backbone type
        if backbone_type == "MFCCCNN":
            self.backbone = MFCCCNN(num_classes=num_classes)
        elif "VGG" in backbone_type:
            self.backbone = models.vgg11_bn(pretrained=True) if backbone_type == "VGG11" else models.vgg16_bn(pretrained=True)
        elif "ResNet" in backbone_type:
            self.backbone = models.resnet18(pretrained=True) if backbone_type == "ResNet18" else models.resnet50(pretrained=True)
        elif backbone_type == "Handcrafted":
            assert num_handcrafted_features is not None, "Specify number of handcrafted features"
            self.backbone = nn.Sequential(
                nn.Linear(num_handcrafted_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU()
            )
        
        # The final classification layers
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.fc2(x)
        return logits



class AudioVisualFusionCNN(nn.Module):
    def __init__(self, visual_backbone='VGG16', num_classes=4):
        super(AudioVisualFusionCNN, self).__init__()

        # Using the chosen visual backbone
        self.image_stream = get_visual_backbone(visual_backbone)
        
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
        self.dropout = nn.Dropout(0.5)
         # first element can get via summary; 
        self.linear = nn.Linear(1280, num_classes) #1280 for 40ms; 1664 for 100ms
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
    audio = torch.randn(1, 1, 32, 7)  # (32,7) for 40ms, (32,18) for 100 ms
    predictions = model(image, audio)
    print (predictions)
    summary(model.cuda(), [(1, 32, 32), (1, 32, 7)]) # image and audio dual inputs
    make_dot(predictions.mean(), params=dict(model.named_parameters()))