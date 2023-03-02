import json
import os
import sys
from typing import Callable, Dict

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




class MultimodalDataset(Dataset):
    def __init__(self,
                 annotation_df,
                 image_path,
                 audio_path,
                 image_transformation,
                #  mel_spectrogram,
                 MFCCs,
                #  spectral_centroid,
                 target_sample_rate,
                #  num_samples,
                 device):
        self.annotations = annotation_df
        self.image_dir = image_path
        self.audio_dir = audio_path
        self.device = device
        # self.img_transform = nn.DataParallel(image_transformation) # parallelize the computation on multiple GPUs.
        self.img_transform = image_transformation
        # self.mel_spectrogram = mel_spectrogram.to(self.device)
        self.MFCCs = MFCCs.to(self.device)
        # self.spectral_centroid = spectral_centroid.to(self.device)
        self.target_sample_rate = target_sample_rate
        # self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        image_sample_path = self._get_image_sample_path(index)

        label = self._get_sample_label(index)
        audio_signal, sr = torchaudio.load(audio_sample_path)
        audio_signal = audio_signal.to(self.device)
        # signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)

        image = PIL.Image.open(image_sample_path).convert('L')
        # Convert PIL.Image object to PyTorch tensor
        # image = torch.from_numpy(np.array(image))    
        image_transformed = self.img_transform(image)
        # conduct the transformations
        # signal_mel_spectrogram = self.mel_spectrogram(audio_signal)
        signal_mfcc = self.MFCCs(audio_signal)
        # signal_spectral_centroid = self.spectral_centroid(audio_signal)
    
        return (image_transformed, signal_mfcc), label


    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 1])
        return path
    
    def _get_image_sample_path(self, index):
        path = os.path.join(self.image_dir, self.annotations.iloc[index, 2])
        return path

    def _get_sample_label(self, index):
        return self.annotations.iloc[index, 6]



class LDEDVisionDataset(Dataset):
    def __init__(self,
                 annotation_df,
                 image_path,
                 image_transformation,
                 device):
        self.annotations = annotation_df
        self.image_dir = image_path
        self.device = device
        self.img_transform = image_transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_sample_path = self._get_image_sample_path(index)
        label = self._get_sample_label(index)
        image = PIL.Image.open(image_sample_path).convert('L')
        image_transformed = self.img_transform(image)
        return image_transformed, label

    
    def _get_image_sample_path(self, index):
        path = os.path.join(self.image_dir, self.annotations.iloc[index, 2])
        return path

    def _get_sample_label(self, index):
        return self.annotations.iloc[index, 6]


class LDEDAudioDataset(Dataset):
    def __init__(self,
                 annotation_df,
                 audio_path,
                #  mel_spectrogram,
                 MFCCs,
                #  spectral_centroid,
                 target_sample_rate,
                 device):
        self.annotations = annotation_df
        self.audio_dir = audio_path
        self.device = device
        # self.mel_spectrogram = mel_spectrogram.to(self.device)
        self.MFCCs = MFCCs.to(self.device)
        # self.spectral_centroid = spectral_centroid.to(self.device)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)

        label = self._get_sample_label(index)
        audio_signal, sr = torchaudio.load(audio_sample_path)
        audio_signal = audio_signal.to(self.device)
        # signal = self._resample_if_necessary(signal, sr)
        # signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_if_necessary(signal)
        # signal = self._right_pad_if_necessary(signal)

        # conduct the transformations
        # signal_mel_spectrogram = self.mel_spectrogram(audio_signal)
        signal_mfcc = self.MFCCs(audio_signal)
        # signal_spectral_centroid = self.spectral_centroid(audio_signal)
        
        return signal_mfcc, label


    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 1])
        return path
    
    def _get_sample_label(self, index):
        return self.annotations.iloc[index, 6]



if __name__ == "__main__":

    # Multimodal_dataset_PATH = os.path.join("C:\\Users\\Asus\\OneDrive_Chen1470\\OneDrive - Nanyang Technological University\\Dataset\\Multimodal_AM_monitoring\\LDED_Acoustic_Visual_Dataset")
    Multimodal_dataset_PATH = os.path.join("/home/chenlequn/Dataset/Multimodal_AM_monitoring/LDED_Acoustic_Visual_Dataset")
    CCD_Image_30Hz_path = os.path.join(Multimodal_dataset_PATH, 'Coaxial_CCD_images_30Hz')
    Audio_segmented_30Hz_PATH = os.path.join(Multimodal_dataset_PATH, 'Audio_signal_all_30Hz')
    Audio_raw_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'raw')
    Audio_equalized_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'equalized')
    Audio_bandpassed_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'bandpassed')
    Audio_denoised_seg_PATH = os.path.join(Audio_segmented_30Hz_PATH, 'denoised')

    ANNOTATIONS_FILE = os.path.join(Multimodal_dataset_PATH, "vision_acoustic_label_v2.csv")
    annotation_df = pd.read_csv(ANNOTATIONS_FILE)

    ## select denoised audio signal
    AUDIO_DIR = Audio_denoised_seg_PATH
    VISON_DIR = CCD_Image_30Hz_path
    SAMPLE_RATE = 44100
    # NUM_SAMPLES = 44100

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32,32)), # original image size: (640,480)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[136.21606092530587], std=[61.86861591088625]), #136.21606092530587 61.86861591088625
            # note that if resize is before normalization, need to re-calculate the mean and std; if resize is after normalize, could induce distortions
        ]) # calculation of mean and std is shown in jupyter notebook 1.
 
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                            n_fft=512,
                                            hop_length=256,
                                            n_mels=32)

    MFCCs = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE,n_mfcc=20)
    # spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=SAMPLE_RATE, hop_length=256)
    
                
    mmd = MultimodalDataset(annotation_df,
                            VISON_DIR,
                            AUDIO_DIR,
                            img_transform,
                            # mel_spectrogram,
                            MFCCs,
                            # spectral_centroid,
                            SAMPLE_RATE,
                            # NUM_SAMPLES,
                            device)

    visiondataset = LDEDVisionDataset(annotation_df,
                                      VISON_DIR,
                                      img_transform,
                                      device)

    audiodataset = LDEDAudioDataset(annotation_df,
                                    AUDIO_DIR,
                                    # mel_spectrogram,
                                    MFCCs,
                                    # spectral_centroid,
                                    SAMPLE_RATE,
                                    device)
    
    # random check
    print(f"There are {len(mmd)} samples in the multimodal dataset.")
    print(f"There are {len(visiondataset)} samples in the visiondataset dataset.")
    print(f"There are {len(audiodataset)} samples in the audiodataset dataset.")
    multimodal_inputs, label = mmd[100]
    image_input_vision, label_vision = visiondataset[50]
    audio_input_audioset, label_audio = audiodataset[3000]

    print (multimodal_inputs[0].shape, multimodal_inputs[1].shape, label)
    # print (image_input_vision.shape, label_vision)
    # print (audio_input_audioset.shape, label_audio)

