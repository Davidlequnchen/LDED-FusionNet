import os
import pickle

import librosa
import numpy as np
from tqdm.auto import tqdm


Multimodal_dataset_PATH = os.path.join("C:\\Users\\Asus\\OneDrive_Chen1470\\OneDrive - Nanyang Technological University\\Dataset\\Multimodal_AM_monitoring\\LDED_Acoustic_Visual_Dataset")
Audio_segmented_10Hz_PATH = os.path.join(Multimodal_dataset_PATH, 'Audio_signal_all_10Hz')
Audio_raw_seg_PATH = os.path.join(Audio_segmented_10Hz_PATH, 'raw')
Audio_equalized_seg_PATH = os.path.join(Audio_segmented_10Hz_PATH, 'equalized')
Audio_bandpassed_seg_PATH = os.path.join(Audio_segmented_10Hz_PATH, 'bandpassed')
Audio_denoised_seg_PATH = os.path.join(Audio_segmented_10Hz_PATH, 'denoised')
AUDIO_FEATURES_PATH = os.path.join(Audio_segmented_10Hz_PATH, 'Audio_features', 'audio_features.p')


def get_librosa_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=None)

    hop_length = 256  # Set the hop length; at 44100 Hz, 256 samples ~= 5.8ms

    # compute magnitute and phase from STFT
    D = librosa.stft(y, hop_length=hop_length)
    S_full, phase = librosa.magphase(D)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)  # Compute MFCC features from the raw signal
    mfcc_delta = librosa.feature.delta(mfcc)  # And the first-order differences (delta features)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=hop_length)
    S_delta = librosa.feature.delta(S)

    spectral_centroid = librosa.feature.spectral_centroid(S=S_full, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S_full, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S_full, roll_percent=0.85, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(S=S_full, hop_length=hop_length)

    audio_feature = np.vstack((mfcc, mfcc_delta, S, S_delta, spectral_centroid,
                               spectral_bandwidth, spectral_rolloff, spectral_flatness))  # combine features

    # binning data
    jump = int(audio_feature.shape[1] / 10)
    return librosa.util.sync(audio_feature, range(1, audio_feature.shape[1], jump))


def save_audio_features() -> None:
    audio_feature = {}
    for filename in tqdm(os.listdir(Audio_denoised_seg_PATH), desc="Computing the audio features"):
        id_ = filename.rsplit(".", maxsplit=1)[0]
        id_ = os.path.basename(id_).split("_")[1][3:]
        audio_feature[id_] = get_librosa_features(os.path.join(Audio_denoised_seg_PATH, filename))
        print(audio_feature[id_].shape)

    with open(AUDIO_FEATURES_PATH, "wb") as file:
        pickle.dump(audio_feature, file)


def get_audio_duration() -> None:
    filenames = os.listdir(Audio_denoised_seg_PATH)
    print(sum(librosa.core.get_duration(filename=os.path.join(Audio_denoised_seg_PATH, filename))
              for filename in tqdm(filenames, desc="Computing the average duration of the audios")) / len(filenames))


def main() -> None:
    get_audio_duration()

    save_audio_features()
    #
    with open(AUDIO_FEATURES_PATH, "rb") as file:
        pickle.load(file)


if __name__ == "__main__":
    main()
