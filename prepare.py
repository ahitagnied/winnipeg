import os
import numpy as np
import librosa
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from config import *

# set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# check if gpu is available
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

def augment_audio(audio, sample_rate=SAMPLE_RATE):
    """apply random augmentations to audio"""
    augmentations = []
    
    # time stretch (speed up or slow down)
    if np.random.random() > 0.5:
        rate = np.random.uniform(0.8, 1.2)
        augmentations.append(lambda x: librosa.effects.time_stretch(x, rate=rate))
    
    # pitch shift
    if np.random.random() > 0.5:
        n_steps = np.random.uniform(-3, 3)
        augmentations.append(lambda x: librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=n_steps))
    
    # add random noise
    if np.random.random() > 0.5:
        noise_factor = np.random.uniform(0.005, 0.02)
        augmentations.append(lambda x: x + noise_factor * np.random.randn(len(x)))
    
    # apply augmentations in random order
    np.random.shuffle(augmentations)
    for aug in augmentations:
        audio = aug(audio)
    
    return audio

def extract_features(file_path):
    """extract mfcc features from an audio file

    args:
        file_path: path to audio file

    returns:
        mfcc features
    """
    try:
        # load audio file
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # ensure consistent length
        if len(audio) > MAX_LENGTH:
            audio = audio[:MAX_LENGTH]
        else:
            audio = np.pad(audio, (0, MAX_LENGTH - len(audio)))

        # extract mfcc features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # add delta features (first and second derivatives)
        delta1 = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # combine features
        mfcc = np.vstack([mfcc, delta1, delta2])

        return mfcc

    except Exception as e:
        print(f"error extracting features from {file_path}: {e}")
        return None

def load_data():
    """load audio files and extract features
    returns:
        features and labels
    """
    features = []
    labels = []
    print("loading data and extracting features...")
    
    # get all emotion folders (exclude test and ipynb_checkpoints folders)
    emotion_folders = [f for f in os.listdir(DATA_PATH)
                      if os.path.isdir(os.path.join(DATA_PATH, f))
                      and f != "Test"
                      and f != ".ipynb_checkpoints"]
    
    # track class counts for weighting
    class_counts = []
    
    # loop through each emotion folder
    for emotion in emotion_folders:
        emotion_path = os.path.join(DATA_PATH, emotion)
        # get all audio files
        audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        print(f"processing {len(audio_files)} files for emotion: {emotion}")
        class_counts.append(len(audio_files))
        
        # loop through each audio file
        for audio_file in tqdm(audio_files):
            file_path = os.path.join(emotion_path, audio_file)
            # extract features
            mfcc = extract_features(file_path)
            if mfcc is not None:
                features.append(mfcc)
                labels.append(emotion)
    
    # convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # calculate weights for loss function
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"data loaded: {features.shape[0]} samples, {len(label_encoder.classes_)} emotions")
    print(f"emotions: {label_encoder.classes_}")
    print(f"class counts: {class_counts}")
    print(f"class weights: {class_weights}")
    
    # at this point don't augment the actual dataset, just return the weights
    # we'll use weighted sampling in the DataLoader instead
    
    return features, encoded_labels, label_encoder.classes_, class_weights.to(device)