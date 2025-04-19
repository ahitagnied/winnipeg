import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, Resample

# this code is a PyTorch dataset class for loading and processing audio files for emotion recognition tasks.
class EmotionDataset(Dataset):
  def __init__(self, root_dir, sample_rate=16000, duration=3.0, n_mels=128, transform=None) -> None:
    self.root_dir = root_dir
    self.sample_rate = sample_rate
    self.duration = duration
    self.n_mels = n_mels
    self.n_samples = int(duration * sample_rate)
    self.transform = transform

    # get all wav files and their emotions
    self.files = []
    self.emotion = []
    self.emotion_to_idx = {}

    for i, emotion in enumerate(os.listdir(root_dir)): # for each i, emotion in the folder, for example (0, 'anger')
      emotion_dir = os.path.join(root_dir, emotion)
      if os.path.isdir(emotion_dir):
        self.emotion_to_idx[emotion] = i

        for file in os.listdir(emotion_dir): # for each file in current emotion folder
          if file.endswith('.wav'): # if its an audio file
            self.files.append(os.path.join(emotion_dir, file)) # append the location of the file to self.files
            self.emotion.append(emotion) # append the emotion to self.emotion

    self.idx_to_emotion = {v : k for k, v in self.emotion_to_idx.items()}

    self.mel_transform = MelSpectrogram( # visual representation of the frequencies in the file
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )

  def __len__(self) -> int:
    return len(self.files) # number of wav files

  def __getitem__(self, idx): # get file at a particular index
    # if idx is a tensor -> list
    if torch.is_tensor(idx):
      idx = idx.tolist()

    # load the audio file
    audio_file = self.files[idx]
    waveform, sample_rate = torchaudio.load(audio_file, backend="soundfile") # load the audio file using soundfile backend

    # convert to mono if needed
    if waveform.shape[0] > 1:
      waveform = torch.mean(waveform, dim=0, keepdim=True)

    # resample if necessary
    if sample_rate != self.sample_rate:
        resampler = Resample(sample_rate, self.sample_rate)
        waveform = resampler(waveform)

    # adjust the length
    if waveform.shape[1] < self.n_samples:
      waveform = F.pad(waveform, (0, self.n_samples - waveform.shape[1]))
    else:
      waveform = waveform[:, :self.n_samples]

    # convert to mel spectrogram
    mel_spectrogram = self.mel_transform(waveform)

    # apply log scaling (common for spectrograms)
    mel_spectrogram = torch.log(mel_spectrogram + 1e-9)

    # get emotion label
    emotion = self.emotion[idx]
    label = self.emotion_to_idx[emotion]

    sample = {'audio': waveform, 'spectrogram': mel_spectrogram, 'label': label, 'emotion': emotion}

    if self.transform:
      sample = self.transform(sample)

    return sample
#----------------------------------------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Make sure dimensions match before adding
        # x shape: [seq_len, batch_size, embedding_dim]
        # pe shape: [seq_len, 1, embedding_dim]
        
        # Check if the embedding dimension matches
        if x.size(2) != self.d_model:
            raise ValueError(f"Expected embedding dimension {self.d_model}, but got {x.size(2)}")
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Winnipeg(nn.Module):
    def __init__(self, nmels=64, nemo=8, seq_len=None):
        super().__init__()

        if seq_len is None:
            # total number of audio samples = 16000 * 4 = 64000
            # number of frames = 64000 / 512 = 125 <- 512 is the hop length param used in stft to get the mel spectrogram
            seq_len = 4 * 16000 // 512 # number of time frames in spectrogram

        # cnn feature extractor - removed third conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        # Third conv layer removed

        # pooling to reduce layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Now using 2 pooling operations instead of 3
        # final size reduction is 2^2 = 4 (not 8 anymore)

        # define a fixed embedding dimension for the transformer
        self.embedding_dim = 64
        
        # updated output dimension after removing third conv layer
        self.output_dim = 64 * (nmels // 4)
        self.projection = nn.Linear(self.output_dim, self.embedding_dim)

        # positional encoding
        self.pos_encoder = PositionalEncoding(self.embedding_dim, dropout=0.1, max_len=seq_len)

        # transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=False
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)

        # fully connected layers -> output classification
        self.fc1 = nn.Linear(self.embedding_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, nemo) # number of emotions

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)

        # x shape: [batch_size, 1, n_mels, time_frames]
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Third conv layer removed

        # x shape is now: [batch_size, 64, n_mels/4, time_frames/4]
        # Reshape properly for the projection layer
        time_frames = x.size(3)
        x = x.permute(0, 3, 1, 2)  # [batch_size, time_frames/4, 64, n_mels/4]
        x = x.reshape(batch_size, time_frames, -1)  # [batch_size, time_frames/4, 64 * (n_mels/4)]
        
        # Project to embedding dimension
        x = self.projection(x)  # [batch_size, time_frames/4, embedding_dim]
        
        # Change from batch_first to time_first for positional encoding
        x = x.transpose(0, 1)  # [time_frames/4, batch_size, embedding_dim]
        
        # apply positional encoding
        x = self.pos_encoder(x)
        
        # apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Change back to batch_first and take the mean of the sequence dimension
        x = x.transpose(0, 1)  # [batch_size, time_frames/4, embedding_dim]
        x = x.mean(dim=1)  # [batch_size, embedding_dim]

        # Final classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
#----------------------------------------------------------------------------------------------------------------