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
        """
        initialize the positional encoding layer
        params:
            d_model: embedding dimension size (int)
            dropout: dropout probability (float, default: 0.1)
            max_len: maximum sequence length to support (int, default: 5000)
        """
        super().__init__()
        # create dropout layer with specified probability
        self.dropout = nn.Dropout(p=dropout)

        # create empty tensor to store positional encodings
        # dimensions: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # create position tensor of shape [max_len, 1]
        # contains values [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # calculate division terms for the sinusoidal formula
        # implements 1/(10000^(2i/d_model)) for even indices
        # dimensions: [d_model/2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # fill even indices with sine values
        # dimensions: pe[:, 0::2] selects even columns of pe
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # fill odd indices with cosine values
        # dimensions: pe[:, 1::2] selects odd columns of pe
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # add batch dimension and transpose to get shape [max_len, 1, d_model]
        # this is needed for broadcasting during forward pass
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # register pe as a buffer (not a parameter)
        # this way it won't be updated during backpropagation
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        add positional encoding to input embeddings
        params:
            x: input embeddings tensor of shape [seq_len, batch_size, d_model]
        returns:
            tensor of same shape with positional information added
        """
        # add positional encoding to input
        # pe is sliced to match input sequence length
        # dimensions: x is [seq_len, batch_size, d_model]
        #             pe[:x.size(0), :] is [seq_len, 1, d_model]
        x = x + self.pe[:x.size(0), :]
        
        # apply dropout and return
        # output dimensions: [seq_len, batch_size, d_model]
        return self.dropout(x)

class Winnipeg(nn.Module):
    def __init__(self, nmels=128, nemo=8, seq_len=None):
        super().__init__()

        if seq_len is None:
            # total number of audio samples = 16000 * 4 = 64000
            # number of frames = 64000 / 512 = 125 <- 512 is the hop length param used in stft to get the mel spectrogram
            seq_len = 4 * 16000 // 512 # number of time frames in spectrogram

        # cnn feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128) # <- 128 channels outputted

        # pooling to reduce layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # use 3 pooling operations, each pooling layer reduces size by 2
        # final size reduction is 2^3 = 8

        # Define a fixed embedding dimension for the transformer
        self.embedding_dim = 256
        
        # find the output size after the conv layers and add a projection layer
        self.output_dim = 128 * (nmels // 8)
        self.projection = nn.Linear(self.output_dim, self.embedding_dim)

        # positional encoding - use the embedding_dim here
        self.pos_encoder = PositionalEncoding(self.embedding_dim, dropout=0.1, max_len=seq_len)

        # transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,  # Use embedding_dim instead of output_dim
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=False  # Change to match positional encoding expectation
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

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

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # x shape: [batch_size, 128, n_mels/8, time_frames/8]
        # Reshape properly for the projection layer
        time_frames = x.size(3)
        x = x.permute(0, 3, 1, 2)  # [batch_size, time_frames/8, 128, n_mels/8]
        x = x.reshape(batch_size, time_frames, -1)  # [batch_size, time_frames/8, 128 * (n_mels/8)]
        
        # Project to embedding dimension
        x = self.projection(x)  # [batch_size, time_frames/8, embedding_dim]
        
        # Change from batch_first to time_first for positional encoding
        x = x.transpose(0, 1)  # [time_frames/8, batch_size, embedding_dim]
        
        # apply positional encoding
        x = self.pos_encoder(x)
        
        # apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Change back to batch_first and take the mean of the sequence dimension
        x = x.transpose(0, 1)  # [batch_size, time_frames/8, embedding_dim]
        x = x.mean(dim=1)  # [batch_size, embedding_dim]

        # Final classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Fixed this line - you had 'x - self.fc2(x)' which is a typo
        
        return x
#----------------------------------------------------------------------------------------------------------------