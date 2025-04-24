import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from config import *

class EmotionDataset(Dataset):
    """dataset for emotion audio classification"""

    def __init__(self, X, y):
        """
        args:
            X: features (mfcc)
            y: labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        """return the number of samples"""
        return len(self.X)

    def __getitem__(self, idx):
        """return a sample at index idx"""
        return torch.FloatTensor(self.X[idx]), self.y[idx]
    
class EmotionCNNGRU(nn.Module):
    """cnn+gru model for audio emotion classification"""

    def __init__(self, n_mfcc, n_classes, dropout_rate=DROPOUT_RATE):
        """
        args:
            n_mfcc: number of mfcc features
            n_classes: number of emotion classes
            dropout_rate: dropout probability
        """
        super(EmotionCNNGRU, self).__init__()

        # cnn layers
        self.conv1 = nn.Conv1d(n_mfcc, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # gru layer
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2,
                          batch_first=True, bidirectional=True, dropout=DROPOUT_RATE)

        # fully connected layers
        self.fc1 = nn.Linear(256, 128)  # 128 = 64*2 (bidirectional)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        """forward pass"""
        # cnn layers (batch_size, n_mfcc, time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = self.pool3(x)

        # reshape for gru (batch_size, time, features)
        x = x.permute(0, 2, 1)

        # gru layer
        x, _ = self.gru(x)
        x = self.dropout(x)

        # use only the last output
        x = x[:, -1, :]

        # fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.fc3(x)

        return x