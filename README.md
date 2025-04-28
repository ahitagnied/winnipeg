# 🍁 Winnipeg: CNNGRU for Audio Emotion Classiciation

This project implements a deep learning system for classifying audio recordings into different emotional categories (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised). The system analyzes speech audio and predicts the emotional state of the speaker based on acoustic patterns.

The CNN+GRU model achieved an accuracy of 73.63% which is close to human performance in classifying such datasets (74% acc/ to Pentari et al.). [1st in Kaggle Leaderboard as of 19th April, 2025]

<img src='assets/loss.png' width:300px>

## 🐉 Model Architecture

The model uses a hybrid CNN+GRU architecture:

- Convolutional Neural Network (CNN): Extracts spectral features from the audio MFCC representations
- Gated Recurrent Unit (GRU): Captures temporal dynamics in the audio data
- Bidirectional processing: Analyzes the audio sequence in both forward and backward directions
- Regularization: Employs dropout and L2 regularization to prevent overfitting

## 📊 Data Preprocessing

Audio samples are standardized to 3s length (66150 samples at 22050Hz). Features include 40 MFCCs extracted with FFT window size=2048 and hop length=512, plus their first and second derivatives (Δ, Δ²), resulting in 120-dimensional feature vectors. Batch normalization is applied for training stability.

## 🚀 Training and Inference

Training employs early stopping (patience=10) with ReduceLROnPlateau scheduling (factor=0.5). Class imbalance is addressed via weighted loss function, with weights inversely proportional to class frequency. Model checkpoints are saved based on validation accuracy maximization.

Pre-trained model weights are loaded from 'best_emotion_model.pth'. Test audio undergoes identical preprocessing (MFCC+Δ+Δ²). Emotion predictions are generated in a single forward pass and exported to submission.csv with [filename, emotion] format.

## 📁 Project Structure
```bash
config.py      # Configuration parameters for audio processing and model training  
prepare.py     # Data preparation and feature extraction functions  
model.py       # Model architecture and dataset class definitions  
train.py       # Training loop and model optimization  
inference.py   # Prediction generation for test data  
```

# 📦 Requirements

Make sure the following packages are installed:

- PyTorch  
- Librosa  
- NumPy  
- Pandas  
- tqdm  

You can install them using:

```bash
pip install torch librosa numpy pandas tqdm
```
# 🏺 References

```bibtex
@article{Pentari2024graphEmotion,
  title={Speech emotion recognition via graph-based representations},
  author={Pentari, A. and Kafentzis, G. and Tsiknakis, M.},
  journal={Scientific Reports},
  volume={14},
  pages={4484},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-024-52989-2},
  url={https://doi.org/10.1038/s41598-024-52989-2}
}
```
```bibtex
@misc{Chung2014GRUvsLSTM,
  title={Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling},
  author={Chung, Junyoung and Gulcehre, Caglar and Cho, KyungHyun and Bengio, Yoshua},
  note={Presented at NIPS 2014 Deep Learning and Representation Learning Workshop},
  year={2014},
  howpublished={\url{https://doi.org/10.48550/arXiv.1412.3555}},
  archivePrefix={arXiv},
  eprint={1412.3555}
}
```
