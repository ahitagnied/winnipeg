# audio parameters
SAMPLE_RATE = 22050  # common sample rate (hz)
DURATION = 3  # fixed length in seconds for all audio
MAX_LENGTH = SAMPLE_RATE * DURATION
N_MFCC = 40  # number of mfcc features
N_FFT = 2048  # fft window size
HOP_LENGTH = 512  # hop length for fft
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4  # l2 regularization
DROPOUT_RATE = 0.5   # dropout rate
PATIENCE = 5  # early stopping patience
DATA_PATH = "data/dataset/" # path to dataset