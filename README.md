# winnipeg: cnngru for audio emotion classification

a compact cnn + bidirectional gru over mfccs (with Δ, Δ²) classifying speech into 7 emotions. standardized 3s audio at 22,050 hz; achieved 73.63% accuracy and topped the kaggle board on 2025-04-19.

training uses class-weighted loss, dropout + l2, batch norm, early stopping, and reduce-lr-on-plateau; inference mirrors preprocessing and writes `submission.csv`.

```bash
# install
pip install torch librosa numpy pandas tqdm

# train
python train.py --config config.py

# infer
python inference.py --ckpt best_emotion_model.pth
