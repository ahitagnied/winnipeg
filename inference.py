import os
import torch
import pandas as pd
from tqdm import tqdm
from gru import EmotionCNNGRU
from config import *
from prepare import *

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

# load saved model and make predictions
def inference():
    """load saved model and generate predictions for test data"""
    # path to test files
    test_path = 'files/data/Test'

    # path to saved model
    model_path = 'results/model.pth'

    # get class mapping from training
    # this matches what was shown in your notebook
    emotion_map = {
        0: 'Angry',
        1: 'Disgusted',
        2: 'Fearful',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Suprised'  # note: spelled as in your code
    }

    # get all test files
    test_files = [f for f in os.listdir(test_path) if f.endswith('.wav')]
    print(f"found {len(test_files)} test files")

    # create dataframe to store results
    results = []

    # load the saved model
    model = EmotionCNNGRU(N_MFCC*3, len(emotion_map)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f"model loaded from {model_path}")

    # make predictions
    with torch.no_grad():
        for file_name in tqdm(test_files):
            file_path = os.path.join(test_path, file_name)

            # extract features
            features = extract_features(file_path)

            if features is not None:
                # convert to tensor and add batch dimension
                features = torch.FloatTensor(features).unsqueeze(0).to(device)

                # make prediction
                outputs = model(features)
                _, predicted = outputs.max(1)

                # get emotion label
                emotion = emotion_map[predicted.item()]

                # add to results
                results.append({
                    'filename': file_name,
                    'emotion': emotion
                })
            else:
                print(f"warning: could not extract features for {file_name}")
                # default to most common class if feature extraction fails
                results.append({
                    'filename': file_name,
                    'emotion': emotion_map[3]  # using happy as default (adjust if needed)
                })

    # create dataframe from results
    df_results = pd.DataFrame(results)

    # save to csv
    df_results.to_csv('results/submission.csv', index=False)
    print(f"predictions saved to submission.csv")

    return df_results

# run the function
predictions = inference()

# display first few predictions
print(predictions.head(10))