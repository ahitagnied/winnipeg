import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Winnipeg, EmotionDataset
from collections import Counter
import numpy as np

# reduced hyperparameters
batch_size = 16  # reduced batch size
epochs = 10  # reduced number of epochs
lr = 0.001
early_stopping_patience = 3  # stop training if validation doesn't improve

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# dataset path
data_dir = '/data/dataset/'

# load dataset
dataset = EmotionDataset(root_dir=data_dir)
print(f"dataset loaded with {len(dataset)} samples")
print(f"emotions: {dataset.emotion_to_idx}")

# create balanced dataset with exactly 250 samples per emotion class
samples_per_class = 500
total_samples = samples_per_class * len(dataset.emotion_to_idx)  # 2000 total samples

# important optimization: access the labels directly from the dataset object
print("creating balanced dataset...")
all_indices = list(range(len(dataset)))
all_labels = [dataset.emotion_to_idx[dataset.emotion[i]] for i in all_indices]  # use internal list

# create stratified subset
balanced_indices = []

# for each emotion class
for emotion_idx in range(len(dataset.emotion_to_idx)):
    # get indices of samples from this class
    class_indices = [i for i, label in enumerate(all_labels) if label == emotion_idx]
    
    # randomly select exactly 250 samples (or fewer if not enough)
    if len(class_indices) >= samples_per_class:
        np.random.seed(42)  # for reproducibility
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
    else:
        # if we have fewer than 250 samples, take all available
        selected_indices = class_indices
        print(f"warning: only {len(class_indices)} samples available for class {emotion_idx}")
    
    balanced_indices.extend(selected_indices)

# create balanced dataset
balanced_dataset = Subset(dataset, balanced_indices)
print(f"created balanced dataset with {len(balanced_dataset)} samples")

# verify class distribution
balanced_labels = [dataset.emotion_to_idx[dataset.emotion[i]] for i in balanced_indices]
print(f"class distribution: {Counter(balanced_labels)}")

# split into train/validation
torch.manual_seed(42)
train_size = int(0.8 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size

train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])
print(f"train set: {len(train_dataset)} samples")
print(f"validation set: {len(val_dataset)} samples")

# create dataloaders with reduced workers to avoid bottlenecks
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# initialize model
model = Winnipeg(
    nmels=128,
    nemo=len(dataset.emotion_to_idx),
    seq_len=int(3.0 * 16000 / 512)
).to(device)

# loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # using adamw with weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True
)

# history tracking
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
early_stopping_counter = 0

# training loop
print("starting training...")
for epoch in range(epochs):
    # training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm.tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs} - training"):
        # get data
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        # forward pass with gradient accumulation to simulate larger batch
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
          outputs = model(spectrograms)
          loss = criterion(outputs, labels)
        
        # backward pass
        loss.backward()
        
        # gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # statistics
        train_loss += loss.item() * spectrograms.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # calculate metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    # validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc=f"epoch {epoch+1}/{epochs} - validation"):
            # get data
            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # statistics
            val_loss += loss.item() * spectrograms.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # calculate metrics
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total
    
    # update history
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    # print results
    print(f"epoch {epoch+1}/{epochs}: train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.4f}, val_loss={epoch_val_loss:.4f}, val_acc={epoch_val_acc:.4f}")
    
    # update learning rate based on validation loss
    scheduler.step(epoch_val_loss)
    
    # save best model and implement early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'best_emotion_model.pt')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    # early stopping check
    if early_stopping_counter >= early_stopping_patience:
        print(f"early stopping triggered after {epoch+1} epochs")
        break

# plot training history
plt.figure(figsize=(12, 5))

# plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='train')
plt.plot(history['val_acc'], label='validation')
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

# plot loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.show()

# load best model for final evaluation
model.load_state_dict(torch.load('best_emotion_model.pt'))
print("training completed!")