import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Winnipeg, EmotionDataset

# dataset path
data_dir = '/content/files/data'

# hyperparameters
batch_size = 32
epochs = 20
lr = 0.001

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# dataset preparation
dataset = EmotionDataset(root_dir=data_dir)
print(f"dataset loaded with {len(dataset)} samples")
print(f"emotions: {dataset.emotion_to_idx}")

# split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# set reproducibility
torch.manual_seed(42)

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

print(f"train set: {len(train_dataset)} samples")
print(f"validation set: {len(val_dataset)} samples")
print(f"test set: {len(test_dataset)} samples")

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# initialize model - make sure the dimensions match what your data expects
model = Winnipeg(
    nmels=128,  # adjust based on your spectrogram parameters
    nemo=len(dataset.emotion_to_idx),
    seq_len=int(3.0 * 16000 / 512)  # adjust based on your audio parameters
).to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# history tracking
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# training loop
print("starting training...")
for epoch in range(epochs):
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs} - training"):
        # get data - ensure dimensions match what the model expects
        spectrograms = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        # stats
        train_loss += loss.item() * spectrograms.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # calculate epoch metrics
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    
    # VALIDATION PHASE
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"epoch {epoch+1}/{epochs} - validation"):
            # get data - ensure dimensions match what the model expects
            spectrograms = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            # forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            # stats
            val_loss += loss.item() * spectrograms.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # calculate validation metrics
    epoch_val_loss = val_loss / val_total
    epoch_val_acc = val_correct / val_total
    
    # update history
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    # print epoch results
    print(f"epoch {epoch+1}/{epochs}: train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.4f}, val_loss={epoch_val_loss:.4f}, val_acc={epoch_val_acc:.4f}")

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

# save the trained model
torch.save(model.state_dict(), 'emotion_model.pt')
print("training completed!")