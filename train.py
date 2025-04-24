import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from prepare import *
from gru import *
from config import *

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

def train():
    """main training function"""
    
    features, labels, classes, class_weights = load_data()
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=42, stratify=labels
    )

    print(f"training set: {X_train.shape[0]} samples")
    print(f"testing set: {X_test.shape[0]} samples")

    # create datasets and dataloaders
    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)

    # sampler
    class_sample_count = np.bincount(y_train)
    weights = 1.0 / class_sample_count
    samples_weights = weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # initialize model
    model = EmotionCNNGRU(N_MFCC*3, len(classes)).to(device)  # *3 for delta features

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.4, patience=2, verbose=True, min_lr=1e-9
    )

    # early stopping setup
    early_stopping_counter = 0
    best_val_loss = float('inf')

    # training loop
    print("starting training...")
    best_accuracy = 0.0
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(EPOCHS):
        # training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS} [train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward pass and optimize
            loss.backward()
            optimizer.step()

            # statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"epoch {epoch+1}/{EPOCHS} [val]"):
                inputs, targets = inputs.to(device), targets.to(device)

                # forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

        # update scheduler
        scheduler.step(test_loss)

        # save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'model.pth')
            print(f"model saved with accuracy: {best_accuracy:.2f}%")

        print(f"epoch {epoch+1}/{EPOCHS}: "
              f"train loss={train_loss:.4f}, train acc={train_accuracy:.2f}%, "
              f"val loss={test_loss:.4f}, val acc={test_accuracy:.2f}%")

        # early stopping check
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"early stopping counter: {early_stopping_counter}/{PATIENCE}")

            if early_stopping_counter >= PATIENCE:
                print(f"early stopping triggered after {epoch+1} epochs")
                break

    # plot training and validation curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='train')
    plt.plot(test_accs, label='val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print(f"best validation accuracy: {best_accuracy:.2f}%")
    print("training completed!")

if __name__ == "__main__":
    train()