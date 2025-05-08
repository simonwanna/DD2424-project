import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


DATA_DIR = 'data'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalise according to https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])  # (already preprocessed spatially before)

train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset   = ImageFolder(root=os.path.join(DATA_DIR, 'val'),   transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def unfreeze_last_blocks(model, l):
    assert 0 <= l <= 4, "l must be in [0, 4]"
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for block in layers[-l:]:
        for param in block.parameters():
            param.requires_grad = True


for epoch in range(1, NUM_EPOCHS + 1):
    # Model setup for each epoch
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 37)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Freeze the layers except the fully connected (fc) layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Gradually unfreeze layers over epochs
    unfreeze_last_blocks(model, min(epoch, 4))  # Unfreeze up to 'epoch' number of layers
    
    model = model.to(DEVICE)

    # Use a lower learning rate for the unfreezing stages
    learning_rate = LEARNING_RATE * (0.1 ** (epoch // 3))  # Reduce by a factor every 3 epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)  # only use parameters that require grad


    def run_epoch(loader, phase='train'):
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        return epoch_loss, epoch_acc.item()


    # Run each epoch
    train_loss, train_acc = run_epoch(train_loader, 'train')
    val_loss, val_acc = run_epoch(val_loader, 'val')

    print(f"Epoch {epoch}/{NUM_EPOCHS}  "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'checkpoints/best_resnet18_finetuned.pth')

print("\033[92m" + f"Training complete. Best val acc: {best_val_acc: .4f}" + "\033[0m")
