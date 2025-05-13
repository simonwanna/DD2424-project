import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = 'data'
BATCH_SIZE = 64
NUM_EPOCHS = 4 # Set to 5 for fewer runs per experiment
LEARNING_RATE = 1e-5
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
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():  # unfreeze final layer
        param.requires_grad = True
    return model

def unfreeze_last_blocks(model, l):
    assert 0 <= l <= 4, "l must be in [0, 4]"
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for block in layers[0:l+1]:
        for param in block.parameters():
            param.requires_grad = True
    return model

# Load pre-trained ResNet18 and modify final layer
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 37)
model = freeze_all_layers(model)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, phase='train'):
    model.train() if phase == 'train' else model.eval()

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

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    best_val_acc = 0.0
    
    for l in range(0, 3):  # Unfreeze more layers progressively, change upper bound for deeper unfreeze
        if l != 0:
            model = unfreeze_last_blocks(model, l) # Get model
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE) # optimizer for unfrozen model
        
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = run_epoch(train_loader, 'train')
            val_loss, val_acc = run_epoch(val_loader, 'val')
            print(f"Epoch {epoch}/{NUM_EPOCHS}  "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'checkpoints/best_resnet18_finetuned_breed.pth')

    print("\033[92m" + f"Training complete. Best val acc: {best_val_acc:.4f}" + "\033[0m")
