import os
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

DATA_DIR = 'data'
BATCH_SIZE = 64
NUM_EPOCHS = 8
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transforms(augment: bool):
    if augment:
        print("Using **augmented** training transforms.")
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        print("Using **non-augmented** (normal) training transforms.")
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def freeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

def unfreeze_last_blocks(model, l):
    assert 0 <= l <= 4, "l must be in [0, 4]"
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for block in layers[0:l+1]:
        for param in block.parameters():
            param.requires_grad = True
    return model

def run_epoch(loader, phase='train', model=None, criterion=None, optimizer=None):
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
    parser = argparse.ArgumentParser(description="Choose data augmentation mode.")
    parser.add_argument('--augment', action='store_true',
                        help="Use this flag to apply data augmentation to training images.")
    args = parser.parse_args()

    train_transform, val_transform = get_transforms(augment=args.augment)

    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset   = ImageFolder(root=os.path.join(DATA_DIR, 'val'),   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)

    os.makedirs('checkpoints', exist_ok=True)

    for l in range(1, 4):  # progressively unfreeze more layers
        print(f"\033[93mUnfreezing {l} layers...\033[0m")
        best_val_acc = 0.0

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 37)
        model = freeze_all_layers(model)
        model = model.to(DEVICE)

        model = unfreeze_last_blocks(model, l)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = run_epoch(train_loader, 'train', model, criterion, optimizer)
            val_loss, val_acc = run_epoch(val_loader, 'val', model, criterion)

            print(f"Epoch {epoch}/{NUM_EPOCHS}  "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'checkpoints/best_resnet18_finetuned_breed_s1_l={l}.pth')
                print(f"\033[92mSaved best model for l={l} with val acc: {val_acc:.4f}\033[0m")

    print(f"\033[92mTraining complete. Best val acc: {best_val_acc:.4f}\033[0m")
