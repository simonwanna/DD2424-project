import os
import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# random seed for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = 'data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transforms(augment: bool):
    # Normalise according to https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18
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


def unfreeze_last_blocks(model, l, finetune_bn_layers=True):
    assert 0 <= l <= 4, "l must be in [0, 4]"
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
    for layer in layers[0:l+1]:
        for name, param in layer.named_parameters():
            if not finetune_bn_layers and 'bn' in name:
                param.requires_grad = False
            else:
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


def main(args):
    base_lr = args.lr
    layer_lrs = {
        'layer1': base_lr * 0.25,
        'layer2': base_lr * 0.5,
        'layer3': base_lr * 1.0,
        'layer4': base_lr * 1.0,
        'fc':     base_lr * 10.0,
    }

    train_transform, val_transform = get_transforms(augment=args.augment)

    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset   = ImageFolder(root=os.path.join(DATA_DIR, 'val'),   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    os.makedirs('checkpoints', exist_ok=True)

    for l in range(1, 4):  # progressively unfreeze more layers
        print(f"\033[93mUnfreezing {l} layers...\033[0m")
        best_val_acc = 0.0

        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
        # weights = ResNet18_Weights.DEFAULT
        # model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 37)
        model = freeze_all_layers(model)
        model = model.to(DEVICE)

        model = unfreeze_last_blocks(model, l, finetune_bn_layers=args.bn_layers)

        criterion = nn.CrossEntropyLoss()
        
        weight_decay = 1e-3 if args.L2_reg else 0.0

        if args.layer_wise_lr:
            param_groups = [
                {'params': module.parameters(), 'lr': layer_lrs[name]}
                for name, module in model.named_children()
                if name in layer_lrs and any(p.requires_grad for p in module.parameters())
            ]
            
            optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=weight_decay)

        for epoch in range(1, args.num_epochs + 1):
            train_loss, train_acc = run_epoch(train_loader, 'train', model, criterion, optimizer)
            val_loss, val_acc = run_epoch(val_loader, 'val', model, criterion)

            print(f"Epoch {epoch}/{args.num_epochs}  "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'checkpoints/best_breed_s1_l={l}_AUG:{args.augment}_LWLR:{args.layer_wise_lr}_L2:{args.L2_reg}_BN:{args.bn_layers}.pth')
                print(f"\033[92mSaved best model for l={l} with val acc: {val_acc:.4f}\033[0m")

    print(f"\033[92mTraining complete. Best val acc: {best_val_acc:.4f}\033[0m")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose data augmentation mode.")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Base learning rate for the model.")
    parser.add_argument('--num_epochs', type=int, default=10,
                        help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training and validation.")
    parser.add_argument('--augment', action='store_true',
                        help="Use this flag to apply data augmentation to training images.")
    parser.add_argument('--layer_wise_lr', action='store_true',
                        help="Use this flag to apply layer-wise learning rates.")
    parser.add_argument('--L2_reg', action='store_true',
                        help="Use this flag to apply L2 regularization.")
    parser.add_argument('--bn_layers', action='store_false',
                        help="Use this flag to avoid finetuning batch normalization layers.")
    parser.add_argument('--imblanced', action='store_true',
                        help="Use this flag to use imbalanced dataset.")
    
    args = parser.parse_args()
    
    main(args)
