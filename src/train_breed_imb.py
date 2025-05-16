import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1),
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
    for param in model.fc.parameters():  # unfreeze final layer
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


# Load pre-trained ResNet18 and modify final layer
def setup_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 37)
    model = freeze_all_layers(model)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    return model, criterion


def run_epoch(loader, phase='train', model=None, criterion=None, optimizer=None):
    model.train() if phase == 'train' else model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    all_preds_list = []
    all_labels_list = []

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
            else:
                all_preds_list.append(preds.cpu())
                all_labels_list.append(labels.cpu())

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = running_corrects.double() / total_samples if total_samples > 0 else 0
    
    if phase != 'train':
        if total_samples > 0:
            all_preds_tensor = torch.cat(all_preds_list)
            all_labels_tensor = torch.cat(all_labels_list)
        else:
            all_preds_tensor = torch.empty(0, dtype=torch.long)
            all_labels_tensor = torch.empty(0, dtype=torch.long)
        return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc, all_preds_tensor, all_labels_tensor
    else:
        return epoch_loss, epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, normalize='true', output_filename='confusion_matrix_heatmap.png'):
    """
    Computes and plots a confusion matrix heatmap.
    Normalization 'true' gives per-class accuracy (recall) on the diagonal.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        print("No data to plot confusion matrix.")
        return

    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    if not np.all(unique_labels < len(class_names)):
        print(f"Warning: Labels in y_true/y_pred exceed number of class_names ({len(class_names)}). Clamping or check data.")
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    
    if normalize == 'true':
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        data_to_plot = cm_normalized
        fmt = '.2f'
        title = 'Normalized Confusion Matrix (Per-Class Accuracy / Recall)'
    else:
        data_to_plot = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(max(10, int(len(class_names)*0.5)), max(8, int(len(class_names)*0.4))))
    sns.heatmap(data_to_plot, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 8})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Confusion matrix heatmap saved to {output_filename}")


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

    # TODO: if imbalanced:, use only 20% of the images from any class that is of cat
    # should be able to use weighted cross entropy to counteract for the imbalance
    if args.imbalanced: # Corrected typo: imblanced -> imbalanced
        print("Using **imbalanced** dataset: reducing cat breed samples in training set.")
        
        CAT_BREED_NAMES = [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 
            'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 
            'Siamese', 'Sphynx'
        ]

        # Get class to index mapping and identify cat breed indices
        class_to_idx = train_dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()} # For printing class names
        cat_breed_indices = {class_to_idx[breed] for breed in CAT_BREED_NAMES if breed in class_to_idx}

        original_samples = train_dataset.samples
        new_samples = []
        
        # Group samples by class
        samples_by_class = {}
        for path, class_idx in original_samples:
            if class_idx not in samples_by_class:
                samples_by_class[class_idx] = []
            samples_by_class[class_idx].append((path, class_idx))

        print("\n--- Sample Counts Before Reduction ---")
        for class_idx, class_samples in samples_by_class.items():
            class_name = idx_to_class[class_idx]
            print(f"Class '{class_name}' (ID: {class_idx}): {len(class_samples)} samples")

        for class_idx, class_samples in samples_by_class.items():
            class_name = train_dataset.classes[class_idx]
            if class_idx in cat_breed_indices:
                random.shuffle(class_samples) # Shuffle for random selection
                num_to_keep = int(len(class_samples) * 0.20)
                new_samples.extend(class_samples[:num_to_keep])
                # This print statement is already good for individual reduction logging
                # print(f"  Reduced class '{class_name}' (cat breed) from {len(class_samples)} to {num_to_keep} samples.")
            else:
                new_samples.extend(class_samples) # Keep all samples for non-cat (dog) breeds
                # print(f"  Kept all {len(class_samples)} samples for class '{class_name}'.")


        if not new_samples:
            print("Warning: No samples left after filtering. Check your CAT_BREED_NAMES and dataset structure.")
        
        train_dataset.samples = new_samples
        train_dataset.imgs = new_samples
        train_dataset.targets = [s[1] for s in new_samples]
        
        print(f"\nTotal training samples after imbalance processing: {len(train_dataset.samples)}")

        print("\n--- Sample Counts After Reduction ---")
        counts_after = {}
        for _, class_idx in new_samples:
            counts_after[class_idx] = counts_after.get(class_idx, 0) + 1
        
        for class_idx in sorted(counts_after.keys()):
            class_name = idx_to_class[class_idx]
            original_count = len(samples_by_class.get(class_idx, []))
            new_count = counts_after[class_idx]
            reduction_info = ""
            if class_idx in cat_breed_indices:
                reduction_info = f" (Reduced from {original_count})"
            print(f"Class '{class_name}' (ID: {class_idx}): {new_count} samples{reduction_info}")

        if args.weighted_loss:
            # Placeholder for weighted loss implementation
            print("Weighted loss to be implemented.")
            pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True,  num_workers=4)

    os.makedirs('checkpoints', exist_ok=True)
    
    weight_decay = 1e-4 if args.L2_reg else 0.0
    best_val_acc = 0.0
    model, criterion = setup_model()
    for l in range(0, 3):  # Unfreeze more layers progressively, change upper bound for deeper unfreeze
        model = unfreeze_last_blocks(model, l, finetune_bn_layers=args.bn_layers)
        
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
            # val_loss, val_acc = run_epoch(val_loader, 'val', model, criterion)
            val_loss, val_acc, val_preds, val_labels = run_epoch(val_loader,   'val',   model, criterion)
            print(f"Epoch {epoch}/{args.num_epochs}  "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'checkpoints/best_breed_s2_l={l}_AUG:{args.augment}_LWLR:{args.layer_wise_lr}_L2:{args.L2_reg}_BN:{args.bn_layers}.pth')
        
    plot_confusion_matrix_heatmap(val_labels.numpy(), val_preds.numpy(), train_dataset.classes)
    print("\033[92m" + f"Training complete. Best val acc: {best_val_acc:.4f}" + "\033[0m")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Choose data augmentation mode.")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Base learning rate for the model.")
    parser.add_argument('--num_epochs', type=int, default=5,
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
    parser.add_argument('--imbalanced', action='store_true',
                        help="Use this flag to use imbalanced dataset.")
    parser.add_argument('--weighted_loss', action='store_true',
                        help="Use this flag to use weighted CE.")
    
    args = parser.parse_args()
    
    main(args)

