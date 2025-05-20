import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse

import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import datetime

# random seed for reproducibility
# torch.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

DATA_DIR = 'data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN_VALS = [0.485, 0.456, 0.406]
STD_VALS = [0.229, 0.224, 0.225]

def get_transforms(augment: bool):
    # Normalise according to https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18
    if augment:
        print("Using **augmented** training transforms.")
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.7),
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


def unnormalize_image(tensor_image):
    """Unnormalizes a tensor image for display."""
    unnormalized = tensor_image.clone()
    mean = torch.tensor(MEAN_VALS).view(3, 1, 1)
    std = torch.tensor(STD_VALS).view(3, 1, 1)
    unnormalized.mul_(std).add_(mean)
    unnormalized = unnormalized.numpy().transpose((1, 2, 0))
    return np.clip(unnormalized, 0, 1)


def visualize_augmentations_on_images(args):
    print("Visualizing augmentations...")

    train_image_folder = os.path.join(DATA_DIR, 'train')
    image_paths = []
    num_images_to_find = 4

    for class_name in os.listdir(train_image_folder):
        class_folder_path = os.path.join(train_image_folder, class_name)
        if os.path.isdir(class_folder_path):
            for img_file in os.listdir(class_folder_path):
                if len(image_paths) < num_images_to_find:
                    image_paths.append(os.path.join(class_folder_path, img_file))
                else:
                    break
        if len(image_paths) >= num_images_to_find:
            break
            
    if not image_paths:
        print(f"No images found in {train_image_folder} to visualize.")
        return

    before_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_VALS, std=STD_VALS)
    ])


    after_transform, _ = get_transforms(augment=args.augment) 

    num_to_show = len(image_paths)
    fig, axs = plt.subplots(num_to_show, 2, figsize=(10, 2.5 * num_to_show))
    if num_to_show == 1: # Ensure axs is 2D for consistent indexing
        axs = np.array([axs]) 

    fig.suptitle(f"Augmentation View (Augment Flag: {'ON' if args.augment else 'OFF'})", fontsize=14)

    for i, img_path in enumerate(image_paths):
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Could not open image {img_path}: {e}")
            if num_to_show == 1:
                 axs[0].axis('off')
                 axs[1].axis('off')
            else:
                axs[i, 0].axis('off')
                axs[i, 1].axis('off')
            continue

        img_tensor_before = before_transform(pil_img.copy())
        axs[i, 0].imshow(unnormalize_image(img_tensor_before))
        axs[i, 0].set_title("Before (Resized, Normalized)")
        axs[i, 0].axis('off')

        temp_after_transform = transforms.Compose([
            transforms.Resize((224,224)),
            *after_transform.transforms
        ])
        img_tensor_after = temp_after_transform(pil_img.copy())
        
        axs[i, 1].imshow(unnormalize_image(img_tensor_after))
        title_after = "After (Augmented)" if args.augment else "After (Standard)"
        axs[i, 1].set_title(title_after)
        axs[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    # save the figure
    plt.savefig('augmented_visualization.png', dpi=300)


# Load pre-trained ResNet34 and modify final layer
def setup_model():
    # weights = ResNet18_Weights.DEFAULT
    # model = resnet18(weights=weights)
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    # weights = ResNet50_Weights.DEFAULT
    # model = resnet50(weights=weights)
    
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
    # viz = True
    # if viz:
    #     visualize_augmentations_on_images(args)
    #     return  # Exit after visualization
    
    base_lr = args.lr
    layer_lrs = {
        'layer1': base_lr * 0.25,
        'layer2': base_lr * 0.5,
        'layer3': base_lr * 0.5,
        'layer4': base_lr * 1.0,
        'fc':     base_lr * 8.0,
    }

    train_transform, val_transform = get_transforms(augment=args.augment)

    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset   = ImageFolder(root=os.path.join(DATA_DIR, 'val'),   transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    os.makedirs('checkpoints', exist_ok=True)
    
    weight_decay = 0.01 if args.L2_reg else 0.0
    best_val_acc = 0.0
    model, criterion = setup_model()
    
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at {start}")
    for l in range(0, 3):  # Unfreeze more layers progressively, change upper bound for deeper unfreeze

        model = unfreeze_last_blocks(model, l, finetune_bn_layers=args.bn_layers)
        
        if args.layer_wise_lr:
            print(f"Using **layer-wise learning rates**")
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
                torch.save(model.state_dict(), f'checkpoints/best_breed_s2_l={l}_AUG:{args.augment}_LWLR:{args.layer_wise_lr}_L2:{args.L2_reg}_BN:{args.bn_layers}.pth')

    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_diff = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    print(f"End time: {end}")
    print(f"Time taken for l={l}: {time_diff}")
    
    print("\033[92m" + f"Training complete. Best val acc: {best_val_acc:.4f}" + "\033[0m")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Choose data augmentation mode.")
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="Base learning rate for the model.")
    parser.add_argument('--num_epochs', type=int, default=6,
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

    
