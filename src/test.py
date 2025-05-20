import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchmetrics
import argparse
from torchcam.methods import GradCAM
from helpers import apply_gradcam_batch

DATA_DIR = 'data'
BATCH_SIZE = 32
NUM_CLASSES = 37
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CAM_BATCHES = 10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def setup_model(model_path):
    weights = ResNet34_Weights.DEFAULT
    model = resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
    return model.to(DEVICE), nn.CrossEntropyLoss(), torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)


def run_test_epoch(loader, model, criterion, accuracy_metric, class_names=None, args=None):
    accuracy_metric.reset()
    model.eval()
    total_loss = 0.0
    num_samples = 0
    cam_batches = []

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        if args and args.cam and len(cam_batches) < CAM_BATCHES:
            cam_batches.append((inputs.clone(), labels.clone()))

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            accuracy_metric.update(preds, labels)

        total_loss += loss.item() * inputs.size(0)
        num_samples += inputs.size(0)

    if args and args.cam and cam_batches:
        cam = GradCAM(model, target_layer='layer4')
        original = {n: p.requires_grad for n, p in model.named_parameters()}
        try:
            for p in model.parameters():
                p.requires_grad_(True)
            for idx, (x, y) in enumerate(cam_batches):
                out = model(x)
                apply_gradcam_batch(cam, x, y, out, class_names=class_names, mode='incorrect', top_k=4, save_path=f'gradcam_wrong_{idx}.png')
                apply_gradcam_batch(cam, x, y, out, class_names=class_names, mode='correct', top_k=4, save_path=f'gradcam_right_{idx}.png')
        finally:
            for n, p in model.named_parameters():
                p.requires_grad_(original[n])
            cam.remove_hooks()

    return total_loss, num_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test.')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_resnet34.pth')
    parser.add_argument('--cam', action='store_true')
    args = parser.parse_args()

    test_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names_main = test_dataset.classes if args.cam else None

    model, criterion, accuracy_metric = setup_model(args.model_path)
    total_loss, num_samples = run_test_epoch(test_loader, model, criterion, accuracy_metric, class_names_main, args)
    print(f"\033[92mTest Loss: {total_loss / num_samples:.4f}")
    print(f"Test Accuracy: {accuracy_metric.compute().item():.4%}\033[0m")
