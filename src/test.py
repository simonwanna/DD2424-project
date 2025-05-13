import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchmetrics


DATA_DIR = 'data'
MODEL_PATH = 'checkpoints/best_resnet18_finetuned_breed_s1_l=1.pth'
BATCH_SIZE = 32
NUM_CLASSES = 37
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root=os.path.join(
    DATA_DIR, 'test'), transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# model setup
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)


def run_test_epoch(loader):
    accuracy_metric.reset()
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, labels)

    return total_loss, num_samples


if __name__ == "__main__":
    total_loss, num_samples = run_test_epoch(test_loader)
    avg_loss = total_loss / num_samples
    accuracy = accuracy_metric.compute().item()

    print("\033[92m" + f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4%}" + "\033[0m")
