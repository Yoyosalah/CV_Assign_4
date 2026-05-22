import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


def data_transformer(img_size):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, test_transforms


def dataloader(data_dir, batch_size=32, img_size=224):
    train_transforms, test_transforms = data_transformer(img_size)

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Classes found: {train_dataset.classes}")
    print(f"Class-to-Index map: {train_dataset.class_to_idx}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Testing images: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, train_dataset.classes


class VGG19(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            self._make_block(3, 64, num_convs=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(64, 128, num_convs=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(128, 256, num_convs=2),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(256, 512, num_convs=4),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_block(512, 512, num_convs=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #Fully Connected layers
        self.classifier = nn.Sequential(
            # After 5 max pools, a 224x224 image becomes 7x7
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes),
        )

    def _make_block(self, in_channels, out_channels, num_convs):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def print_metrics(y_true, y_pred, class_names):
    print("\n── Classification Report ──────────────────────")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("── Confusion Matrix ───────────────────────────")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return cm


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 4
    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 0.001
    WEIGHT_DECAY = 5e-4
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, classes = dataloader(
        data_dir="Dataset/Master Folder",
        batch_size=BATCH_SIZE,
        img_size=224
    )

    model = VGG19(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'vgg19_best.pth')

    print("\n── Final Evaluation (best checkpoint) ────────")
    model.load_state_dict(torch.load('vgg19_best.pth'))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.2f}%")
    cm = print_metrics(labels, preds, classes)


if __name__ == '__main__':
    main()