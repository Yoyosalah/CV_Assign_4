import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os


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


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], 1) # instead of adding concatenate


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels) # transition layers are used 4 shrinking (via pooling) and reducing the channel count (via 1x1 conv)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)  #
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity #Add the original input back BEFORE the final ReLU
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 has [2, 2, 2, 2] blocks per layer
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # Global Average Pooling flattens the image flexibly regardless of size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, num_classes=4, growth_rate=32):
        super(DenseNet121, self).__init__()

        # DenseNet121 has exactly this many layers in its 4 Dense Blocks
        block_config = [6, 12, 24, 16]
        num_channels = 64

        # 1. Initial Convolution Block
        self.features = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Build the Dense Blocks and Transition Layers
        for i, num_layers in enumerate(block_config):
            # Build a Dense Block
            block_layers = []
            for _ in range(num_layers):
                block_layers.append(DenseLayer(num_channels, growth_rate))
                num_channels += growth_rate  # Channels grow by 32 every single layer

            self.features.add_module(f'denseblock_{i + 1}', nn.Sequential(*block_layers))

            # Add a Transition Layer after every block EXCEPT the last one
            if i != len(block_config) - 1:
                out_channels = num_channels // 2  # Compress channels by 50%
                self.features.add_module(f'transition_{i + 1}', TransitionLayer(num_channels, out_channels))
                num_channels = out_channels

        # 3. Final BatchNorm and ReLU
        self.features.add_module('norm5', nn.BatchNorm2d(num_channels))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        # 4. Global Average Pooling and Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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


def dataloader(data_dir, batch_size=32, img_size=224, train_split=0.7, val_split=0.15, test_split=0.15):
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {train_split + val_split + test_split}"

    train_transforms, test_transforms = data_transformer(img_size)

    # Load entire dataset from root directory
    print(f"Loading dataset from: {data_dir}")
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    print(f"Classes found: {full_dataset.classes}")
    print(f"Class-to-Index map: {full_dataset.class_to_idx}")
    print(f"Total images: {len(full_dataset)}")

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size  # Remainder goes to test

    print(f"\nSplitting dataset:")
    print(f"  Training:   {train_size} images ({train_split * 100:.1f}%)")
    print(f"  Validation: {val_size} images ({val_split * 100:.1f}%)")
    print(f"  Testing:    {test_size} images ({test_split * 100:.1f}%)")

    # Split dataset with fixed random seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply test transforms (no augmentation) to val and test sets
    # We need to create new dataset objects with the test transform
    val_dataset_no_aug = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    test_dataset_no_aug = datasets.ImageFolder(root=data_dir, transform=test_transforms)

    # Get the same indices from the split
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # Create subset datasets with test transforms
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset_no_aug, val_indices)
    test_dataset = Subset(test_dataset_no_aug, test_indices)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.classes


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 4
    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 0.001
    WEIGHT_DECAY = 5e-4
    PATIENCE = 5  # Early stopping patience
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, classes = dataloader(
        data_dir="Dataset",
        batch_size=BATCH_SIZE,
        img_size=224,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )

    #VGG model
    print(f"\n  VGG Model")
    model = VGG19(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_loss, patience_counter = 0.0, float('inf'), 0
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

        if val_loss < best_loss - 0.001:
            best_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n── Final Evaluation (best checkpoint) ────────")
    model.load_state_dict(torch.load('vgg19_best.pth'))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.2f}%")
    cm = print_metrics(labels, preds, classes)

    #Resnet model
    print(f"\n  ResNet Model")
    model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_loss, patience_counter = 0.0, float('inf'), 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'ResNet_best.pth')

        if val_loss < best_loss - 0.001:
            best_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n── Final Evaluation (best checkpoint) ────────")
    model.load_state_dict(torch.load('ResNet_best.pth'))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.2f}%")
    cm = print_metrics(labels, preds, classes)

    #DenseNet model
    model = DenseNet121(num_classes=NUM_CLASSES).to(DEVICE)
    print(f"\n  DenseNet Model")
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_loss, patience_counter = 0.0, float('inf'), 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'DenseNet_best.pth')

        if val_loss < best_loss - 0.001:
            best_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n── Final Evaluation (best checkpoint) ────────")
    model.load_state_dict(torch.load('DenseNet_best.pth'))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.2f}%")
    cm = print_metrics(labels, preds, classes)



if __name__ == '__main__':
    main()