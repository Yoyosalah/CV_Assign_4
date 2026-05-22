import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
















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