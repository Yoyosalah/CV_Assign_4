import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from main import ResNet18, dataloader, evaluate, print_metrics, VGG19, DenseNet121


def evaluate_saved_model(model_path, model_class, data_dir, img_size=224):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, _, test_loader, classes = dataloader(
        data_dir=data_dir,
        batch_size=32,
        img_size=img_size,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )

    model = model_class(num_classes=4).to(DEVICE)

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Successfully loaded model from {model_path}")
    except RuntimeError as e:
        print(f"⚠ Warning: Strict loading failed. Attempting with strict=False...")
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            print(f"Missing keys: {incompatible_keys.missing_keys}")
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

    model.eval()

    criterion = nn.CrossEntropyLoss()
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nModel: {model_path}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print_metrics(labels, preds, classes)


if __name__ == '__main__':
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60 + "\n")

    evaluate_saved_model('vgg19_best.pth', VGG19, "Dataset")
    print("\n" + "=" * 60 + "\n")

    evaluate_saved_model('ResNet_best.pth', ResNet18, "Dataset")
    print("\n" + "=" * 60 + "\n")

    evaluate_saved_model('DenseNet_best.pth', DenseNet121, "Dataset")
