import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from main import ResNet18, data_transformer, evaluate, print_metrics, VGG19,DenseNet121


def evaluate_saved_model(model_path, model_class, data_dir, img_size=224):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, test_transforms = data_transformer(img_size)  # Reuse your existing transformer
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = model_class(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nModel: {model_path}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print_metrics(labels, preds, test_dataset.classes)


if __name__ == '__main__':
    evaluate_saved_model('vgg19_best.pth', VGG19, "Dataset/Master Folder")
    evaluate_saved_model('ResNet_best.pth', ResNet18, "Dataset/Master Folder")
    evaluate_saved_model('DenseNet_best.pth', DenseNet121, "Dataset/Master Folder")
