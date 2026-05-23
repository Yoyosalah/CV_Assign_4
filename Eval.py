# eval.py
import torch
import torch.nn as nn
import os

from main import dataloader, evaluate, print_metrics, VGG19, ResNet18, DenseNet121, MobileNet, InceptionV3
from transfer_learning import get_densenet_tl_model

def evaluate_saved_model(model_name, model_path, model, loaders, criterion, device, class_names):
    if not os.path.exists(model_path):
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nModel: {model_name} ({model_path})")

    for split_name, loader in loaders.items():
        _, acc, preds, labels = evaluate(model, loader, criterion, device)
        print(f"{split_name} Accuracy: {acc:.2f}%")
        if split_name == 'Test':
            print_metrics(labels, preds, class_names)

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = "Dataset/Master Folder"
    NUM_CLASSES = 4
    BATCH_SIZE = 32

    train_loader, val_loader, test_loader, classes = dataloader(
        data_dir=DATA_DIR, 
        batch_size=BATCH_SIZE, 
        img_size=224
    )
    
    loaders = {
        'Train': train_loader,
        'Validation': val_loader,
        'Test': test_loader
    }
    
    criterion = nn.CrossEntropyLoss()

    models_list = [
        ("VGG19", 'vgg19_best.pth', VGG19(num_classes=NUM_CLASSES).to(DEVICE)),
        ("ResNet18", 'ResNet_best.pth', ResNet18(num_classes=NUM_CLASSES).to(DEVICE)),
        ("DenseNet121", 'DenseNet_best.pth', DenseNet121(num_classes=NUM_CLASSES).to(DEVICE)),
        ("MobileNet", 'MobileNet_best.pth', MobileNet(num_classes=NUM_CLASSES).to(DEVICE)),
        ("InceptionV3", 'InceptionV3_best.pth', InceptionV3(num_classes=NUM_CLASSES).to(DEVICE)),
        ("DenseNet121 Transfer Learning", 'DenseNet121_TransferLearning_best.pth', get_densenet_tl_model(NUM_CLASSES, DEVICE, pretrained=False))
    ]

    for model_name, model_path, model in models_list:
        evaluate_saved_model(model_name, model_path, model, loaders, criterion, DEVICE, classes)

if __name__ == '__main__':
    main()