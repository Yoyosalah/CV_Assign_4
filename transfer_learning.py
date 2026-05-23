# transferlearnin.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os

from main import dataloader, train_one_epoch, evaluate, print_metrics

def get_densenet_tl_model(num_classes, device, pretrained=True):
    if pretrained:
        model_tl = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).to(device)
  
        for param in model_tl.parameters():
            param.requires_grad = False

        for param in model_tl.features.denseblock4.parameters():
            param.requires_grad = True
    else:
        model_tl = models.densenet121(weights=None).to(device)

    num_features = model_tl.classifier.in_features
    model_tl.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_features, num_classes)
    ).to(device)
    
    return model_tl

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 4
    EPOCHS = 30
    BATCH_SIZE = 32
    WEIGHT_DECAY = 5e-4
    DATA_DIR = "Dataset/Master Folder"

    print(f"Device: {DEVICE}")
    print("\n── Loading Data ───────────────────────────────")
    train_loader, val_loader, test_loader, classes = dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=224
    )

    print("\n── Calculating Inverse Class Frequency ────────")
    train_labels = train_loader.dataset.targets 

    weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(train_labels), 
        y=train_labels
    )
    print(f"Calculated Class Weights: {weights}")
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    print(f"\n── Initializing Pre-trained DenseNet121 ───────")
    
    model_tl = get_densenet_tl_model(NUM_CLASSES, DEVICE, pretrained=True)
    
    total_params = sum(p.numel() for p in model_tl.parameters())
    trainable_params_count = sum(p.numel() for p in model_tl.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params_count:,}")

    #Setup Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model_tl.parameters())
    optimizer_tl = optim.Adam(trainable_params, lr=1e-4, weight_decay=WEIGHT_DECAY)

    #Setup the Smart Scheduler
    scheduler_tl = optim.lr_scheduler.ReduceLROnPlateau(optimizer_tl, mode='max', factor=0.5, patience=3)

    #Training Loop
    print("\n── Starting Transfer Learning Training ────────")
    best_acc_tl = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model_tl, train_loader, criterion, optimizer_tl, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model_tl, val_loader, criterion, DEVICE)
        
        # Pass validation accuracy to the scheduler
        scheduler_tl.step(val_acc)
        current_lr = optimizer_tl.param_groups[0]['lr']

        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"LR: {current_lr:.6f}  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")
              
        if val_acc > best_acc_tl:
            best_acc_tl = val_acc
            torch.save(model_tl.state_dict(), 'DenseNet121_TransferLearning_best.pth')

    # Final Evaluation
    print("\n── Final Evaluation (Transfer Learning) ────────")
    model_tl.load_state_dict(torch.load('DenseNet121_TransferLearning_best.pth', map_location=DEVICE))
    _, test_acc, preds, labels = evaluate(model_tl, test_loader, criterion, DEVICE)
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    cm = print_metrics(labels, preds, classes)

if __name__ == '__main__':
    main()