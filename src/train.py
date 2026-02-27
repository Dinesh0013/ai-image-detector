import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Training'):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()           # clear gradients from last step
        outputs = model(images)         # forward pass
        loss = criterion(outputs, labels)  # compute loss
        loss.backward()                 # backpropagation
        optimizer.step()                # update weights

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():             # no gradient computation during evaluation
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), correct / total, auc


def train(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')
    model = model.to(device)

    criterion = nn.BCELoss()          # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']  # L2 regularization
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1   # reduce LR by 10x every 3 epochs
    )

    best_val_auc = 0

    for epoch in range(config['epochs']):
        print(f'\nEpoch {epoch+1}/{config["epochs"]}')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        # Log everything to W&B
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | AUC: {val_auc:.4f}')

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f'experiments/best_model.pth')
            print(f'New best model saved with AUC: {val_auc:.4f}')

    return model