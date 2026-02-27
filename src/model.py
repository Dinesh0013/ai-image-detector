import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name='resnet50', pretrained=True, freeze_backbone=False):
    """
    model_name      : 'resnet50' or 'efficientnet'
    pretrained      : use ImageNet pretrained weights
    freeze_backbone : if True, only train the final layer (faster but less accurate)
                      if False, fine-tune the entire network (slower but better)
    """
    if model_name == 'resnet50':
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        # ResNet50's final layer outputs 1000 classes (ImageNet)
        # We replace it with a layer that outputs 2 classes (real vs fake)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),       # regularization — randomly drops 50% of neurons during training
            nn.Linear(in_features, 1),  # single output neuron
            nn.Sigmoid()           # squashes output to 0-1 probability
        )

    elif model_name == 'efficientnet':
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        )

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')