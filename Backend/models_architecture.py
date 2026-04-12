import torch
import torch.nn as nn
from torchvision import models
from utils.cbam import CBAM


class DenseNetCBAMV2(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        base = models.densenet121(weights=None)

        self.features = base.features
        self.features.add_module("cbam", CBAM(1024))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)


class ResNet50CBAMV2(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        base = models.resnet50(weights=None)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = nn.Sequential(base.layer3, CBAM(1024))
        self.layer4 = nn.Sequential(base.layer4, CBAM(2048))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)

        return self.classifier(x)


class MobileNetCBAMV2(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        base = models.mobilenet_v3_large(weights=None)

        self.features = base.features
        self.cbam = CBAM(960)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)

        return self.classifier(x)


class EfficientNetCBAMV2(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        base = models.efficientnet_v2_s(weights=None)

        self.features = base.features

        for idx, ch in {4:128,5:160,6:256}.items():
            self.features[idx] = nn.Sequential(
                self.features[idx],
                CBAM(ch)
            )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):

        x = self.features(x)
        x = self.pool(x)

        return self.classifier(x)