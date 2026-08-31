import torch
import torch.nn as nn
import torchvision.models as models

CLASS_NAMES = [
    "epidural",
    "intracerebral",
    "intraventricular",
    "normal",
    "subarachnoid",
    "subdural",
]
NUM_CLASSES = len(CLASS_NAMES)


class HaemorrhageNet(nn.Module):
    """DenseNet-121 backbone with a custom classifier head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        in_features = backbone.classifier.in_features  # 1024 for densenet121

        # Keep the feature extractor ("features") — this is what Grad-CAM hooks into.
        self.features = backbone.features

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.features(x)          # [B, 1024, H, W]  <- Grad-CAM target
        out = self.classifier(feats)       # [B, num_classes]
        return out


def build_model(pretrained: bool = True) -> HaemorrhageNet:
    model = HaemorrhageNet(num_classes=NUM_CLASSES, pretrained=pretrained)
    return model


def load_trained_model(weights_path: str, device: str = "cpu") -> HaemorrhageNet:
    """Load a model with weights fine-tuned by train.py."""
    model = build_model(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
