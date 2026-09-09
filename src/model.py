"""Shared model definition — imported by train.py (training) and
inference.py (serving), so the architecture is defined exactly once."""
import torch.nn as nn

ACTIVITY_LABELS = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]


class HARConvNet(nn.Module):
    """Small 1D CNN — deliberately simple; the pipeline is the deliverable."""

    def __init__(self, n_channels=9, n_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))
