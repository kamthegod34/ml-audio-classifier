import torch.nn as nn

class small_CNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channel=16, out_channel=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1,1),

            nn.Flatten(),
            nn.Linear(64, n_classes),
        )

        def forward(self, x):
            return self.network(x)