import torch.nn as nn
import torch.nn.functional as F


class OneShotModel(nn.Module):
    """
    One shot learning siamese model
    Input images should be 84 x 84, 3 channels, so shape is (3, 84, 84)
    """

    def __init__(self):
        super(OneShotModel, self).__init__()
        sizes = [3, 96, 256, 384, 384, 256, 128, 32, 1]
        self.seq_block = nn.Sequential(
            nn.Conv2d(sizes[0], sizes[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(sizes[1]),

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[1], sizes[2], kernel_size=5),
            nn.BatchNorm2d(sizes[2]),

            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3),
            nn.BatchNorm2d(sizes[3]),

            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[3], sizes[4], kernel_size=3),
            nn.BatchNorm2d(sizes[4]),

            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[4], sizes[5], kernel_size=1),
            nn.BatchNorm2d(sizes[5]),

            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[5], sizes[6], kernel_size=1),
            nn.BatchNorm2d(sizes[6]),

            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[6], sizes[7], kernel_size=1),
            nn.BatchNorm2d(sizes[7]),

            nn.ReLU(inplace=True),
            nn.Conv2d(sizes[7], sizes[8], kernel_size=1),
            nn.BatchNorm2d(sizes[8]),

        )  # 1 1 1 2
        self.fc = nn.Linear(4, 2)

    def feed_forward(self, x):
        x = self.seq_block(x).flatten(start_dim=1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        output1 = self.feed_forward(x1)
        output2 = self.feed_forward(x2)
        return output1, output2
