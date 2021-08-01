import torch.nn as nn
# import torch.nn.functional as F
from .transformer import TransformerBlock


class OneShotModel(nn.Module):
    """
    One shot learning siamese model
    Input images should be 84 x 84, 3 channels, so shape is (3, 84, 84)
    """

    def __init__(self, hidden, attn_heads, dropout, n_layers):
        super(OneShotModel, self).__init__()
        self.hidden = hidden  # equals 84 * 84
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.mask = None
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, self.dropout) for _ in
             range(self.n_layers, 1, -1)])
        self.fc = nn.Sequential(
            nn.Linear(3 * self.hidden, self.hidden),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.Linear(self.hidden // 2, 2),
        )  # 1 1 1 2

    def feed_forward(self, x):
        x = x.flatten(start_dim=2)
        for block in self.transformer_blocks:
            x = block(x, self.mask)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        output1 = self.feed_forward(x1)
        output2 = self.feed_forward(x2)
        return output1, output2
