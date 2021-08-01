#  Copyright (c) 2021. Only can be used after citation or my affiliate's approval
import torch.nn as nn


class Syll2words:
    def __init__(self, hidden, device):
        self.linear = nn.Linear(9 * hidden, hidden).to(device)
        self.conv1 = nn.Conv2d(9, 6, 3, padding=1).to(device)
        self.conv2 = nn.Conv2d(6, 3, 3, padding=1).to(device)
        self.conv3 = nn.Conv2d(3, 1, 3, padding=1).to(device)

    def __call__(self, x,  *args, **kwargs):
        x = x.permute(0, 2, 1, 3)
        x = self.conv3(self.conv2(self.conv1(x)))
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], x.shape[3])
        # temp = []
        # for batch in x:
        #     tmp_inner = []
        #     for word in batch:
        #         tmp_inner.append(self.linear(word))
        #     temp.append(tmp_inner)
        # temp2 = []
        # for n in temp:
        #     temp2.append(torch.stack(n))
        # x = torch.stack(temp2)
        return x
