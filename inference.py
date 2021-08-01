from torchvision import transforms
from torch.utils import data

from tqdm import tqdm

import torch

from core.dataset import CustomData

from core.model import OneShotModel, ContrastiveLoss


def inference(img1, img2, modelx):  # image should be standard 84x84x3, numpy
    img1 = transforms.ToTensor()(img1)
    img2 = transforms.ToTensor()(img2)
    img1 = img1.reshape((1, 3, 84, 84))
    img2 = img2.reshape((1, 3, 84, 84))
    o1, o2 = modelx(img1, img2)
    return (o1 - o2).pow(2).sum(1)


if __name__ == '__main__':
    dataset = CustomData("D://projects//torchtrain/ds//val//val", size=32, cuda=True)
    loader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    print("inference on train ds")
    precision_list = []
    precision_list2 = []
    criterion = ContrastiveLoss(1.0)

    model = torch.load("model")

    # model = OneShotModel().cuda()
    # model = OneShotModel(hidden=32 ** 2, attn_heads=8, dropout=.1, n_layers=4).cuda()

    for i, (img1, img2, target) in tqdm(enumerate(loader), total=len(loader)):
        if i > 1500:  # for a quick partial infer
            break
        out1, out2 = model(img1, img2)
        diff = 1 - (criterion(out1, out2, target).item() / 0.71)  # relative to the untrained model
        precision_list.append(diff)
        target = target.detach().cpu().numpy()
        dist = torch.sqrt(torch.sum(torch.pow((out1 - out1), 2), 1)).cpu().detach().numpy()
        diff = 1 - abs(target - dist) / (1 + target)
        precision_list2.append(diff)
    print("precision on val dataset is {}% relative and {}% absolute".format(
        round(float(sum(precision_list) / len(precision_list)), 3) * 100,
        round(float(sum(precision_list2) / len(precision_list2)), 3) * 100)
    )
# relative is needed because absolute value works weird, untrained model has 75% precision and all trained 75.1% or same
