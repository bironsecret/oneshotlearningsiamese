import random

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import PIL
# import cv2


class CustomData(Dataset):
    """
    CustomData dataset
    """

    def __init__(self, dirpath, size=84, should_invert=False, cuda=True):  # target should be either 1 for
        # same class or 0
        super(Dataset, self).__init__()
        self.dirpath = dirpath
        self.size = size
        self.imageFolderDataset = torchvision.datasets.ImageFolder(root=self.dirpath)

        self.should_invert = should_invert
        self.to_tensor = transforms.ToTensor()
        self.cuda = cuda

    def __getitem__(self, index):
        # Training images
        images = self.imageFolderDataset.imgs
        # img = cv2.imread(images[index][0])
        img = Image.open(images[index][0])

        target = 1.0 if random.random() > 0.5 else 0.0

        if target == 1:
            try:
                if images[index + 1][1] == images[index][1]:
                    rand = Image.open(images[index + 1][0])
                else:
                    rand = Image.open(images[index - 1][0])
            except:
                rand = Image.open(images[index - 1][0])
        else:
            rand = Image.open(random.choice(images)[0])

        if self.should_invert:
            img = PIL.ImageOps.invert(img)
            rand = PIL.ImageOps.invert(rand)

        img = img.resize((self.size, self.size)).convert('RGB')
        rand = rand.resize((self.size, self.size)).convert('RGB')

        img = np.array(img, dtype='uint8')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_as_tensor = self.to_tensor(img)

        rand = np.array(rand, dtype='uint8')
        rand_as_tensor = self.to_tensor(rand)

        # return img_as_tensor, images[index][1]
        if self.cuda:
            img_as_tensor = img_as_tensor.cuda()
            rand_as_tensor = rand_as_tensor.cuda()
            target = torch.tensor(target, dtype=torch.float).cuda()

        return img_as_tensor, rand_as_tensor, target

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
