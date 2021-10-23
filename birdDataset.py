import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(data.Dataset):
    def __init__(self,
                 root,
                 label,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        fh = open(label)
        c = 0
        imgs = []
        class_names = []

        with open('classes.txt') as f:
            lines = f.readlines()

            for line in lines:
                class_names.append(line[:-1])

        for line in fh.readlines():
            cls = line.split()
            fn = cls.pop(0)
            fn1 = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn)):
                imgs.append((fn, np.longlong(fn1[0:3]) - 1))
            c = c + 1
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        label = np.array(label)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes
