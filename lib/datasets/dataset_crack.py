import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CrackDataset(Dataset):
    def __init__(self, root_path, img_size, split, transform=None):
        self.img_size = img_size
        self.split = split
        if not transform:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        if not (split == 'train' or split == 'val' or split == 'test'):
            raise KeyError("Stage should be \"train\" or \"val\" or \"test\"")
        txt_path = os.path.join(root_path, split + ".txt")
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            img_list = [line[0] for line in lines]
            ann_list = [line[1] for line in lines]
        self.img_path_list = [os.path.join(root_path, img) for img in img_list]
        self.label_path_list = [os.path.join(root_path, img) for img in ann_list]
        self.dataset_size = len(img_list)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        label_path = self.label_path_list[idx]
        data = Image.open(img_path)
        label = Image.open(label_path)

        # check size
        if data.size != (self.img_size, self.img_size):
            data = data.resize((self.img_size, self.img_size))
        if label.size != (self.img_size, self.img_size):
            label = label.resize((self.img_size, self.img_size))

        data = data.convert("RGB")  # to RGB
        data = self.transforms(data)

        label = label.convert("L")  # to grayscale
        label = torch.squeeze(self.transforms(label))
        label = torch.round(label)

        sample = {'image': data, 'label': label}
        return sample