import cv2
import torch
from torch.nn import functional as f
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, sampler


class ChestClassifierDataset(Dataset):
    def __init__(self, df, image_folder,
                 transform=None, is_train=True):
        super(ChestClassifierDataset, self).__init__()

        self.df = df
        self.image_folder = image_folder
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def load_image_box(self, image_name):
        image_path = [i for i in self.image_folder if image_name in i][0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        image_name = self.df.loc[idx, 'image_id']
        images = self.load_image_box(image_name)

        if self.transform is not None:
            sample = self.transform(image=images)
            images = sample['image']
        if self.is_train:
            label = self.df.loc[idx, 'label']
            return images, torch.tensor(label).float()
        else:
            return image_name, images

    def collate_fn(self, batch):
        return tuple(zip(*batch))