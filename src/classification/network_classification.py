from collections import OrderedDict
import torch
from torch.nn import functional as f
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, sampler
import timm


class EfficientnetCus(nn.Module):

    def __init__(self, model, num_class, model_weight=None, is_train=True):
        super(EfficientnetCus, self).__init__()

        self.is_train = is_train
        self.model = timm.create_model(f'tf_efficientnet_{model}_ns', pretrained=is_train,
                                       in_chans=3, num_classes=num_class)
        if model_weight is not None:
            new_keys = self.model.state_dict().keys()
            values = torch.load(model_weight, map_location=lambda storage, loc: storage).values()
            self.model.load_state_dict(OrderedDict(zip(new_keys, values)))

    def forward(self, image):
        if self.is_train:
            out = self.model(image)

            return out.squeeze(-1)
        else:
            vertical = image.flip(1)
            horizontal = image.flip(2)
            rotate90 = torch.rot90(image, 1, (1, 2))
            rotate90_ = torch.rot90(image, 1, (2, 1))
            out = torch.stack([image, vertical, horizontal, rotate90, rotate90_])

            return torch.sigmoid(self.model(out)).mean()
