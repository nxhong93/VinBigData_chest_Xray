import albumentations as al
from albumentations.pytorch import ToTensorV2, ToTensor


def aug(file):
    if file=='train':
        return al.Compose([
            al.VerticalFlip(p=0.5),
            al.HorizontalFlip(p=0.5),
            al.RandomRotate90(p=0.5),
            al.OneOf([
                al.GaussNoise(0.002, p=0.5),
                al.IAAAffine(p=0.5),
            ], p=0.2),
            al.OneOf([
                al.Blur(blur_limit=(3, 10), p=0.4),
                al.MedianBlur(blur_limit=3, p=0.3),
                al.MotionBlur(p=0.3)
            ], p=0.3),
            al.OneOf([
                al.RandomBrightness(p=0.3),
                al.RandomContrast(p=0.4),
                al.RandomGamma(p=0.3)
            ], p=0.5),
            al.Cutout(num_holes=20, max_h_size=20, max_w_size=20, p=0.5),
            al.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(0.9, 1), rotate_limit=45, p=0.3),
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ])

    elif file=='validation':
        return al.Compose([
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ])

    elif file=='test':
        return al.Compose([
            al.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
            ToTensorV2(p=1)
        ], p=1)