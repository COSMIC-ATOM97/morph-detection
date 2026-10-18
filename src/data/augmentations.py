import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms(img_size: int = 224) -> A.Compose:
    """
    Anti-shortcut augmentation pipeline to prevent the network from memorizing
    dataset-specific digital artifacts and sensor noise.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ImageCompression(quality_lower=20, quality_upper=70, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.6),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Standard validation and testing transforms."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
