import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Pre-computed ImageNet means and stds for MobileNet transfer learning
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

CLASS_MAPPING = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
    'mel': 4, 'nv': 5, 'vasc': 6
}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

def get_transforms(is_train=True):
    """
    Returns torchvision transforms for training or evaluation.
    We avoid unrealistic transforms (like extreme color jitter) that could destroy medical context.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'image_path' and 'dx' columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        
        label_str = self.dataframe.loc[idx, 'dx']
        label = CLASS_MAPPING[label_str]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
