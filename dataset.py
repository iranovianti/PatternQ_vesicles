import os
import pandas as pd
import numpy as np
from PIL import Image
from tifffile import imread

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HPADataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        filename = row['Id'] + '_green.png'
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        target = np.array(row['Target'].split(), dtype=int)
        label = np.zeros(28)
        label[target] = 1
        label = torch.tensor(label.astype('float32'))

        return image, label

class HPADataset_2classes(Dataset):
    def __init__(self, image_dir, label_csv, class1, class2, transform=None, seed=42):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

        class_channel = {'Nuc': 'blue', 'MT': 'red', 'ER': 'yellow'}
        self.channel1 = class_channel[class1]
        self.channel2 = class_channel[class2]

        rng = np.random.default_rng(seed)
        self.random_labels = rng.integers(0, 2, size=len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        label = np.zeros(2, dtype='float32')
        label[self.random_labels[idx]] = 1.0

        channel = self.channel1 if self.random_labels[idx] == 0 else self.channel2
        filename = row['Id'] + '_' + channel + '.png'
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

class PatternBlend(Dataset):
    def __init__(self, image_ids, image_dir, class1, class2,
                 alphas=np.arange(0.0, 1.1, 0.1), transform=None, return_meta=False):
        self.image_ids = image_ids
        self.image_dir = image_dir

        self.alphas = alphas
        self.transform = transform if transform else transforms.ToTensor()
        self.return_meta = return_meta

        self.class_channel = {'Nuc': 'blue', 'MT': 'red', 'ER': 'yellow', 'POI': 'green'}

        self.channel1 = self.class_channel[class1]
        self.channel2 = self.class_channel[class2]

        self.index_map = [ (i, alpha) for i in range(len(image_ids)) for alpha in self.alphas ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        i, alpha = self.index_map[idx]
        image_id = self.image_ids[i]

        path1 = os.path.join(self.image_dir, f"{image_id}_{self.channel1}.png")
        path2 = os.path.join(self.image_dir, f"{image_id}_{self.channel2}.png")
        
        img1 = Image.open(path1)
        img2 = Image.open(path2)

        tensor1 = transforms.functional.to_tensor(img1)
        tensor2 = transforms.functional.to_tensor(img2)

        # Normalize
        tensor1 = (tensor1 - tensor1.min()) / (tensor1.max() - tensor1.min() + 1e-8)
        tensor2 = (tensor2 - tensor2.min()) / (tensor2.max() - tensor2.min() + 1e-8)

        blended = alpha * tensor1 + (1 - alpha) * tensor2
        blended = self.transform(blended)

        label = torch.tensor([alpha], dtype=torch.float32)

        if self.return_meta:
            return blended, label, image_id, alpha
        else:
            return blended, label

class BaseDataset(Dataset):
    def __init__(self, image_ids, image_labels, image_dir, image_size, transform=None):
        self.image_ids = list(image_ids)
        self.image_labels = list(image_labels)
        
        self.image_dir = image_dir
        self.image_size= image_size
        self.transform = transform
    
    def _build_path(self, image_id):
        return os.path.join(self.image_dir, image_id)

    def _load_image(self, path):
        if path.lower().endswith((".tif", ".tiff")):
            return np.asarray(imread(path))
        else:
            return np.asarray(Image.open(path))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        path = self._build_path(image_id)
        img = self._load_image(path)
        tensor = prepare_image(img, self.image_size, self.transform)

        label = self.image_labels[image_id]
        label = torch.tensor([label], dtype=torch.float32)

        return tensor, label

# Helper functions

def robust_normalize(image, lower=1, upper=99):
    p_low, p_high = np.percentile(image, (lower, upper))
    denom = max(p_high - p_low, 1e-6)
    image = np.clip(image, p_low, p_high)
    return (image - p_low) / denom

def prepare_image(np_img, image_size, transform=None):
    np_img = robust_normalize(np_img, 1, 99.9)
    t = transforms.functional.to_tensor(np_img)
    if transform is not None:
        t = transform(t)
    t = transforms.Resize(image_size)(t)
    return t