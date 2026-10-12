from torch.utils.data import  Dataset
from torch.nn.functional import one_hot
import torch
from PIL import Image
class PlantDataset(Dataset):
    def __init__(self, paths, labels, transform = None):
        self.paths = paths
        self.transform = transform
        self.unique_classes = sorted(set(labels))
        self.class_to_idx = torch.tensor(list(map(lambda x: self.unique_classes.index(x), labels)))
        self.labels = one_hot(self.class_to_idx, len(self.unique_classes))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, label

