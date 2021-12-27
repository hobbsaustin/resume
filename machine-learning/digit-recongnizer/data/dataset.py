from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, file_name):
        self.df = file_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((.5, ), (.5, ))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image = self.df.iloc[item, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.df.iloc[item, 0]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

