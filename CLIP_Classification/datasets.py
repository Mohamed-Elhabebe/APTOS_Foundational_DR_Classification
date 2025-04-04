import os
from PIL import Image
from torch.utils.data import Dataset

class DRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        folder_names_to_idx = {'class_0': 0, 'class_1': 1}
        self.image_paths = []
        self.labels = []

        for folder_name, label in folder_names_to_idx.items():
            cls_dir = os.path.join(root_dir, folder_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
