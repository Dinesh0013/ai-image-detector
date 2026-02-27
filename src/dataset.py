import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# These values come from your EDA pixel stats above
# We will update them after running Task 1
MEAN = [0.485, 0.456, 0.406]  # ImageNet defaults for now
STD  = [0.229, 0.224, 0.225]

class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir : path to data/raw/faces
        split    : 'train', 'valid', or 'test'
        """
        self.samples = []
        self.transform = transform

        for label, class_idx in [('real', 0), ('fake', 1)]:
            folder = os.path.join(root_dir, split, label)
            for fname in os.listdir(folder):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(folder, fname),
                        class_idx
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split='train'):
    """
    Training gets augmentations to make the model robust.
    Validation and test get only resizing and normalization — 
    we never augment evaluation data because we want consistent results.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def get_dataloader(root_dir, split='train', batch_size=32):
    dataset = FaceDataset(
        root_dir=root_dir,
        split=split,
        transform=get_transforms(split)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),  # only shuffle training data
        num_workers=0               # set to 4 if on Linux/Mac
    )