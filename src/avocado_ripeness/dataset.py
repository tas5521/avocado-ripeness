from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class AvocadoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: train / valid / test のディレクトリ
        transform: torchvision.transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # 画像パス一覧取得
        self.image_paths = sorted(list(self.root_dir.glob("**/*.jpg")))

        # ラベル取得（フォルダ名から）
        self.labels = [p.parent.name for p in self.image_paths]

        # ラベル→index変換
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.targets[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
