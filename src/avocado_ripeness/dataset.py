from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# 5段階 → 3段階のマッピング
# フォルダ1,2（未熟+やや未熟）→ 0（未熟）
# フォルダ3（適熟）→ 1（適熟）
# フォルダ4,5（やや過熟+過熟）→ 2（過熟）
LABEL_MAP_3CLASS = {
    '1': 0, '2': 0,
    '3': 1,
    '4': 2, '5': 2,
}


class AvocadoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=5):
        """
        root_dir: train / valid / test のディレクトリ
        transform: torchvision.transforms
        num_classes: 分類クラス数（5: 5段階、3: 3段階）
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_classes = num_classes

        # 画像パス一覧取得
        self.image_paths = sorted(list(self.root_dir.glob("**/*.jpg")))

        # ラベル取得（フォルダ名から）
        self.labels = [p.parent.name for p in self.image_paths]

        if num_classes == 3:
            # 3段階分類
            self.classes = ['unripe', 'ripe', 'overripe']
            self.class_to_idx = {'unripe': 0, 'ripe': 1, 'overripe': 2}
            self.targets = [LABEL_MAP_3CLASS[label] for label in self.labels]
        else:
            # 5段階分類（従来通り）
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
