from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

# 3段階分類のマッピング設定
# "merge": フォルダ1,2→未熟、3→適熟、4,5→過熟（全データ使用）
# "select": フォルダ1→未熟、3→適熟、5→過熟（境界クラス2,4を除外）
LABEL_MAP_3CLASS_MERGE = {
    '1': 0, '2': 0,
    '3': 1,
    '4': 2, '5': 2,
}

LABEL_MAP_3CLASS_SELECT = {
    '1': 0,
    '3': 1,
    '5': 2,
}

# 使用するフォルダ（selectモード）
SELECTED_FOLDERS = {'1', '3', '5'}


class AvocadoDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=5, class_mode="select"):
        """
        root_dir: train / valid / test のディレクトリ
        transform: torchvision.transforms
        num_classes: 分類クラス数（5: 5段階、3: 3段階）
        class_mode: 3段階の統合方法（"merge": 全データ使用、"select": 1,3,5のみ）
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.num_classes = num_classes

        # 画像パス一覧取得
        all_paths = sorted(list(self.root_dir.glob("**/*.jpg")))

        if num_classes == 3 and class_mode == "select":
            # フォルダ1, 3, 5のみ使用
            self.image_paths = [
                p for p in all_paths if p.parent.name in SELECTED_FOLDERS]
            self.labels = [p.parent.name for p in self.image_paths]
            self.classes = ['unripe', 'ripe', 'overripe']
            self.class_to_idx = {'unripe': 0, 'ripe': 1, 'overripe': 2}
            self.targets = [LABEL_MAP_3CLASS_SELECT[label] for label in self.labels]
        elif num_classes == 3:
            # 全データ使用してマージ
            self.image_paths = all_paths
            self.labels = [p.parent.name for p in self.image_paths]
            self.classes = ['unripe', 'ripe', 'overripe']
            self.class_to_idx = {'unripe': 0, 'ripe': 1, 'overripe': 2}
            self.targets = [LABEL_MAP_3CLASS_MERGE[label] for label in self.labels]
        else:
            # 5段階分類（従来通り）
            self.image_paths = all_paths
            self.labels = [p.parent.name for p in self.image_paths]
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
