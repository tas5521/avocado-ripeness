"""
DataLoaderの作成モジュール

訓練用とバリデーション用のDataLoaderを作成する関数を提供する。
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .dataset import AvocadoDataset


def get_train_transforms(use_augmentation=True):
    """
    訓練用のtransformsを返す（データ拡張付き）

    Args:
        use_augmentation: データ拡張を使用するか（デフォルト: True）

    Returns:
        transforms.Compose: 訓練用の画像前処理パイプライン
    """
    if use_augmentation:
        # データ拡張を使用
        return transforms.Compose([
            transforms.Resize((256, 256)),  # 少し大きめにリサイズ
            transforms.RandomCrop(224),  # ランダムクロップ（224x224）
            transforms.RandomHorizontalFlip(p=0.5),  # 50%の確率で水平反転
            transforms.RandomRotation(degrees=15),  # ±15度のランダム回転
            transforms.ColorJitter(
                brightness=0.2,  # 明るさを±20%変更
                contrast=0.2,    # コントラストを±20%変更
                saturation=0.2,  # 彩度を±20%変更
                hue=0.1          # 色相を±10%変更
            ),
            transforms.ToTensor(),  # PIL画像をPyTorchテンソルに変換（0-255 → 0.0-1.0）
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNetの平均値
                std=[0.229, 0.224, 0.225]    # ImageNetの標準偏差
            ),
        ])
    else:
        # データ拡張なし（シンプル版）
        return transforms.Compose([
            transforms.Resize((224, 224)),  # 224x224にリサイズ
            transforms.ToTensor(),  # PIL画像をPyTorchテンソルに変換（0-255 → 0.0-1.0）
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNetの平均値
                std=[0.229, 0.224, 0.225]    # ImageNetの標準偏差
            ),
        ])


def get_valid_transforms():
    """
    バリデーション用のtransformsを返す

    バリデーションではデータ拡張は使用しない（評価の一貫性のため）

    Returns:
        transforms.Compose: バリデーション用の画像前処理パイプライン
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224にリサイズ
        transforms.ToTensor(),  # PIL画像をPyTorchテンソルに変換
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNetの平均値
            std=[0.229, 0.224, 0.225]    # ImageNetの標準偏差
        ),
    ])


def _create_weighted_sampler(dataset):
    """
    データセットのクラス分布に基づいてWeightedRandomSamplerを作成する

    少数クラスが多数クラスと同じ頻度でサンプリングされるようになる。

    Args:
        dataset: AvocadoDataset（targetsプロパティを持つ）

    Returns:
        WeightedRandomSampler
    """
    class_counts = Counter(dataset.targets)
    total = len(dataset)
    num_classes = len(class_counts)

    # 各クラスの重み（逆頻度）
    class_weight_map = {}
    for cls_idx, count in class_counts.items():
        class_weight_map[cls_idx] = total / (num_classes * count)

    # 各サンプルに対応する重みを割り当て
    sample_weights = torch.tensor(
        [class_weight_map[t] for t in dataset.targets],
        dtype=torch.float64
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total,
        replacement=True
    )


def create_dataloader(
    data_dir,
    batch_size=32,
    shuffle=True,
    transform=None,
    num_workers=0,
    num_classes=5,
    use_oversampling=False
):
    """
    DataLoaderを作成する

    Args:
        data_dir (str or Path): データディレクトリのパス（train/valid/test）
        batch_size (int): バッチサイズ（デフォルト: 32）
        shuffle (bool): データをシャッフルするか（デフォルト: True）
        transform: 画像の前処理（Noneの場合は適用しない）
        num_workers (int): データ読み込みに使うプロセス数（デフォルト: 0）
        num_classes (int): 分類クラス数（5: 5段階、3: 3段階）
        use_oversampling (bool): オーバーサンプリングを使用するか

    Returns:
        DataLoader: 作成されたDataLoader
    """
    # データセットを作成
    dataset = AvocadoDataset(data_dir, transform=transform, num_classes=num_classes)

    if use_oversampling:
        # WeightedRandomSamplerを使用（shuffleと併用不可）
        sampler = _create_weighted_sampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    return dataloader
