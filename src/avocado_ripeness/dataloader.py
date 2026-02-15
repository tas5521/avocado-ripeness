"""
DataLoaderの作成モジュール

訓練用とバリデーション用のDataLoaderを作成する関数を提供する。
"""

from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import AvocadoDataset


def get_train_transforms():
    """
    訓練用のtransformsを返す

    Returns:
        transforms.Compose: 訓練用の画像前処理パイプライン
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224にリサイズ
        transforms.ToTensor(),  # PIL画像をPyTorchテンソルに変換（0-255 → 0.0-1.0）
    ])


def get_valid_transforms():
    """
    バリデーション用のtransformsを返す

    Returns:
        transforms.Compose: バリデーション用の画像前処理パイプライン
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 224x224にリサイズ
        transforms.ToTensor(),  # PIL画像をPyTorchテンソルに変換
    ])


def create_dataloader(
    data_dir,
    batch_size=32,
    shuffle=True,
    transform=None,
    num_workers=0
):
    """
    DataLoaderを作成する

    Args:
        data_dir (str or Path): データディレクトリのパス（train/valid/test）
        batch_size (int): バッチサイズ（デフォルト: 32）
        shuffle (bool): データをシャッフルするか（デフォルト: True）
        transform: 画像の前処理（Noneの場合は適用しない）
        num_workers (int): データ読み込みに使うプロセス数（デフォルト: 0）

    Returns:
        DataLoader: 作成されたDataLoader
    """
    # データセットを作成
    dataset = AvocadoDataset(data_dir, transform=transform)

    # DataLoaderを作成
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader
