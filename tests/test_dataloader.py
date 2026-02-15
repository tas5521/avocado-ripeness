"""
Step 2: DataLoaderの動作確認スクリプト

以下を確認する。
- DataLoaderの作成
- バッチでデータを取得できるか
- バッチの形状を確認
- transformsが正しく適用されているか
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_train_transforms,
    get_valid_transforms
)


# データパス
DATA_DIR = Path("data/processed/avocado_ripeness")


def test_dataloader_creation():
    """DataLoaderが正しく作成できるか確認"""
    print("=" * 60)
    print("1. DataLoaderの動作確認")
    print("=" * 60)

    train_dir = DATA_DIR / "train"
    print(f"\nデータディレクトリ: {train_dir}")

    # DataLoaderを作成（transformsあり）
    # 注意: DataLoaderはテンソルを期待するため、transformは必須です
    print("\n1. transformsありでDataLoaderを作成...")
    train_transform = get_train_transforms()
    dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=4,
        shuffle=True,
        transform=train_transform,
        num_workers=0
    )

    print("   DataLoader作成成功！")
    print(f"   データセットのサイズ: {len(dataloader.dataset)}")
    print(f"   バッチ数: {len(dataloader)}")

    # 最初のバッチを取得
    images, labels = next(iter(dataloader))

    print(f"\n   バッチの画像数: {len(images)}")
    print(f"   画像の型: {type(images)}")
    print(f"   画像の形状: {images.shape}")
    print(f"   ラベルの形状: {labels.shape}")
    print(f"   ラベルの値: {labels.tolist()}")


def test_dataloader_with_transforms():
    """transformsを使ったDataLoaderの動作確認"""
    print("\n" + "=" * 60)
    print("2. transformsを使ったDataLoaderの動作確認")
    print("=" * 60)

    train_dir = DATA_DIR / "train"

    # 訓練用のtransformsを取得
    train_transform = get_train_transforms()
    print(f"\n訓練用transforms: {train_transform}")

    # DataLoaderを作成（transformsあり）
    print("\n訓練用DataLoaderを作成...")
    train_dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=4,
        shuffle=True,
        transform=train_transform,
        num_workers=0
    )

    print("   DataLoader作成成功！")

    # バッチを取得
    images, labels = next(iter(train_dataloader))

    print(f"\n   画像の形状: {images.shape}")
    print("   期待値: [batch_size, channels, height, width] = [4, 3, 224, 224]")
    print(f"   実際: {images.shape}")

    print(f"\n   画像の値の範囲: {images.min().item():.3f} ～ {images.max().item():.3f}")
    print("   （ToTensor()により、0-255 → 0.0-1.0に変換されている）")

    print(f"\n   ラベルの形状: {labels.shape}")
    print(f"   ラベルの値: {labels.tolist()}")


def test_train_valid_dataloaders():
    """訓練用とバリデーション用のDataLoaderを両方作成"""
    print("\n" + "=" * 60)
    print("3. 訓練用とバリデーション用のDataLoaderを作成")
    print("=" * 60)

    train_dir = DATA_DIR / "train"
    valid_dir = DATA_DIR / "valid"

    # transformsを取得
    train_transform = get_train_transforms()
    valid_transform = get_valid_transforms()

    # 訓練用DataLoaderを作成
    print("\n訓練用DataLoaderを作成...")
    train_dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=32,
        shuffle=True,  # 訓練時はシャッフル
        transform=train_transform,
        num_workers=0
    )

    print(f"   訓練データセットのサイズ: {len(train_dataloader.dataset)}")
    print(f"   訓練バッチ数: {len(train_dataloader)}")

    # バリデーション用DataLoaderを作成
    print("\nバリデーション用DataLoaderを作成...")
    valid_dataloader = create_dataloader(
        data_dir=valid_dir,
        batch_size=32,
        shuffle=False,  # バリデーション時はシャッフルしない
        transform=valid_transform,
        num_workers=0
    )

    print(f"   バリデーションデータセットのサイズ: {len(valid_dataloader.dataset)}")
    print(f"   バリデーションバッチ数: {len(valid_dataloader)}")

    # それぞれのバッチを取得して確認
    print("\n訓練用バッチを取得...")
    train_images, train_labels = next(iter(train_dataloader))
    print(f"   画像の形状: {train_images.shape}")
    print(f"   ラベルの形状: {train_labels.shape}")

    print("\nバリデーション用バッチを取得...")
    valid_images, valid_labels = next(iter(valid_dataloader))
    print(f"   画像の形状: {valid_images.shape}")
    print(f"   ラベルの形状: {valid_labels.shape}")


def test_dataloader_iteration():
    """DataLoaderをループで回す方法を確認"""
    print("\n" + "=" * 60)
    print("4. DataLoaderをループで回す")
    print("=" * 60)

    train_dir = DATA_DIR / "train"
    train_transform = get_train_transforms()

    dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=32,
        shuffle=True,
        transform=train_transform,
        num_workers=0
    )

    print(f"\nデータセットのサイズ: {len(dataloader.dataset)}")
    print("バッチサイズ: 32")
    print(f"バッチ数: {len(dataloader)}")

    # 最初の3バッチだけ処理してみる
    print("\n最初の3バッチを処理中...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 3:
            break

        print(f"\nバッチ {batch_idx + 1}:")
        print(f"   画像の形状: {images.shape}")
        print(f"   ラベルの形状: {labels.shape}")
        print(f"   ラベルの値（最初の5つ）: {labels[:5].tolist()}")


if __name__ == "__main__":
    # 1. DataLoaderの作成確認
    test_dataloader_creation()

    # 2. transformsを使ったDataLoaderの確認
    test_dataloader_with_transforms()

    # 3. 訓練用とバリデーション用のDataLoaderを作成
    test_train_valid_dataloaders()

    # 4. DataLoaderをループで回す
    test_dataloader_iteration()
