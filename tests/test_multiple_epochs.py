"""
複数エポック訓練の動作確認

train.pyのtrain_multiple_epochs関数をテストする。
- 複数エポックの訓練が実行できるか
- 訓練とバリデーションが交互に実行されるか
- エポックごとの結果が記録されるか
- 設定ファイルの使い方を確認
"""

import torch.optim as optim
import torch.nn as nn
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.model import EfficientNetB0Model  # noqa: E402
from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_train_transforms,
    get_valid_transforms
)
from src.avocado_ripeness.train import train_multiple_epochs  # noqa: E402
from src.avocado_ripeness.utils import get_device  # noqa: E402
from src.avocado_ripeness.config import (  # noqa: E402
    TRAIN_DIR,
    VALID_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    NUM_CLASSES
)


def test_multiple_epochs():
    """複数エポック訓練のテスト"""
    print("=" * 60)
    print("複数エポック訓練の動作確認")
    print("=" * 60)

    # デバイスを取得
    device = get_device()
    print(f"\n使用デバイス: {device}")

    # モデルを作成
    model = EfficientNetB0Model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print(f"\nモデルを作成しました（クラス数: {NUM_CLASSES}）")

    # 損失関数とオプティマイザーを作成
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"損失関数とオプティマイザーを作成しました（学習率: {LEARNING_RATE}）")

    # データローダーを作成
    train_transform = get_train_transforms()
    valid_transform = get_valid_transforms()

    print("\nデータローダーを作成中...")
    # テスト用に小さなバッチサイズを使用（メモリ節約）
    test_batch_size = 16  # 通常の32から16に減らす
    train_dataloader = create_dataloader(
        data_dir=TRAIN_DIR,
        batch_size=test_batch_size,
        shuffle=True,
        transform=train_transform,
        num_workers=NUM_WORKERS
    )

    valid_dataloader = create_dataloader(
        data_dir=VALID_DIR,
        batch_size=test_batch_size,
        shuffle=False,
        transform=valid_transform,
        num_workers=NUM_WORKERS
    )

    print(f"  訓練データセットサイズ: {len(train_dataloader.dataset)}")
    print(f"  バリデーションデータセットサイズ: {len(valid_dataloader.dataset)}")
    print(f"  バッチサイズ: {test_batch_size}（テスト用に小さく設定）")

    # 複数エポック訓練を実行（テスト用に少ないエポック数）
    test_num_epochs = 2  # テスト用に2エポックのみ（動作確認のため）
    print(f"\n訓練エポック数: {test_num_epochs}（テスト用）")

    history = train_multiple_epochs(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=test_num_epochs
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print("訓練履歴")
    print("=" * 60)
    print("\nエポックごとの結果:")
    for epoch in range(test_num_epochs):
        print(f"\nEpoch {epoch + 1}:")
        print(
            f"  訓練 - Loss: {history['train_loss'][epoch]:.4f}, Accuracy: {history['train_accuracy'][epoch]:.4f}")
        print(
            f"  バリデーション - Loss: {history['valid_loss'][epoch]:.4f}, Accuracy: {history['valid_accuracy'][epoch]:.4f}")

    # 改善の確認
    print("\n" + "=" * 60)
    print("改善の確認")
    print("=" * 60)
    if len(history['train_loss']) > 1:
        first_loss = history['train_loss'][0]
        last_loss = history['train_loss'][-1]
        first_acc = history['train_accuracy'][0]
        last_acc = history['train_accuracy'][-1]

        print(f"\n訓練損失: {first_loss:.4f} → {last_loss:.4f}")
        if last_loss < first_loss:
            print("  ✓ 訓練損失が改善しました！")
        else:
            print("  ⚠ 訓練損失が改善していません")

        print(f"\n訓練Accuracy: {first_acc:.4f} → {last_acc:.4f}")
        if last_acc > first_acc:
            print("  ✓ 訓練Accuracyが改善しました！")
        else:
            print("  ⚠ 訓練Accuracyが改善していません")


def test_config_usage():
    """設定ファイルを確認"""
    print("\n" + "=" * 60)
    print("=" * 60)

    print("\n設定ファイル（config.py）から設定をインポート:")
    print(f"  BATCH_SIZE = {BATCH_SIZE}")
    print(f"  NUM_EPOCHS = {NUM_EPOCHS}")
    print(f"  LEARNING_RATE = {LEARNING_RATE}")
    print(f"  NUM_CLASSES = {NUM_CLASSES}")
    print(f"  TRAIN_DIR = {TRAIN_DIR}")
    print(f"  VALID_DIR = {VALID_DIR}")


if __name__ == "__main__":
    # 1. 設定ファイルを確認
    test_config_usage()

    # 2. 複数エポック訓練のテスト
    test_multiple_epochs()
