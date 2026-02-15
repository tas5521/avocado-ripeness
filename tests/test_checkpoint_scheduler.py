"""
チェックポイント、学習率スケジューラー、Early stoppingの動作確認

train.pyの機能をテストする。
- チェックポイントの保存と読み込み
- 学習率スケジューラーの動作
- Early stoppingの動作
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
from src.avocado_ripeness.train import (  # noqa: E402
    train_multiple_epochs,
    save_checkpoint,
    load_checkpoint
)
from src.avocado_ripeness.utils import get_device  # noqa: E402
from src.avocado_ripeness.config import (  # noqa: E402
    TRAIN_DIR,
    VALID_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    NUM_CLASSES,
    USE_SCHEDULER,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
    USE_EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE
)

# テスト用の設定
TEST_CHECKPOINT_DIR = Path("checkpoints/test")
TEST_NUM_EPOCHS = 3  # テスト用に少ないエポック数


def test_checkpoint_save():
    """チェックポイントの保存をテスト"""
    print("=" * 60)
    print("チェックポイント、学習率スケジューラー、Early stopping")
    print("=" * 60)

    print("\n1. チェックポイントの保存")
    print("-" * 60)

    device = get_device()
    model = EfficientNetB0Model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ダミーの履歴を作成
    history = {
        'train_loss': [1.5, 1.3],
        'train_accuracy': [0.3, 0.4],
        'valid_loss': [1.6, 1.4],
        'valid_accuracy': [0.25, 0.35]
    }

    # チェックポイントを保存
    print("\nチェックポイントを保存中...")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=2,
        history=history,
        checkpoint_dir=TEST_CHECKPOINT_DIR,
        is_best=True
    )

    # チェックポイントファイルの確認
    checkpoint_files = list(TEST_CHECKPOINT_DIR.glob("*.pth"))
    print(f"  保存されたファイル数: {len(checkpoint_files)}")
    for f in checkpoint_files:
        print(f"    - {f.name}")


def test_checkpoint_load():
    """チェックポイントの読み込みをテスト"""
    print("\n2. チェックポイントの読み込み")
    print("-" * 60)

    device = get_device()

    # 新しいモデルとオプティマイザーを作成して読み込み
    print("\nチェックポイントを読み込み中...")
    new_model = EfficientNetB0Model(num_classes=NUM_CLASSES, pretrained=False)
    new_model = new_model.to(device)
    new_optimizer = optim.Adam(new_model.parameters(), lr=LEARNING_RATE)

    checkpoint = load_checkpoint(
        model=new_model,
        optimizer=new_optimizer,
        checkpoint_path=TEST_CHECKPOINT_DIR / "best_model.pth",
        device=device
    )

    print(f"  読み込んだエポック: {checkpoint['epoch']}")
    print(f"  履歴の長さ: {len(checkpoint['history']['train_loss'])}")


def test_training_with_features():
    """チェックポイント、スケジューラー、Early stoppingを使った訓練をテスト"""
    print("\n" + "=" * 60)
    print("実用的な訓練機能のテスト")
    print("-" * 60)

    device = get_device()
    print(f"\n使用デバイス: {device}")

    # モデルを作成
    model = EfficientNetB0Model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print("\nモデルを作成しました")

    # 損失関数とオプティマイザーを作成
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("損失関数とオプティマイザーを作成しました")

    # 学習率スケジューラーを作成
    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR
        )
        print(f"学習率スケジューラーを作成しました（patience={SCHEDULER_PATIENCE}）")

    # データローダーを作成
    train_transform = get_train_transforms()
    valid_transform = get_valid_transforms()

    print("\nデータローダーを作成中...")
    train_dataloader = create_dataloader(
        data_dir=TRAIN_DIR,
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=train_transform,
        num_workers=NUM_WORKERS
    )

    valid_dataloader = create_dataloader(
        data_dir=VALID_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False,
        transform=valid_transform,
        num_workers=NUM_WORKERS
    )

    print(f"  訓練データセットサイズ: {len(train_dataloader.dataset)}")
    print(f"  バリデーションデータセットサイズ: {len(valid_dataloader.dataset)}")

    # Early stoppingの設定
    early_stopping_patience = None
    if USE_EARLY_STOPPING:
        early_stopping_patience = EARLY_STOPPING_PATIENCE
        print(f"Early stopping: patience={early_stopping_patience}")

    # 訓練を実行
    print(f"\n訓練を開始します（{TEST_NUM_EPOCHS}エポック）...")
    history = train_multiple_epochs(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=TEST_NUM_EPOCHS,
        scheduler=scheduler,
        checkpoint_dir=TEST_CHECKPOINT_DIR,
        save_best_model=True,
        save_every_epoch=False,
        early_stopping_patience=early_stopping_patience
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print("訓練履歴")
    print("=" * 60)
    for epoch in range(len(history['train_loss'])):
        print(f"\nEpoch {epoch + 1}:")
        print(
            f"  訓練 - Loss: {history['train_loss'][epoch]:.4f}, Accuracy: {history['train_accuracy'][epoch]:.4f}")
        print(
            f"  バリデーション - Loss: {history['valid_loss'][epoch]:.4f}, Accuracy: {history['valid_accuracy'][epoch]:.4f}")


if __name__ == "__main__":
    # 1. チェックポイントの保存
    test_checkpoint_save()

    # 2. チェックポイントの読み込み
    test_checkpoint_load()

    # 3. 実用的な訓練機能のテスト
    test_training_with_features()
