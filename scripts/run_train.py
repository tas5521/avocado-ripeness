"""
本番訓練実行スクリプト

config.pyの設定を読み込んで訓練を実行する。
"""

import torch.optim as optim
import torch.nn as nn
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.model import (  # noqa: E402
    EfficientNetB0Model,
    EfficientNetLite0Model,
    EfficientNetLite1Model
)
from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_train_transforms,
    get_valid_transforms
)
from src.avocado_ripeness.train import train_multiple_epochs  # noqa: E402
from src.avocado_ripeness.utils import get_device, calculate_class_weights  # noqa: E402
from src.avocado_ripeness.config import (  # noqa: E402
    TRAIN_DIR,
    VALID_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    USE_SCHEDULER,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
    USE_EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    USE_DATA_AUGMENTATION,
    DROPOUT_RATE,
    USE_CLASS_WEIGHTS,
    USE_OVERSAMPLING,
    CLASS_MODE,
    MODEL_NAME,
    PRETRAINED
)


def _create_model(device):
    """モデルを作成してデバイスに移動"""
    print("\nモデルを作成中...")
    if MODEL_NAME == "efficientnet_lite0":
        model = EfficientNetLite0Model(
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
            dropout_rate=DROPOUT_RATE
        )
        model_display_name = "EfficientNet-Lite0"
    elif MODEL_NAME == "efficientnet_lite1":
        model = EfficientNetLite1Model(
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
            dropout_rate=DROPOUT_RATE
        )
        model_display_name = "EfficientNet-Lite1"
    elif MODEL_NAME == "efficientnet_b0":
        model = EfficientNetB0Model(
            num_classes=NUM_CLASSES,
            pretrained=PRETRAINED,
            dropout_rate=DROPOUT_RATE
        )
        model_display_name = "EfficientNet-B0"
    else:
        raise ValueError(
            f"不明なモデル名: {MODEL_NAME}。"
            f"サポートされているモデル: 'efficientnet_b0', 'efficientnet_lite0', 'efficientnet_lite1'"
        )

    model = model.to(device)
    print(f"  モデル: {model_display_name}")
    print(f"  クラス数: {NUM_CLASSES}")
    print(f"  事前訓練済み: {PRETRAINED}")
    if DROPOUT_RATE > 0:
        print(f"  ドロップアウト率: {DROPOUT_RATE}")
    return model


def _create_dataloaders():
    """訓練用とバリデーション用のDataLoaderを作成"""
    print("\nデータローダーを作成中...")
    train_transform = get_train_transforms(use_augmentation=USE_DATA_AUGMENTATION)
    valid_transform = get_valid_transforms()

    if USE_DATA_AUGMENTATION:
        print("  データ拡張: 有効")
    else:
        print("  データ拡張: 無効")

    train_dataloader = create_dataloader(
        data_dir=TRAIN_DIR,
        batch_size=BATCH_SIZE,
        shuffle=not USE_OVERSAMPLING,
        transform=train_transform,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        use_oversampling=USE_OVERSAMPLING,
        class_mode=CLASS_MODE
    )

    if USE_OVERSAMPLING:
        print("  オーバーサンプリング: 有効（少数クラスを多めにサンプリング）")
    if NUM_CLASSES == 3:
        print(f"  クラス統合方法: {CLASS_MODE}")

    valid_dataloader = create_dataloader(
        data_dir=VALID_DIR,
        batch_size=BATCH_SIZE,
        shuffle=False,
        transform=valid_transform,
        num_workers=NUM_WORKERS,
        num_classes=NUM_CLASSES,
        class_mode=CLASS_MODE
    )

    print(f"  訓練データセットサイズ: {len(train_dataloader.dataset)}")
    print(f"  バリデーションデータセットサイズ: {len(valid_dataloader.dataset)}")
    print(f"  バッチサイズ: {BATCH_SIZE}")
    return train_dataloader, valid_dataloader


def _create_criterion(train_dataloader, device):
    """損失関数を作成（クラス重み付けあり/なし）"""
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(
            train_dataloader.dataset, NUM_CLASSES
        )
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\n損失関数: CrossEntropyLoss（クラス重み付き）")
        print("  クラス重み:")
        if NUM_CLASSES == 3:
            class_names = ["未熟", "適熟", "過熟"]
        else:
            class_names = ["未熟(1)", "やや未熟(2)", "適熟(3)", "やや過熟(4)", "過熟(5)"]
        for (name, w) in zip(class_names, class_weights):
            print(f"    {name}: {w:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("\n損失関数: CrossEntropyLoss")
    return criterion


def _create_optimizer_and_scheduler(model):
    """オプティマイザーと学習率スケジューラーを作成"""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("オプティマイザー: Adam")
    print(f"学習率: {LEARNING_RATE}")

    scheduler = None
    if USE_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE,
            min_lr=SCHEDULER_MIN_LR
        )
        print("\n学習率スケジューラー: 有効")
        print(f"  factor: {SCHEDULER_FACTOR}")
        print(f"  patience: {SCHEDULER_PATIENCE}")
        print(f"  min_lr: {SCHEDULER_MIN_LR}")

    early_stopping_patience = None
    if USE_EARLY_STOPPING:
        early_stopping_patience = EARLY_STOPPING_PATIENCE
        print("\nEarly stopping: 有効")
        print(f"  patience: {EARLY_STOPPING_PATIENCE}")

    return optimizer, scheduler, early_stopping_patience


def _print_training_summary(history):
    """訓練結果のサマリーを表示"""
    print("\n" + "=" * 60)
    print("訓練完了 - 最終結果")
    print("=" * 60)
    print(f"\n最終エポック: {len(history['train_loss'])}")
    print(f"最終訓練損失: {history['train_loss'][-1]:.4f}")
    print(f"最終訓練Accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"最終バリデーション損失: {history['valid_loss'][-1]:.4f}")
    print(f"最終バリデーションAccuracy: {history['valid_accuracy'][-1]:.4f}")

    best_epoch = history['valid_loss'].index(min(history['valid_loss'])) + 1
    print(f"\n最良エポック: {best_epoch}")
    print(f"最良バリデーション損失: {min(history['valid_loss']):.4f}")
    print(f"最良バリデーションAccuracy: {history['valid_accuracy'][best_epoch - 1]:.4f}")
    print("\n最良モデルは以下に保存されています:")
    print(f"  {CHECKPOINT_DIR / 'best_model.pth'}")


def main():
    """本番訓練を実行"""
    print("=" * 60)
    print("アボカド成熟度分類モデル - 本番訓練")
    print("=" * 60)

    device = get_device()
    print(f"\n使用デバイス: {device}")

    model = _create_model(device)
    train_dataloader, valid_dataloader = _create_dataloaders()
    criterion = _create_criterion(train_dataloader, device)
    optimizer, scheduler, early_stopping_patience = _create_optimizer_and_scheduler(
        model)

    print(f"\nチェックポイント保存先: {CHECKPOINT_DIR}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"訓練を開始します（最大{NUM_EPOCHS}エポック）")
    print(f"{'=' * 60}")

    history = train_multiple_epochs(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        scheduler=scheduler,
        checkpoint_dir=CHECKPOINT_DIR,
        save_best_model=True,
        save_every_epoch=False,
        early_stopping_patience=early_stopping_patience
    )

    _print_training_summary(history)


if __name__ == "__main__":
    main()
