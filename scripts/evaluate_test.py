"""
テストセット評価スクリプト

テストセット全体を評価して、Accuracy、Confusion Matrix、クラスごとの精度を表示します。
"""

import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.predict import load_model_from_checkpoint  # noqa: E402
from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_valid_transforms
)
from src.avocado_ripeness.evaluate import (  # noqa: E402
    evaluate_test_set,
    print_confusion_matrix,
    print_classification_report
)
from src.avocado_ripeness.config import (  # noqa: E402
    TEST_DIR,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DROPOUT_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASS_MODE
)
from src.avocado_ripeness.utils import get_device  # noqa: E402


CLASS_NAMES_5 = {
    0: "未熟 (Unripe)",
    1: "やや未熟 (Slightly Unripe)",
    2: "適熟 (Ripe)",
    3: "やや過熟 (Slightly Overripe)",
    4: "過熟 (Overripe)"
}

CLASS_NAMES_3 = {
    0: "未熟 (Unripe)",
    1: "適熟 (Ripe)",
    2: "過熟 (Overripe)"
}


def _get_class_names(num_classes):
    """クラス数に応じたクラス名辞書を返す"""
    if num_classes == 3:
        return CLASS_NAMES_3
    return CLASS_NAMES_5


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="テストセット全体を評価して精度を計算"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model.pth"),
        help=f"チェックポイントファイルのパス（デフォルト: {CHECKPOINT_DIR / 'best_model.pth'}）"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"クラス数（デフォルト: {NUM_CLASSES}）"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DROPOUT_RATE,
        help=f"ドロップアウト率（デフォルト: {DROPOUT_RATE}）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"バッチサイズ（デフォルト: {BATCH_SIZE}）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用するデバイス（mps/cuda/cpu、Noneの場合は自動選択）"
    )

    args = parser.parse_args()

    # デバイスを取得
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print("=" * 60)
    print("テストセット評価")
    print("=" * 60)
    print(f"\n使用デバイス: {device}")
    print(f"チェックポイント: {args.checkpoint}")
    print(f"テストセット: {TEST_DIR}")

    # モデルを読み込む
    print("\nモデルを読み込み中...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=device,
        dropout_rate=args.dropout_rate
    )
    print("✓ モデルの読み込み完了")

    # テストデータローダーを作成
    print("\nテストデータローダーを作成中...")
    test_transform = get_valid_transforms()
    test_dataloader = create_dataloader(
        data_dir=TEST_DIR,
        batch_size=args.batch_size,
        shuffle=False,
        transform=test_transform,
        num_workers=NUM_WORKERS,
        num_classes=args.num_classes,
        class_mode=CLASS_MODE
    )

    print(f"  テストデータセットサイズ: {len(test_dataloader.dataset)}")
    print(f"  バッチ数: {len(test_dataloader)}")
    print(f"  バッチサイズ: {args.batch_size}")

    # 損失関数を作成
    criterion = nn.CrossEntropyLoss()

    # テストセットを評価
    print("\n" + "=" * 60)
    print("テストセット評価を実行中...")
    print("=" * 60)

    results = evaluate_test_set(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print("テストセット評価結果")
    print("=" * 60)
    print(f"\nテスト損失: {results['loss']:.4f}")
    print(f"テストAccuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

    # Confusion Matrixを表示
    class_names_dict = _get_class_names(args.num_classes)
    class_names_list = [class_names_dict[i] for i in range(args.num_classes)]
    print_confusion_matrix(results['confusion_matrix'], class_names_list)

    # クラスごとの精度指標を表示
    print_classification_report(
        results['labels'],
        results['predictions'],
        class_names_list
    )

    print("\n" + "=" * 60)
    print("評価完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
