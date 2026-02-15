"""
推論実行スクリプト

コマンドラインから推論を実行するスクリプト
"""

import sys
import argparse
import torch
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.predict import (  # noqa: E402
    load_model_from_checkpoint,
    predict_single_image,
    predict_batch
)
from src.avocado_ripeness.config import (  # noqa: E402
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DROPOUT_RATE
)
from src.avocado_ripeness.utils import get_device  # noqa: E402


# クラス名の定義（成熟度のラベル）
CLASS_NAMES = {
    0: "未熟 (Unripe)",
    1: "やや未熟 (Slightly Unripe)",
    2: "適熟 (Ripe)",
    3: "やや過熟 (Slightly Overripe)",
    4: "過熟 (Overripe)"
}


def print_prediction_result(result, image_path):
    """推論結果を表示"""
    print(f"\n{'=' * 60}")
    print(f"画像: {image_path}")
    print(f"{'=' * 60}")
    print(f"\n予測クラス: {result['predicted_class']} - {result['predicted_class_name']}")
    print(f"確率: {result['probabilities'][result['predicted_class']]:.2%}")

    print("\n全クラスの確率:")
    for i, (class_idx, prob, class_name) in enumerate(result['top_k']):
        marker = "✓" if i == 0 else " "
        print(f"  {marker} {class_idx}: {class_name:30s} {prob:.2%}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="アボカド成熟度分類モデルで推論を実行"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="推論する画像ファイルのパス（またはディレクトリ）"
    )
    default_checkpoint = str(CHECKPOINT_DIR / "best_model.pth")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help=f"チェックポイントファイルのパス（デフォルト: {default_checkpoint}）"
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
    print("アボカド成熟度分類モデル - 推論")
    print("=" * 60)
    print(f"\n使用デバイス: {device}")
    print(f"チェックポイント: {args.checkpoint}")

    # モデルを読み込む
    print("\nモデルを読み込み中...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=device,
        dropout_rate=args.dropout_rate
    )
    print("✓ モデルの読み込み完了")

    # 画像パスを確認
    image_path = Path(args.image_path)

    if image_path.is_file():
        # 単一画像の推論
        print("\n単一画像の推論を実行します...")
        result = predict_single_image(
            model=model,
            image_path=image_path,
            device=device,
            class_names=[CLASS_NAMES[i] for i in range(args.num_classes)]
        )
        print_prediction_result(result, image_path)

    elif image_path.is_dir():
        # ディレクトリ内の全画像を推論
        print("\nディレクトリ内の全画像を推論します...")
        image_files = sorted(list(image_path.glob("*.jpg")))

        if not image_files:
            print(f"エラー: {image_path} に画像ファイルが見つかりません")
            return

        print(f"  見つかった画像数: {len(image_files)}")

        results = predict_batch(
            model=model,
            image_paths=image_files,
            device=device,
            class_names=[CLASS_NAMES[i] for i in range(args.num_classes)]
        )

        # 結果を表示
        for image_file, result in zip(image_files, results):
            print_prediction_result(result, image_file)

        # 統計情報
        print(f"\n{'=' * 60}")
        print("統計情報")
        print(f"{'=' * 60}")
        class_counts = {}
        for result in results:
            pred_class = result['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

        print("\n予測クラスの分布:")
        for class_idx in range(args.num_classes):
            count = class_counts.get(class_idx, 0)
            percentage = (count / len(results)) * 100 if results else 0
            print(
                f"  {class_idx}: {CLASS_NAMES[class_idx]:30s} {count:3d}枚 ({percentage:.1f}%)")

    else:
        print(f"エラー: {image_path} が見つかりません")
        return

    print(f"\n{'=' * 60}")
    print("推論完了")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
