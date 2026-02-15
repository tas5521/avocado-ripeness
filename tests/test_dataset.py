"""
Step 1: AvocadoDatasetの動作確認スクリプト

このスクリプトでは、AvocadoDatasetクラスが正しく動作するかを確認します。
- データセットが正しく初期化できるか
- 画像が読み込めるか
- ラベルが正しく取得できているか
- データセットのサイズはいくつか
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.dataset import AvocadoDataset  # noqa: E402


# データパス
DATA_DIR = Path("data/processed/avocado_ripeness")


def test_dataset_basic():
    """基本的な動作確認"""
    print("=" * 60)
    print("Step 1: AvocadoDatasetの動作確認")
    print("=" * 60)

    # 訓練データセットを作成
    train_dir = DATA_DIR / "train"
    print(f"\nデータディレクトリ: {train_dir}")
    print(f"ディレクトリが存在するか: {train_dir.exists()}")

    if not train_dir.exists():
        print("エラー: データディレクトリが見つかりません")
        return

    # データセットを初期化（transformなしでまず確認）
    print("\nデータセットを初期化中...")
    dataset = AvocadoDataset(train_dir, transform=None)

    # データセットのサイズを確認
    print(f"\nデータセットのサイズ: {len(dataset)}")

    # クラス情報を確認
    print(f"\nクラス数: {len(dataset.classes)}")
    print(f"クラス名: {dataset.classes}")
    print(f"クラス→インデックス変換: {dataset.class_to_idx}")

    # 最初のデータを取得して確認
    print("\n最初のデータを取得中...")
    image, label = dataset[0]

    print(f"\n画像の型: {type(image)}")
    print(f"画像のサイズ: {image.size if hasattr(image, 'size') else 'N/A'}")
    print(f"画像のモード: {image.mode if hasattr(image, 'mode') else 'N/A'}")
    print(f"ラベル（インデックス）: {label}")
    print(f"ラベル（クラス名）: {dataset.classes[label]}")

    # 画像パスを確認
    print(f"\n画像パス: {dataset.image_paths[0]}")

    # いくつかのデータを確認
    print("\n最初の5つのデータのラベル:")
    for i in range(min(5, len(dataset))):
        _, label = dataset[i]
        class_name = dataset.classes[label]
        print(f"  データ {i}: ラベル={label} (クラス={class_name})")

    print("\n" + "=" * 60)
    print("動作確認完了！")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset_basic()
