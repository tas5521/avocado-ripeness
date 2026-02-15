"""
utilsモジュールのテスト

utils.pyの関数をテストします。
"""

import torch
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.utils import (  # noqa: E402
    get_device,
    calculate_accuracy
)


def test_calculate_accuracy():
    """Accuracy計算関数のテスト"""
    print("=" * 60)
    print("utilsモジュールのテスト")
    print("=" * 60)

    print("\n1. Accuracy計算関数のテスト")
    print("-" * 60)

    # ダミーデータでテスト
    outputs = torch.tensor([
        [2.0, 1.0, 0.1, 0.1, 0.1],  # 予測: クラス0
        [0.1, 2.0, 1.0, 0.1, 0.1],  # 予測: クラス1
        [0.1, 0.1, 0.1, 2.0, 1.0],  # 予測: クラス3
        [0.1, 0.1, 2.0, 0.1, 0.1],  # 予測: クラス2
    ])

    labels = torch.tensor([0, 1, 3, 2])  # すべて正解

    accuracy = calculate_accuracy(outputs, labels)
    print(f"\n  出力の形状: {outputs.shape}")
    print(f"  ラベルの値: {labels.tolist()}")
    print(f"  Accuracy: {accuracy:.4f} (期待値: 1.0000)")

    # 一部間違いがある場合
    labels_wrong = torch.tensor([0, 1, 2, 2])  # 3番目が間違い
    accuracy_wrong = calculate_accuracy(outputs, labels_wrong)
    print(f"\n  ラベルの値（一部間違い）: {labels_wrong.tolist()}")
    print(f"  Accuracy: {accuracy_wrong:.4f} (期待値: 0.7500)")


def test_get_device():
    """デバイス選択関数のテスト"""
    print("\n" + "=" * 60)
    print("2. デバイス選択関数のテスト")
    print("-" * 60)

    device = get_device()
    print(f"\n選択されたデバイス: {device}")

    if device.type == "mps":
        print("  Apple Silicon GPU (MPS) を使用")
    elif device.type == "cuda":
        print("  NVIDIA GPU (CUDA) を使用")
    else:
        print("  CPU を使用")


if __name__ == "__main__":
    test_calculate_accuracy()
    test_get_device()

    print("\n" + "=" * 60)
    print("utilsモジュールのテスト完了！")
    print("=" * 60)
