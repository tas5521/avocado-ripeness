"""
データ分布確認スクリプト

訓練データ、バリデーションデータ、テストデータの各クラスの分布を確認します。
"""

import sys
import unicodedata
from pathlib import Path
from collections import Counter


def _display_width(s):
    """文字列の表示幅を計算する（全角文字は2、半角文字は1）"""
    width = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


def _pad_right(s, target_width):
    """文字列を目標の表示幅まで右側をスペースで埋める"""
    padding = target_width - _display_width(s)
    if padding > 0:
        return s + ' ' * padding
    return s

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.dataset import AvocadoDataset  # noqa: E402
from src.avocado_ripeness.config import (  # noqa: E402
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    NUM_CLASSES
)
from src.avocado_ripeness.utils import calculate_class_weights  # noqa: E402


def _get_class_names(num_classes):
    """クラス数に応じた表示名を返す"""
    if num_classes == 3:
        return {0: "未熟", 1: "適熟", 2: "過熟"}
    return {
        0: "未熟 (1)",
        1: "やや未熟 (2)",
        2: "適熟 (3)",
        3: "やや過熟 (4)",
        4: "過熟 (5)"
    }


def check_distribution(data_dir, split_name):
    """
    データセットの分布を確認する

    Args:
        data_dir: データディレクトリのパス
        split_name: データセット名（train/valid/test）

    Returns:
        dict: クラスインデックスをキー、データ数を値とする辞書
    """
    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} データセットの分布")
    print(f"{'=' * 60}")

    if not data_dir.exists():
        print(f"エラー: {data_dir} が見つかりません")
        return None

    # データセットを作成
    dataset = AvocadoDataset(data_dir, transform=None, num_classes=NUM_CLASSES)

    # クラスごとのデータ数をカウント
    class_counts = Counter(dataset.targets)

    # クラス名のマッピング
    class_names = _get_class_names(NUM_CLASSES)

    # 結果を表示
    total = len(dataset)
    name_width = max(_display_width(n) for n in class_names.values()) + 2
    print(f"\n総データ数: {total}")
    print(f"\n{_pad_right('クラス', name_width)} {'データ数':>10} {'割合':>10}")
    print("-" * 60)

    # クラス順に表示
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = (count / total) * 100 if total > 0 else 0
        class_name = class_names.get(class_idx, f"クラス {class_idx}")

        # バーグラフ（最大20文字）
        bar_length = int((count / max(class_counts.values()))
                         * 20) if class_counts else 0
        bar = "█" * bar_length

        padded = _pad_right(class_name, name_width)
        print(f"{padded} {count:>10} {percentage:>9.1f}% {bar}")

    # 統計情報
    counts_list = list(class_counts.values())
    if counts_list:
        min_count = min(counts_list)
        max_count = max(counts_list)
        avg_count = sum(counts_list) / len(counts_list)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print("-" * 60)
        print("\n統計情報:")
        print(f"  最小データ数: {min_count}")
        print(f"  最大データ数: {max_count}")
        print(f"  平均データ数: {avg_count:.1f}")
        print(f"  不均衡比 (最大/最小): {imbalance_ratio:.2f}")

        if imbalance_ratio > 2.0:
            print("  ⚠️  警告: データの不均衡が大きいです（比率 > 2.0）")
        elif imbalance_ratio > 1.5:
            print("  ⚠️  注意: データにやや不均衡があります（比率 > 1.5）")
        else:
            print("  ✓ データは比較的均等に分布しています")

    return class_counts


def main():
    """メイン関数"""
    print("=" * 60)
    print("データ分布確認")
    print("=" * 60)

    # 各データセットの分布を確認
    train_dist = check_distribution(TRAIN_DIR, "train")
    valid_dist = check_distribution(VALID_DIR, "valid")
    test_dist = check_distribution(TEST_DIR, "test")

    # 全体の分布を表示
    print(f"\n{'=' * 60}")
    print("全体の分布")
    print(f"{'=' * 60}")

    if train_dist and valid_dist and test_dist:
        class_names = _get_class_names(NUM_CLASSES)

        name_width = max(_display_width(n) for n in class_names.values()) + 2
        header = _pad_right('クラス', name_width)
        print(f"\n{header} {'Train':>10} {'Valid':>10} {'Test':>10} {'合計':>10}")
        print("-" * 60)

        for class_idx in range(NUM_CLASSES):
            train_count = train_dist.get(class_idx, 0)
            valid_count = valid_dist.get(class_idx, 0)
            test_count = test_dist.get(class_idx, 0)
            total_count = train_count + valid_count + test_count

            class_name = class_names.get(class_idx, f"クラス {class_idx}")
            padded = _pad_right(class_name, name_width)
            print(
                f"{padded} {train_count:>10} {valid_count:>10}"
                f" {test_count:>10} {total_count:>10}"
            )


    # クラス重みを表示（訓練データから計算）
    if train_dist:
        print(f"\n{'=' * 60}")
        print("クラス重み（訓練データから計算）")
        print(f"{'=' * 60}")
        print("※ 少数クラスほど大きい重みになります\n")

        train_dataset = AvocadoDataset(TRAIN_DIR, transform=None, num_classes=NUM_CLASSES)
        weights = calculate_class_weights(train_dataset, NUM_CLASSES)

        class_names = _get_class_names(NUM_CLASSES)

        name_width = max(_display_width(n) for n in class_names.values()) + 2
        header = _pad_right('クラス', name_width)
        print(f"{header} {'重み':>10}")
        print("-" * 30)

        for i in range(NUM_CLASSES):
            class_name = class_names.get(i, f"クラス {i}")
            padded = _pad_right(class_name, name_width)
            print(f"{padded} {weights[i]:>10.4f}")


if __name__ == "__main__":
    main()
