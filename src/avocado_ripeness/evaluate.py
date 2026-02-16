"""
評価モジュール

テストセットでの評価機能を提供します。
"""

import unicodedata

import torch
from sklearn.metrics import confusion_matrix, classification_report

from .utils import get_device


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


def _pad_left(s, target_width):
    """文字列を目標の表示幅まで左側をスペースで埋める"""
    padding = target_width - _display_width(s)
    if padding > 0:
        return ' ' * padding + s
    return s


def evaluate_test_set(model, test_dataloader, criterion, device=None):
    """
    テストセット全体を評価する

    Args:
        model: 評価するモデル
        test_dataloader: テストデータのDataLoader
        criterion: 損失関数
        device: 使用するデバイス（Noneの場合は自動選択）

    Returns:
        dict: 評価結果
            {
                'loss': float,
                'accuracy': float,
                'predictions': list,  # 予測クラスのリスト
                'labels': list,  # 正解ラベルのリスト
                'confusion_matrix': array  # numpy配列
            }
    """
    if device is None:
        device = get_device()

    model.eval()

    all_predictions = []
    all_labels = []
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # 予測クラスを取得
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / num_batches

    # Accuracyを計算（予測クラスと正解ラベルを直接比較）
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels) if len(all_labels) > 0 else 0.0

    # Confusion Matrixを計算
    cm = confusion_matrix(all_labels, all_predictions)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'confusion_matrix': cm
    }


def print_confusion_matrix(cm, class_names):
    """
    Confusion Matrixを表示する

    Args:
        cm: Confusion Matrix（numpy配列）
        class_names: クラス名のリスト
    """
    print("\n" + "=" * 60)
    print("Confusion Matrix")
    print("=" * 60)

    # 短い名前を作成（括弧以降を削除）
    short_names = []
    for name in class_names:
        if " (" in name:
            short_names.append(name[:name.index(" (")])
        else:
            short_names.append(name)

    # 列幅を計算（全角文字を考慮）
    col_width = 8  # 数値列の幅
    row_width = max(_display_width(name) for name in short_names) + 2

    # ヘッダーを表示
    print(f"\n{_pad_right('予測 →', row_width)}", end="")
    for name in short_names:
        print(_pad_left(name, col_width), end="")
    print(f"{'合計':>6}")

    # 各行を表示（対角線のセルは[数値]で囲む）
    for i, name in enumerate(short_names):
        print(_pad_right(name, row_width), end="")
        for j in range(len(short_names)):
            count = cm[i][j]
            if i == j:
                cell = f"[{count}]"
            else:
                cell = str(count)
            print(f"{cell:>{col_width}}", end="")
        print(f"{cm[i].sum():>6}")

    # 合計行を表示
    print(_pad_right("合計", row_width), end="")
    for j in range(len(short_names)):
        print(f"{cm[:, j].sum():>{col_width}}", end="")
    print(f"{cm.sum():>6}")


def print_classification_report(labels, predictions, class_names):
    """
    クラスごとの精度指標を表示する

    Args:
        labels: 正解ラベルのリスト
        predictions: 予測クラスのリスト
        class_names: クラス名のリスト
    """
    print("\n" + "=" * 60)
    print("クラスごとの精度指標")
    print("=" * 60)

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # クラス名列の表示幅（全角文字を考慮）
    name_width = max(_display_width(name) for name in class_names)
    name_width = max(name_width, 20)

    header = _pad_right('クラス', name_width)
    print(f"\n{header} {'Precision':>12} {'Recall':>12} {'F1-score':>12} {'Support':>12}")
    print("-" * (name_width + 12 * 4 + 4))

    for name in class_names:
        metrics = report[name]
        padded = _pad_right(name, name_width)
        print(f"{padded} {metrics['precision']:>12.4f} {metrics['recall']:>12.4f} "
              f"{metrics['f1-score']:>12.4f} {metrics['support']:>12.0f}")

    print("-" * (name_width + 12 * 4 + 4))
    padded = _pad_right('平均 (macro)', name_width)
    print(f"{padded} {report['macro avg']['precision']:>12.4f} "
          f"{report['macro avg']['recall']:>12.4f} {report['macro avg']['f1-score']:>12.4f} "
          f"{report['macro avg']['support']:>12.0f}")
    padded = _pad_right('平均 (weighted)', name_width)
    print(f"{padded} {report['weighted avg']['precision']:>12.4f} "
          f"{report['weighted avg']['recall']:>12.4f} {report['weighted avg']['f1-score']:>12.4f} "
          f"{report['weighted avg']['support']:>12.0f}")
    padded = _pad_right('全体Accuracy', name_width)
    print(f"{padded} {report['accuracy']:>12.4f} {'':>12} {'':>12} "
          f"{len(labels):>12.0f}")
