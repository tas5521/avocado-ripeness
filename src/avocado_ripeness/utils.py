"""
ユーティリティ関数モジュール

共通で使用する関数を定義します。
"""

import torch
from collections import Counter


def get_device():
    """
    使用可能なデバイスを自動選択

    Returns:
        torch.device: 使用するデバイス（mps/cuda/cpu）
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def calculate_accuracy(outputs, labels):
    """
    Accuracyを計算する

    Args:
        outputs: モデルの出力 [batch_size, num_classes]
        labels: 正解ラベル [batch_size]

    Returns:
        float: Accuracy（0.0-1.0）
    """
    # 予測クラスを取得（最大値のインデックス）
    _, predicted = torch.max(outputs, 1)

    # 正解数を計算
    correct = (predicted == labels).sum().item()

    # Accuracyを計算
    accuracy = correct / len(labels)

    return accuracy


def calculate_class_weights(dataset, num_classes):
    """
    データセットのクラス分布から逆頻度重みを計算する

    少数クラスほど大きい重み、多数クラスほど小さい重みになる。
    重みの合計がクラス数になるように正規化する。

    Args:
        dataset: AvocadoDataset（targetsプロパティを持つ）
        num_classes: クラス数

    Returns:
        torch.Tensor: 各クラスの重み [num_classes]
    """
    class_counts = Counter(dataset.targets)
    total = len(dataset)

    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
