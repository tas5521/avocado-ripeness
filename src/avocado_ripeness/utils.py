"""
ユーティリティ関数モジュール

共通で使用する関数を定義します。
"""

import torch


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
