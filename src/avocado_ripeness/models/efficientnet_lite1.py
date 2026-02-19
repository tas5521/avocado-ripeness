"""
EfficientNet-Lite1モデルの実装

EfficientNet-Lite1を使った転移学習モデル
モバイル推論に最適化されたバージョン。
"""

import torch.nn as nn
import timm


def create_efficientnet_lite1(num_classes=5, pretrained=True, dropout_rate=0.3):
    """
    EfficientNet-Lite1を使った転移学習モデルを作成する

    Args:
        num_classes: 分類クラス数（デフォルト: 5）
        pretrained: 事前訓練済み重みを使用するか（デフォルト: True）
        dropout_rate: ドロップアウト率（デフォルト: 0.3、0.0で無効）

    Returns:
        EfficientNet-Lite1モデル（最終層をnum_classesに置き換え済み、ドロップアウト付き）

    注意:
        EfficientNet-Lite1はモバイル環境での推論に最適化された軽量モデル。
        - Batch Normalization Foldingを使用（推論時の高速化）
        - ReLU6活性化関数を使用（ハードウェア最適化）
        - EfficientNet-B0より軽量で高速な推論が可能
        サイズ: 約4MB、推論速度: モバイルで0.1秒以下程度
    """
    # EfficientNet-Lite1を読み込む（timmを使用）
    # timmでは 'tf_efficientnet_lite1' が正しいモデル名
    if pretrained:
        model = timm.create_model('tf_efficientnet_lite1', pretrained=True)
    else:
        model = timm.create_model('tf_efficientnet_lite1', pretrained=False)

    # 最終層（分類層）を置き換える（ドロップアウト付き）
    # timmのEfficientNet-Liteはclassifierが直接Linear
    num_features = model.classifier.in_features

    if dropout_rate > 0:
        # ドロップアウト + Linear層
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    else:
        # Linear層のみ
        model.classifier = nn.Linear(num_features, num_classes)

    return model


class EfficientNetLite1Model(nn.Module):
    """
    EfficientNet-Lite1を使った転移学習モデル

    EfficientNet-Lite1をラップする。
    モバイル推論に最適化されたバージョン。
    """

    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.3):
        """
        Args:
            num_classes: 分類クラス数（デフォルト: 5）
            pretrained: 事前訓練済み重みを使用するか（デフォルト: True）
            dropout_rate: ドロップアウト率（デフォルト: 0.3、0.0で無効）
        """
        super().__init__()
        self.model = create_efficientnet_lite1(
            num_classes=num_classes, pretrained=pretrained, dropout_rate=dropout_rate)

    def forward(self, x):
        """
        順伝播（forward pass）

        Args:
            x: 入力テンソル [batch_size, channels, height, width]
               例: [4, 3, 224, 224]

        Returns:
            出力テンソル [batch_size, num_classes]
               例: [4, 5]
        """
        return self.model(x)

    def get_model_size_mb(self):
        """
        モデルのサイズをMB単位で取得

        Returns:
            モデルのサイズ（MB）
        """
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = (param_size + buffer_size) / (1024 * 1024)  # MBに変換
        return total_size
