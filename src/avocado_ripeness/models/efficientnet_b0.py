"""
EfficientNet-B0モデルの実装

EfficientNet-B0を使った転移学習モデル
"""

import torch.nn as nn
import torchvision.models as models


def create_efficientnet_b0(num_classes=5, pretrained=True, dropout_rate=0.3):
    """
    EfficientNet-B0を使った転移学習モデルを作成する

    Args:
        num_classes: 分類クラス数（デフォルト: 5）
        pretrained: 事前訓練済み重みを使用するか（デフォルト: True）
        dropout_rate: ドロップアウト率（デフォルト: 0.3、0.0で無効）

    Returns:
        EfficientNet-B0モデル（最終層をnum_classesに置き換え済み、ドロップアウト付き）

    注意:
        EfficientNet-B0はモバイル環境での推論に適した軽量モデル。
        サイズ: 約5MB、推論速度: モバイルで0.1-0.5秒程度
    """
    # EfficientNet-B0を読み込む
    # weights='IMAGENET1K_V1'で事前訓練済み重みを使用
    if pretrained:
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    else:
        model = models.efficientnet_b0(weights=None)

    # 最終層（分類層）を置き換える（ドロップアウト付き）
    num_features = model.classifier[1].in_features  # 元のクラス数を取得

    if dropout_rate > 0:
        # ドロップアウト + Linear層
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    else:
        # Linear層のみ
        model.classifier[1] = nn.Linear(num_features, num_classes)

    return model


class EfficientNetB0Model(nn.Module):
    """
    EfficientNet-B0を使った転移学習モデル

    EfficientNet-B0をラップする。
    """

    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.3):
        """
        Args:
            num_classes: 分類クラス数（デフォルト: 5）
            pretrained: 事前訓練済み重みを使用するか（デフォルト: True）
            dropout_rate: ドロップアウト率（デフォルト: 0.3、0.0で無効）
        """
        super().__init__()
        self.model = create_efficientnet_b0(
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
