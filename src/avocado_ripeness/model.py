"""
CNNモデルの定義モジュール（後方互換性のためのラッパー）

このモジュールは後方互換性のために残されています。
新しいコードでは `from src.avocado_ripeness.models import ...` を使用することを推奨します。
"""

# models/から全てを再エクスポート（後方互換性のため）
from .models import (  # noqa: F401
    create_efficientnet_b0,
    EfficientNetB0Model,
    create_efficientnet_lite0,
    EfficientNetLite0Model,
    create_efficientnet_lite1,
    EfficientNetLite1Model
)

__all__ = [
    'create_efficientnet_b0',
    'EfficientNetB0Model',
    'create_efficientnet_lite0',
    'EfficientNetLite0Model',
    'create_efficientnet_lite1',
    'EfficientNetLite1Model',
]
