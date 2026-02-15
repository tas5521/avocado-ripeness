"""
推論ロジックモジュール

訓練済みモデルで推論を実行する機能を実装
"""

import torch
from PIL import Image

from .model import EfficientNetB0Model
from .dataloader import get_valid_transforms
from .utils import get_device


def load_model_from_checkpoint(checkpoint_path, num_classes=5, device=None, dropout_rate=0.3):
    """
    チェックポイントからモデルを読み込む

    Args:
        checkpoint_path: チェックポイントファイルのパス
        num_classes: クラス数（デフォルト: 5）
        device: 使用するデバイス（Noneの場合は自動選択）
        dropout_rate: ドロップアウト率（チェックポイント保存時の設定に合わせる）

    Returns:
        model: 読み込んだモデル（評価モード）
    """
    if device is None:
        device = get_device()

    # モデルを作成
    model = EfficientNetB0Model(
        num_classes=num_classes,
        pretrained=False,  # チェックポイントから読み込むのでFalse
        dropout_rate=dropout_rate
    )

    # チェックポイントを読み込む
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # モデルを評価モードに設定
    model.eval()
    model = model.to(device)

    return model


def preprocess_image(image_path, transform=None):
    """
    画像を前処理して推論用のテンソルに変換する

    Args:
        image_path: 画像ファイルのパス
        transform: 前処理のtransforms（Noneの場合はバリデーション用を使用）

    Returns:
        tensor: 前処理済みの画像テンソル [1, 3, 224, 224]
    """
    if transform is None:
        transform = get_valid_transforms()

    # 画像を読み込む
    image = Image.open(image_path).convert("RGB")

    # 前処理を適用
    tensor = transform(image)

    # バッチ次元を追加 [3, 224, 224] → [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)

    return tensor


def predict_single_image(model, image_path, device=None, class_names=None):
    """
    単一画像の推論を実行する

    Args:
        model: 訓練済みモデル
        image_path: 画像ファイルのパス
        device: 使用するデバイス（Noneの場合は自動選択）
        class_names: クラス名のリスト（Noneの場合は0, 1, 2...を使用）

    Returns:
        dict: 推論結果
            {
                'predicted_class': int,  # 予測クラス
                'predicted_class_name': str,  # 予測クラス名
                'probabilities': list,  # 各クラスの確率
                'top_k': list,  # 上位kクラス（確率順）
            }
    """
    if device is None:
        device = get_device()

    # クラス数を取得（ドロップアウトがある場合とない場合に対応）
    classifier = model.model.classifier[1]
    if isinstance(classifier, torch.nn.Sequential):
        # ドロップアウト + Linear層の場合
        num_classes = classifier[-1].out_features
    else:
        # Linear層のみの場合
        num_classes = classifier.out_features

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 画像を前処理
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # 推論を実行
    with torch.no_grad():
        outputs = model(image_tensor)

        # Softmaxで確率に変換
        probabilities = torch.softmax(outputs, dim=1)

        # 予測クラスを取得
        predicted_class = probabilities.argmax(dim=1).item()

        # 確率をリストに変換
        prob_list = probabilities[0].cpu().tolist()

    # 上位kクラスを取得（確率順）
    top_k = sorted(
        [(i, prob_list[i], class_names[i]) for i in range(len(prob_list))],
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'predicted_class': predicted_class,
        'predicted_class_name': class_names[predicted_class],
        'probabilities': prob_list,
        'top_k': top_k
    }


def predict_batch(model, image_paths, device=None, class_names=None):
    """
    複数画像のバッチ推論を実行する

    Args:
        model: 訓練済みモデル
        image_paths: 画像ファイルのパスのリスト
        device: 使用するデバイス（Noneの場合は自動選択）
        class_names: クラス名のリスト（Noneの場合は0, 1, 2...を使用）

    Returns:
        list: 各画像の推論結果のリスト
    """
    if device is None:
        device = get_device()

    results = []
    for image_path in image_paths:
        result = predict_single_image(model, image_path, device, class_names)
        results.append(result)

    return results
