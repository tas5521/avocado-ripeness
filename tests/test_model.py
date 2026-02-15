"""
モデルが正しく動作するかを確認する。
- モデルの作成
- データがモデルを通ることを確認
- 出力の形状を確認
"""

import torch
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.model import (  # noqa: E402
    create_efficientnet_b0,
    EfficientNetB0Model
)
from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_train_transforms
)

# データパス
DATA_DIR = Path("data/processed/avocado_ripeness")


def test_efficientnet_b0_creation():
    """EfficientNet-B0モデルの作成確認"""
    print("\n" + "=" * 60)
    print("EfficientNet-B0転移学習モデルの動作確認")
    print("=" * 60)

    # モデルを作成
    print("\n1. EfficientNet-B0モデルを作成中...")
    model = create_efficientnet_b0(num_classes=5, pretrained=True)
    print("   モデル作成成功！")

    # モデルの構造を確認
    print("\n   モデルの構造（最後の部分）:")
    print(f"   {model.classifier}")

    # パラメータ数を確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   総パラメータ数: {total_params:,}")
    print(f"   学習可能パラメータ数: {trainable_params:,}")

    # モデルサイズを確認
    wrapper_model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model_size_mb = wrapper_model.get_model_size_mb()
    print(f"   モデルサイズ: {model_size_mb:.2f} MB")
    print("   （モバイル環境での推論に適したサイズ）")


def test_efficientnet_b0_forward():
    """EfficientNet-B0モデルにデータを通す"""
    print("\n" + "=" * 60)
    print("2. EfficientNet-B0モデルにデータを通す")
    print("=" * 60)

    # モデルを作成
    model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model.eval()
    print("\nモデルを作成しました（評価モード）")

    # ダミーデータを作成
    print("\nダミーデータを作成中...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    print(f"   入力の形状: {dummy_input.shape}")

    # モデルにデータを入力
    print("\nモデルにデータを入力中...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n   出力の形状: {output.shape}")
    print("   期待値: [batch_size, num_classes] = [4, 5]")
    print(f"   実際: {output.shape}")

    # 出力の値を確認
    print("\n   出力の値（最初のバッチ）:")
    print(f"   {output[0].tolist()}")


def test_efficientnet_b0_with_real_data():
    """実際のデータでEfficientNet-B0をテスト"""
    print("\n" + "=" * 60)
    print("3. 実際のデータでEfficientNet-B0をテスト")
    print("=" * 60)

    # モデルを作成
    model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model.eval()

    # データローダーを作成
    train_dir = DATA_DIR / "train"
    train_transform = get_train_transforms()

    print("\nデータローダーを作成中...")
    dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=4,
        shuffle=True,
        transform=train_transform,
        num_workers=0
    )

    # バッチを取得
    print("\nバッチを取得中...")
    images, labels = next(iter(dataloader))

    print(f"   画像の形状: {images.shape}")
    print(f"   ラベルの形状: {labels.shape}")
    print(f"   ラベルの値: {labels.tolist()}")

    # モデルにデータを入力
    print("\nモデルにデータを入力中...")
    import time
    start_time = time.time()
    with torch.no_grad():
        output = model(images)
    inference_time = time.time() - start_time

    print(f"\n   出力の形状: {output.shape}")
    print(f"   推論時間: {inference_time*1000:.2f} ms（CPU）")
    print(f"   1画像あたり: {inference_time*1000/len(images):.2f} ms")

    # 各画像の出力を確認
    print("\n   各画像の出力（5クラスのスコア）:")
    for i in range(len(images)):
        print(f"\n   画像 {i+1}:")
        print(f"     ラベル: {labels[i].item()} (クラス {labels[i].item() + 1})")
        # Softmaxで確率に変換
        probs = torch.softmax(output[i], dim=0)
        predicted_class = output[i].argmax().item()
        predicted_prob = probs[predicted_class].item()
        print(f"     予測クラス: {predicted_class} (確率: {predicted_prob:.3f})")
        print(f"     全クラスの確率: {[f'{p:.3f}' for p in probs.tolist()]}")


if __name__ == "__main__":
    test_efficientnet_b0_creation()
    test_efficientnet_b0_forward()
    test_efficientnet_b0_with_real_data()
