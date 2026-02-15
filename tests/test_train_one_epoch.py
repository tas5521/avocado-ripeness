"""
Step 6: 1エポック分の訓練ループの動作確認

train.pyの関数をテストします。
- 1エポック分の訓練が実行できるか
- 損失が計算されるか
- Accuracyが計算されるか
- プログレスバーが表示されるか
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from src.avocado_ripeness.model import EfficientNetB0Model  # noqa: E402
from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_train_transforms
)
from src.avocado_ripeness.train import train_one_epoch  # noqa: E402
from src.avocado_ripeness.utils import (  # noqa: E402
    get_device,
    calculate_accuracy
)

# データパス
DATA_DIR = Path("data/processed/avocado_ripeness")


def test_train_one_epoch():
    """1エポック分の訓練を実行"""
    print("=" * 60)
    print("Step 6: 1エポック分の訓練ループの動作確認")
    print("=" * 60)

    # デバイスを取得
    device = get_device()
    print(f"\n使用デバイス: {device}")

    # モデルを作成
    model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model = model.to(device)
    print("\nモデルを作成しました")

    # 損失関数とオプティマイザーを作成
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("損失関数とオプティマイザーを作成しました")

    # データローダーを作成（小さなバッチサイズでテスト）
    train_dir = DATA_DIR / "train"
    train_transform = get_train_transforms()

    print("\nデータローダーを作成中...")
    dataloader = create_dataloader(
        data_dir=train_dir,
        batch_size=32,
        shuffle=True,
        transform=train_transform,
        num_workers=0
    )

    print(f"  データセットサイズ: {len(dataloader.dataset)}")
    print(f"  バッチ数: {len(dataloader)}")
    print(f"  バッチサイズ: 32")

    # 訓練前の状態を確認
    print("\n訓練前の状態を確認中...")
    model.eval()
    with torch.no_grad():
        # 最初のバッチで確認
        images, labels = next(iter(dataloader))
        images = images.to(device)
        labels = labels.to(device)
        outputs_before = model(images)
        loss_before = criterion(outputs_before, labels).item()
        accuracy_before = calculate_accuracy(outputs_before, labels)

    print(f"  損失: {loss_before:.4f}")
    print(f"  Accuracy: {accuracy_before:.4f}")

    # 1エポック訓練を実行
    print("\n" + "=" * 60)
    print("1エポック分の訓練を開始...")
    print("=" * 60)

    results = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    # 訓練後の状態を確認
    print("\n訓練後の状態を確認中...")
    model.eval()
    with torch.no_grad():
        outputs_after = model(images)
        loss_after = criterion(outputs_after, labels).item()
        accuracy_after = calculate_accuracy(outputs_after, labels)

    print(f"  損失: {loss_after:.4f}")
    print(f"  Accuracy: {accuracy_after:.4f}")

    # 結果を表示
    print("\n" + "=" * 60)
    print("訓練結果")
    print("=" * 60)
    print(f"平均損失: {results['loss']:.4f}")
    print(f"平均Accuracy: {results['accuracy']:.4f}")

    print("\n訓練前後の比較:")
    print(f"  損失: {loss_before:.4f} → {loss_after:.4f}")
    print(f"  Accuracy: {accuracy_before:.4f} → {accuracy_after:.4f}")

    # 改善の確認
    if results['loss'] < loss_before:
        print("\n✓ 損失が改善しました！")
    if results['accuracy'] > accuracy_before:
        print("✓ Accuracyが改善しました！")


if __name__ == "__main__":
    test_train_one_epoch()

    print("\n" + "=" * 60)
    print("Step 6 完了！")
    print("=" * 60)
    print("\n理解すべきポイント:")
    print("1. 1エポック = データセット全体を1回処理")
    print("2. 各バッチで訓練ステップを実行")
    print("3. 損失とAccuracyを累積して平均を計算")
    print("4. プログレスバーで進捗を確認")
    print("5. モデルが改善しているか確認（損失が下がる、Accuracyが上がる）")
    print("=" * 60)
