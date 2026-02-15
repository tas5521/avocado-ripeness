"""
バリデーションループの動作確認

train.pyのvalidate_one_epoch関数をテストする。
- バリデーションループが実行できるか
- model.eval()とtorch.no_grad()が正しく使われているか
- 損失とAccuracyが計算されるか
"""

import torch.nn as nn
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.model import EfficientNetB0Model  # noqa: E402
from src.avocado_ripeness.dataloader import (  # noqa: E402
    create_dataloader,
    get_valid_transforms
)
from src.avocado_ripeness.train import validate_one_epoch  # noqa: E402
from src.avocado_ripeness.utils import get_device  # noqa: E402

# データパス
DATA_DIR = Path("data/processed/avocado_ripeness")


def test_validate_one_epoch():
    """バリデーションループのテスト"""
    print("=" * 60)
    print("Step 7: バリデーションループの動作確認")
    print("=" * 60)

    # デバイスを取得
    device = get_device()
    print(f"\n使用デバイス: {device}")

    # モデルを作成（事前訓練済みモデルを使用）
    model = EfficientNetB0Model(num_classes=5, pretrained=True)
    model = model.to(device)
    print("\nモデルを作成しました")

    # 損失関数を作成
    criterion = nn.CrossEntropyLoss()
    print("損失関数を作成しました")

    # バリデーションデータローダーを作成
    valid_dir = DATA_DIR / "valid"
    valid_transform = get_valid_transforms()

    print("\nバリデーションデータローダーを作成中...")
    val_dataloader = create_dataloader(
        data_dir=valid_dir,
        batch_size=32,
        shuffle=False,
        transform=valid_transform,
        num_workers=0
    )

    print(f"  データセットサイズ: {len(val_dataloader.dataset)}")
    print(f"  バッチ数: {len(val_dataloader)}")
    print("  バッチサイズ: 32")
    print("  シャッフル: False")

    # バリデーションを実行
    print("\n" + "=" * 60)
    print("バリデーションを開始...")
    print("=" * 60)

    results = validate_one_epoch(
        model=model,
        dataloader=val_dataloader,
        criterion=criterion,
        device=device
    )

    # 結果を表示
    print("\n" + "=" * 60)
    print("バリデーション結果")
    print("=" * 60)
    print(f"平均損失: {results['loss']:.4f}")
    print(f"平均Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    test_validate_one_epoch()
