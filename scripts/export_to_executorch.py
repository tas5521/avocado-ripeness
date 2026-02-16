"""
Executorchへの変換スクリプト

訓練済みモデルをExecutorch形式に変換してモバイル推論用にエクスポートする。
"""

import sys
import argparse
import torch
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.model import EfficientNetB0Model  # noqa: E402
from src.avocado_ripeness.config import (  # noqa: E402
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DROPOUT_RATE,
    IMAGE_SIZE
)


def export_to_executorch(
    checkpoint_path,
    output_path,
    num_classes=5,
    dropout_rate=0.3,
    image_size=224
):
    """
    モデルをExecutorch形式にエクスポートする

    Args:
        checkpoint_path: チェックポイントファイルのパス
        output_path: 出力ファイルのパス（.pte形式）
        num_classes: クラス数
        dropout_rate: ドロップアウト率
        image_size: 入力画像サイズ
    """
    print("=" * 60)
    print("Executorchへの変換")
    print("=" * 60)
    print(f"\nチェックポイント: {checkpoint_path}")
    print(f"出力先: {output_path}")
    print(f"入力サイズ: {image_size}x{image_size}")

    # デバイスを設定（CPUでエクスポート）
    device = torch.device("cpu")

    # モデルを作成
    print("\nモデルを作成中...")
    model = EfficientNetB0Model(
        num_classes=num_classes,
        pretrained=False,
        dropout_rate=dropout_rate
    )

    # チェックポイントを読み込む
    print("チェックポイントを読み込み中...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    print("✓ モデルの読み込み完了")

    # ダミー入力を作成（推論時の入力形状に合わせる）
    print(f"\nダミー入力を作成中... (形状: [1, 3, {image_size}, {image_size}])")
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # torch.export を使ってエクスポート
    print("\ntorch.exportでエクスポート中...")
    try:
        exported_program = torch.export.export(model, (dummy_input,))
        print("✓ torch.export完了")
    except Exception as e:
        print(f"✗ torch.exportエラー: {e}")
        print("\n注意: PyTorch 2.1以降が必要です。")
        print("torch.exportの代わりにtorch.jit.traceを使用します...")

        # フォールバック: torch.jit.traceを使用
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.eval()

        # Executorchへの変換（executorchが必要）
        try:
            # torch.jit.traceからは直接executorchに変換できないため、
            # torch.exportを使う必要がある
            print("\n注意: torch.jit.traceからは直接Executorchに変換できません")
            print("torch.exportが使用できない場合は、PyTorch 2.1以降が必要です")
            jit_output = output_path.replace('.pte', '.pt')
            traced_model.save(jit_output)
            print(f"✓ TorchScript形式で保存: {jit_output}")
            print("  注意: これはExecutorch形式ではありません")
            return

            # ファイルに保存
            with open(output_path, "wb") as f:
                f.write(executorch_program.buffer)
            print(f"✓ Executorch形式で保存完了: {output_path}")
            return
        except ImportError:
            print("\n✗ executorchがインストールされていません")
            print("\nインストール方法:")
            print("  pip install executorch")
            print("\nまたは、torch.jit.trace形式で保存しますか？")
            jit_output = output_path.replace('.pte', '.pt')
            traced_model.save(jit_output)
            print(f"✓ TorchScript形式で保存: {jit_output}")
            print("  注意: これはExecutorch形式ではありません")
            return

    # Executorchへの変換
    print("\nexecutorchで変換中...")
    try:
        from executorch.exir import to_edge

        # to_edgeの正しい呼び出し方法（ExportedProgramを直接渡す）
        edge_program = to_edge(exported_program)
        executorch_program = edge_program.to_executorch()

        # ファイルに保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)

        print(f"✓ Executorch形式で保存完了: {output_path}")
        print(f"\nファイルサイズ: {output_path.stat().st_size / (1024*1024):.2f} MB")

    except ImportError:
        print("\n✗ executorchがインストールされていません")
        print("\nインストール方法:")
        print("  pip install executorch")
        print("\nまたは、torch.export形式で保存しますか？")
        export_output = output_path.replace('.pte', '_exported.pt')
        torch.save(exported_program, export_output)
        print(f"✓ torch.export形式で保存: {export_output}")
        print("  注意: これはExecutorch形式ではありません")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="訓練済みモデルをExecutorch形式に変換"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model.pth"),
        help=f"チェックポイントファイルのパス（デフォルト: {CHECKPOINT_DIR / 'best_model.pth'}）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/avocado_ripeness.pte",
        help="出力ファイルのパス（デフォルト: models/avocado_ripeness.pte）"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"クラス数（デフォルト: {NUM_CLASSES}）"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DROPOUT_RATE,
        help=f"ドロップアウト率（デフォルト: {DROPOUT_RATE}）"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help=f"入力画像サイズ（デフォルト: {IMAGE_SIZE}）"
    )

    args = parser.parse_args()

    export_to_executorch(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        image_size=args.image_size
    )

    print("\n" + "=" * 60)
    print("変換完了")
    print("=" * 60)
    print("\n次のステップ:")
    print("1. モバイルアプリでExecutorchランタイムを使用して推論")
    print("2. 入力画像は [1, 3, 224, 224] の形状で正規化済みテンソル")
    print("3. 出力は [1, 5] の形状で各クラスのlogits")


if __name__ == "__main__":
    main()
