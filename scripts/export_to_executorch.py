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

from src.avocado_ripeness.model import (  # noqa: E402
    EfficientNetB0Model,
    EfficientNetLite0Model,
    EfficientNetLite1Model
)
from src.avocado_ripeness.config import (  # noqa: E402
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DROPOUT_RATE,
    IMAGE_SIZE,
    MODEL_NAME
)


def _create_model(model_name, num_classes, dropout_rate):
    """
    モデルを作成する

    Args:
        model_name: モデル名
        num_classes: クラス数
        dropout_rate: ドロップアウト率

    Returns:
        作成されたモデル
    """
    print("\nモデルを作成中...")
    if model_name == "efficientnet_lite0":
        model = EfficientNetLite0Model(
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=dropout_rate
        )
        print("  モデル: EfficientNet-Lite0")
    elif model_name == "efficientnet_lite1":
        model = EfficientNetLite1Model(
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=dropout_rate
        )
        print("  モデル: EfficientNet-Lite1")
    elif model_name == "efficientnet_b0":
        model = EfficientNetB0Model(
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=dropout_rate
        )
        print("  モデル: EfficientNet-B0")
    else:
        raise ValueError(
            f"不明なモデル名: {model_name}。"
            f"サポートされているモデル: 'efficientnet_b0', 'efficientnet_lite0', 'efficientnet_lite1'"
        )
    return model


def _load_checkpoint(model, checkpoint_path, device):
    """
    チェックポイントからモデルを読み込む

    Args:
        model: モデルインスタンス
        checkpoint_path: チェックポイントファイルのパス
        device: デバイス
    """
    print("チェックポイントを読み込み中...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    print("✓ モデルの読み込み完了")


def _export_with_torch_export(model, dummy_input, output_path):
    """
    torch.exportを使用してモデルをエクスポートする

    Args:
        model: モデルインスタンス
        dummy_input: ダミー入力
        output_path: 出力ファイルのパス

    Returns:
        ExportedProgramまたはNone（失敗時）
    """
    print("\ntorch.exportでエクスポート中...")
    try:
        exported_program = torch.export.export(model, (dummy_input,))
        print("✓ torch.export完了")
        return exported_program
    except Exception as e:
        print(f"✗ torch.exportエラー: {e}")
        print("\n注意: PyTorch 2.1以降が必要です。")
        print("torch.exportの代わりにtorch.jit.traceを使用します...")
        _fallback_to_torchscript(model, dummy_input, output_path)
        return None


def _fallback_to_torchscript(model, dummy_input, output_path):
    """
    torch.jit.traceを使用してフォールバック（Executorchには変換できない）

    Args:
        model: モデルインスタンス
        dummy_input: ダミー入力
        output_path: 出力ファイルのパス
    """
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.eval()
    print("\n注意: torch.jit.traceからは直接Executorchに変換できません")
    print("torch.exportが使用できない場合は、PyTorch 2.1以降が必要です")
    jit_output = output_path.replace('.pte', '.pt')
    traced_model.save(jit_output)
    print(f"✓ TorchScript形式で保存: {jit_output}")
    print("  注意: これはExecutorch形式ではありません")


def _convert_to_executorch(exported_program, output_path):
    """
    ExportedProgramをExecutorch形式に変換する

    Args:
        exported_program: torch.exportでエクスポートされたプログラム
        output_path: 出力ファイルのパス
    """
    print("\nexecutorchで変換中...")
    try:
        from executorch.exir import to_edge

        edge_program = to_edge(exported_program)
        executorch_program = edge_program.to_executorch()

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


def export_to_executorch(
    checkpoint_path,
    output_path,
    num_classes=5,
    dropout_rate=0.3,
    image_size=224,
    model_name=None
):
    """
    モデルをExecutorch形式にエクスポートする

    Args:
        checkpoint_path: チェックポイントファイルのパス
        output_path: 出力ファイルのパス（.pte形式）
        num_classes: クラス数
        dropout_rate: ドロップアウト率
        image_size: 入力画像サイズ
        model_name: モデル名（Noneの場合はconfigから読み取る）
    """
    print("=" * 60)
    print("Executorchへの変換")
    print("=" * 60)
    print(f"\nチェックポイント: {checkpoint_path}")
    print(f"出力先: {output_path}")
    print(f"入力サイズ: {image_size}x{image_size}")

    if model_name is None:
        model_name = MODEL_NAME

    device = torch.device("cpu")
    model = _create_model(model_name, num_classes, dropout_rate)
    _load_checkpoint(model, checkpoint_path, device)

    print(f"\nダミー入力を作成中... (形状: [1, 3, {image_size}, {image_size}])")
    dummy_input = torch.randn(1, 3, image_size, image_size)

    exported_program = _export_with_torch_export(model, dummy_input, output_path)
    if exported_program is not None:
        _convert_to_executorch(exported_program, output_path)


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
