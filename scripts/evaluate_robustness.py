"""
2つのチェックポイント（拡張あり学習 / なし学習）を、同一の摂動付きテストで比較する。

各presetごとにテストAccuracyを算出し、頑健性の比較に使う。
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.predict import load_model_from_checkpoint  # noqa: E402
from src.avocado_ripeness.dataloader import create_dataloader  # noqa: E402
from src.avocado_ripeness.evaluate import evaluate_test_set, _pad_right, _pad_left  # noqa: E402
from src.avocado_ripeness.robustness import (  # noqa: E402
    get_perturbation_eval_transform,
    list_perturbation_presets,
    describe_preset,
)
from src.avocado_ripeness.config import (  # noqa: E402
    TEST_DIR,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DROPOUT_RATE,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASS_MODE,
)
from src.avocado_ripeness.utils import get_device  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="摂動下で2チェックポイントのテスト Accuracy を比較する"
    )
    parser.add_argument(
        "--checkpoint-no-aug",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model.pth"),
        help="拡張なしで学習したチェックポイント（デフォルト: checkpoints/best_model.pth）",
    )
    parser.add_argument(
        "--checkpoint-aug",
        type=str,
        default=str(CHECKPOINT_DIR / "best_model_augumented.pth"),
        help="拡張ありで学習したチェックポイント（デフォルト: checkpoints/best_model_augumented.pth）",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help=f"クラス数（デフォルト: {NUM_CLASSES}）",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DROPOUT_RATE,
        help=f"ドロップアウト率（デフォルト: {DROPOUT_RATE}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"バッチサイズ（デフォルト: {BATCH_SIZE}）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="mps / cuda / cpu（未指定時は自動）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"DataLoader のワーカー数（デフォルト: {NUM_WORKERS}。不安定な環境では 0 を推奨）",
    )
    parser.add_argument(
        "--presets",
        type=str,
        nargs="*",
        default=None,
        help="評価する preset 名（省略時はすべて）。例: clean brightness_0.75 rotate_10",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    all_presets = list_perturbation_presets()
    if args.presets:
        for p in args.presets:
            if p not in all_presets:
                print(f"エラー: 不明な preset {p!r}。利用可能: {all_presets}")
                sys.exit(1)
        presets = args.presets
    else:
        presets = all_presets

    criterion = nn.CrossEntropyLoss()

    models = {}
    print("モデルを読み込み中...")
    models["拡張なし学習"] = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_no_aug,
        num_classes=args.num_classes,
        device=device,
        dropout_rate=args.dropout_rate,
    )
    models["拡張あり学習"] = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_aug,
        num_classes=args.num_classes,
        device=device,
        dropout_rate=args.dropout_rate,
    )
    print("✓ 読み込み完了\n")

    # preset × モデル で accuracy を格納
    table = {p: {} for p in presets}
    n_samples = None

    for preset in presets:
        transform = get_perturbation_eval_transform(preset)
        test_dataloader = create_dataloader(
            data_dir=TEST_DIR,
            batch_size=args.batch_size,
            shuffle=False,
            transform=transform,
            num_workers=args.num_workers,
            num_classes=args.num_classes,
            class_mode=CLASS_MODE,
        )
        if n_samples is None:
            n_samples = len(test_dataloader.dataset)
        for label, model in models.items():
            print(f"  評価中: preset={preset} / {label} ...", flush=True)
            results = evaluate_test_set(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                device=device,
            )
            table[preset][label] = results["accuracy"]

    # 表示
    print("=" * 72)
    print("摂動下テスト Accuracy 比較（同一テストセット・同一摂動条件）")
    print("=" * 72)
    print(f"使用デバイス: {device}")
    print(f"テストディレクトリ: {TEST_DIR}")
    print(f"拡張なし学習: {args.checkpoint_no_aug}")
    print(f"拡張あり学習: {args.checkpoint_aug}")
    print(f"サンプル数: {n_samples}")
    print()

    preset_w = 30
    desc_w = 30
    num_w = 10

    header = (
        _pad_right("preset", preset_w)
        + _pad_right("説明", desc_w)
        + _pad_left("拡張なし", num_w)
        + _pad_left("拡張あり", num_w)
    )
    print(header)
    print("-" * (preset_w + desc_w + num_w * 2))

    for preset in presets:
        a_no = table[preset]["拡張なし学習"]
        a_aug = table[preset]["拡張あり学習"]
        desc = describe_preset(preset)
        row = (
            _pad_right(preset, preset_w)
            + _pad_right(desc, desc_w)
            + _pad_left(f"{a_no:.4f}", num_w)
            + _pad_left(f"{a_aug:.4f}", num_w)
        )
        print(row)

    # clean がある場合、各モデルの clean → 摂動での劣化幅を表示
    if "clean" in table and len(presets) > 1:
        clean_no = table["clean"]["拡張なし学習"]
        clean_aug = table["clean"]["拡張あり学習"]
        print()
        print(_pad_right("", preset_w)
              + _pad_right("", desc_w)
              + _pad_left("拡張なし", num_w)
              + _pad_left("拡張あり", num_w))
        print(_pad_right("", preset_w)
              + _pad_right("(cleanとの差)", desc_w)
              + _pad_left("劣化幅", num_w)
              + _pad_left("劣化幅", num_w))
        print("-" * (preset_w + desc_w + num_w * 2))
        for preset in presets:
            if preset == "clean":
                continue
            d_no = table[preset]["拡張なし学習"] - clean_no
            d_aug = table[preset]["拡張あり学習"] - clean_aug
            desc = describe_preset(preset)
            row = (
                _pad_right(preset, preset_w)
                + _pad_right(desc, desc_w)
                + _pad_left(f"{d_no:+.4f}", num_w)
                + _pad_left(f"{d_aug:+.4f}", num_w)
            )
            print(row)
        print()
        print("※ 劣化幅 = 各 preset の Accuracy − clean の Accuracy（負なら摂動で精度低下）。")
        print("  劣化幅が小さい（0 に近い）ほど、その摂動に対して頑健。")

    print("=" * 72)


if __name__ == "__main__":
    main()
