"""
推論機能の動作確認

predict.pyの機能をテストする。
- チェックポイントからのモデル読み込み
- 単一画像の推論
- 推論結果の表示
"""

import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.avocado_ripeness.predict import (  # noqa: E402
    load_model_from_checkpoint,
    preprocess_image,
    predict_single_image
)
from src.avocado_ripeness.config import (  # noqa: E402
    CHECKPOINT_DIR,
    NUM_CLASSES,
    DROPOUT_RATE,
    TEST_DIR
)
from src.avocado_ripeness.utils import get_device  # noqa: E402


# クラス名の定義
CLASS_NAMES = {
    0: "未熟 (Unripe)",
    1: "やや未熟 (Slightly Unripe)",
    2: "適熟 (Ripe)",
    3: "やや過熟 (Slightly Overripe)",
    4: "過熟 (Overripe)"
}


def test_load_model():
    """チェックポイントからのモデル読み込みをテスト"""
    print("=" * 60)
    print("推論機能の動作確認")
    print("=" * 60)

    print("\n1. チェックポイントからのモデル読み込み")
    print("-" * 60)

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        print(f"エラー: チェックポイントが見つかりません: {checkpoint_path}")
        print("先に訓練を実行してください: python scripts/run_train.py")
        return None

    device = get_device()
    print(f"\n使用デバイス: {device}")
    print(f"チェックポイント: {checkpoint_path}")

    print("\nモデルを読み込み中...")
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=NUM_CLASSES,
        device=device,
        dropout_rate=DROPOUT_RATE
    )
    print("✓ モデルの読み込み完了")
    print(f"  モデルサイズ: {model.get_model_size_mb():.2f} MB")
    print(f"  評価モード: {not model.training}")

    return model


def test_preprocess_image():
    """画像の前処理をテスト"""
    print("\n2. 画像の前処理")
    print("-" * 60)

    # テスト用の画像を探す
    test_dir = TEST_DIR
    if not test_dir.exists():
        print(f"エラー: テストディレクトリが見つかりません: {test_dir}")
        return None

    # 最初の画像を取得
    image_files = sorted(list(test_dir.glob("**/*.jpg")))
    if not image_files:
        print(f"エラー: {test_dir} に画像ファイルが見つかりません")
        return None

    test_image = image_files[0]
    print(f"\nテスト画像: {test_image}")

    print("\n画像を前処理中...")
    tensor = preprocess_image(test_image)
    print("✓ 前処理完了")
    print(f"  テンソルの形状: {tensor.shape}")
    print("  期待値: [1, 3, 224, 224]")

    return test_image


def test_single_image_prediction(model, image_path):
    """単一画像の推論をテスト"""
    print("\n3. 単一画像の推論")
    print("-" * 60)

    device = get_device()
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    print(f"\n画像: {image_path}")
    print("推論を実行中...")

    result = predict_single_image(
        model=model,
        image_path=image_path,
        device=device,
        class_names=class_names
    )

    print("\n推論結果:")
    print(f"  予測クラス: {result['predicted_class']} - {result['predicted_class_name']}")
    print(f"  確率: {result['probabilities'][result['predicted_class']]:.2%}")

    print("\n全クラスの確率:")
    for i, (class_idx, prob, class_name) in enumerate(result['top_k']):
        marker = "✓" if i == 0 else " "
        print(f"  {marker} {class_idx}: {class_name:30s} {prob:.2%}")

    return result


def test_batch_prediction(model):
    """複数画像のバッチ推論をテスト"""
    print("\n4. 複数画像のバッチ推論")
    print("-" * 60)

    # テスト用の画像を探す
    test_dir = TEST_DIR
    if not test_dir.exists():
        print(f"エラー: テストディレクトリが見つかりません: {test_dir}")
        return

    image_files = sorted(list(test_dir.glob("**/*.jpg")))[:5]  # 最初の5枚のみ
    if not image_files:
        print(f"エラー: {test_dir} に画像ファイルが見つかりません")
        return

    print(f"\n推論する画像数: {len(image_files)}")

    from src.avocado_ripeness.predict import predict_batch  # noqa: E402

    device = get_device()
    class_names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]

    print("バッチ推論を実行中...")
    results = predict_batch(
        model=model,
        image_paths=image_files,
        device=device,
        class_names=class_names
    )

    print("\n推論結果:")
    for image_file, result in zip(image_files, results):
        print(f"\n  {image_file.name}:")
        print(f"    予測: {result['predicted_class']} - {result['predicted_class_name']}")
        print(f"    確率: {result['probabilities'][result['predicted_class']]:.2%}")


if __name__ == "__main__":
    # 1. モデルの読み込み
    model = test_load_model()
    if model is None:
        print("\nモデルの読み込みに失敗しました。訓練を先に実行してください。")
        sys.exit(1)

    # 2. 画像の前処理
    test_image = test_preprocess_image()
    if test_image is None:
        print("\n画像の前処理テストをスキップします。")
    else:
        # 3. 単一画像の推論
        test_single_image_prediction(model, test_image)

    # 4. バッチ推論
    test_batch_prediction(model)
