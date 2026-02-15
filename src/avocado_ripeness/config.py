"""
設定ファイル

訓練や推論に使用するハイパーパラメータや設定を定義する。
必要になったら設定を追加していく。
"""

from pathlib import Path

# データパス
DATA_DIR = Path("data/processed/avocado_ripeness")
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

# モデル設定
NUM_CLASSES = 5  # アボカドの成熟度クラス数
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True

# 訓練設定
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # データ読み込みの並列処理数（0 = メインスレッドのみ）

# 画像設定
IMAGE_SIZE = 224  # 入力画像サイズ（224x224）

# デバイス設定（自動選択されるが、明示的に指定することも可能）
# DEVICE = "mps"  # Apple Silicon GPU
# DEVICE = "cuda"  # NVIDIA GPU
# DEVICE = "cpu"  # CPU
