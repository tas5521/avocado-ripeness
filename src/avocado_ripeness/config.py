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

# チェックポイント設定
CHECKPOINT_DIR = Path("checkpoints")  # チェックポイント保存ディレクトリ
SAVE_BEST_MODEL = True  # 最良モデルを保存するか
SAVE_EVERY_EPOCH = False  # 毎エポック保存するか（Falseの場合は最良モデルのみ）

# 学習率スケジューラー設定
USE_SCHEDULER = True  # 学習率スケジューラーを使用するか
SCHEDULER_FACTOR = 0.5  # 学習率を減らす倍率
SCHEDULER_PATIENCE = 3  # 何エポック改善がなければ学習率を下げるか
SCHEDULER_MIN_LR = 1e-6  # 最小学習率

# Early stopping設定
USE_EARLY_STOPPING = True  # Early stoppingを使用するか
EARLY_STOPPING_PATIENCE = 5  # 何エポック改善がなければ訓練を停止するか

# デバイス設定（自動選択されるが、明示的に指定することも可能）
# DEVICE = "mps"  # Apple Silicon GPU
# DEVICE = "cuda"  # NVIDIA GPU
# DEVICE = "cpu"  # CPU
