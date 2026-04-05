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
NUM_CLASSES = 3  # アボカドの成熟度クラス数（3: 未熟/適熟/過熟、5: 5段階）
# "efficientnet_b0", "efficientnet_lite0", "efficientnet_lite1"
MODEL_NAME = "efficientnet_lite0"
PRETRAINED = True

# 訓練設定
BATCH_SIZE = 32
NUM_EPOCHS = 50  # エポック数を増加（Early stoppingで早く終わる可能性あり）
LEARNING_RATE = 0.0005  # 学習率を下げて安定した収束を目指す
NUM_WORKERS = 2  # データ読み込みの並列処理数（2-4が推奨、MPSでは2が安全）

# 画像設定
IMAGE_SIZE = 224  # 入力画像サイズ（224x224）

# 過学習対策設定
USE_DATA_AUGMENTATION = False  # データ拡張を使用するか（デフォルト: True）
DROPOUT_RATE = 0.3  # ドロップアウト率（0.0-1.0、0.0で無効）

# クラス重み付け設定
USE_CLASS_WEIGHTS = True  # 訓練データの分布に基づくクラス重み付けを使用するか

# オーバーサンプリング設定
USE_OVERSAMPLING = False  # 少数クラスのオーバーサンプリングを使用するか

# 3段階分類の統合方法
CLASS_MODE = "select"  # "merge": 全データ使用(1,2→未熟, 3→適熟, 4,5→過熟), "select": 1,3,5のみ使用

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
EARLY_STOPPING_PATIENCE = 7  # 何エポック改善がなければ訓練を停止するか

# デバイス設定（自動選択されるが、明示的に指定することも可能）
# DEVICE = "mps"  # Apple Silicon GPU
# DEVICE = "cuda"  # NVIDIA GPU
# DEVICE = "cpu"  # CPU
