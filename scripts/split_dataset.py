from pathlib import Path
import shutil
import random

import pandas as pd
from sklearn.model_selection import train_test_split


# ===== 設定 =====
EXCEL_PATH = "data/raw/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset.xlsx"
IMAGE_DIR = Path(
    "data/raw/Hass Avocado Ripening Photographic Dataset/Avocado Ripening Dataset")
OUTPUT_DIR = Path("data/processed/avocado_ripeness")

SEED = 100
VALID_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)


def validate_columns(df):
    """必須列の存在確認"""
    required_cols = ["File Name", "Ripening Index Classification", "Sample"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")


def create_split_map(samples):
    """サンプルをtrain/valid/testに分割してマップを作成"""
    train_samples, temp_samples = train_test_split(
        samples, test_size=(VALID_RATIO + TEST_RATIO), random_state=SEED
    )

    valid_samples, test_samples = train_test_split(
        temp_samples,
        test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        random_state=SEED,
    )

    split_map = {}
    for s in train_samples:
        split_map[s] = "train"
    for s in valid_samples:
        split_map[s] = "valid"
    for s in test_samples:
        split_map[s] = "test"

    return split_map


def create_output_directories():
    """出力ディレクトリを作成"""
    for split in ["train", "valid", "test"]:
        for cls in ["1", "2", "3", "4", "5"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def copy_files(df, split_map):
    """ファイルをコピーし、欠損ファイルのリストを返す"""
    missing_files = []

    for _, row in df.iterrows():
        file_name = row["File Name"] + ".jpg"
        label = str(int(row["Ripening Index Classification"]))
        sample = row["Sample"]

        split = split_map[sample]

        src = IMAGE_DIR / file_name
        dst = OUTPUT_DIR / split / label / file_name

        if not src.exists():
            missing_files.append(file_name)
            continue

        shutil.copy2(src, dst)

    return missing_files


def save_missing_files_log(missing_files):
    """欠損ファイルのログを保存"""
    if missing_files:
        log_path = OUTPUT_DIR / "missing_files.txt"
        with open(log_path, "w") as f:
            for m in missing_files:
                f.write(m + "\n")

        print(f"Missing files: {len(missing_files)}")
        print(f"Saved: {log_path}")


def main():
    df = pd.read_excel(EXCEL_PATH)
    validate_columns(df)

    samples = df["Sample"].unique()
    split_map = create_split_map(samples)

    create_output_directories()

    missing_files = copy_files(df, split_map)
    save_missing_files_log(missing_files)


if __name__ == "__main__":
    main()
