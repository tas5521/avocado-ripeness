# アボカド熟度判定

アボカドの熟度を推定するAIモデル作成用のリポジトリです。\
学習・評価・モバイル向けエクスポートを行います。\
[アプリのUIのリポジトリはこちら](https://github.com/tas5521/avocado-ripeness_app)

## 概要

＜課題＞\
買い物や食事の際、アボカドが「食べ頃かどうか」を判断しづらい。

＜解決方法＞\
スマホのカメラでアボカドを撮影し、熟度を推定するAIアプリを開発しました。


## 特徴

- CNNによる画像分類: 転移学習による学習パイプラインを構築
- 評価を可視化: Accuracy・混同行列で評価
- オンデバイス推論: モバイル実行形式への変換


## App Store リンク

[アボカド熟度チェッカー](https://apps.apple.com/jp/app/%E3%82%A2%E3%83%9C%E3%82%AB%E3%83%89%E7%86%9F%E5%BA%A6%E3%83%81%E3%82%A7%E3%83%83%E3%82%AB%E3%83%BC/id6759849445)


## 技術スタック

| 領域 | 技術 |
|------|------|
| 機械学習 | Python3, PyTorch, torchvision, timm |
| 評価 | scikit-learn（混同行列・分類レポートなど） |
| オンデバイス推論 | ExecuTorch |


## AI・機械学習の詳細

- 転移学習: ImageNet事前学習済みモデル（EfficientNet）をbackboneとして使用。
- 学習〜評価のパイプライン: 訓練・バリデーション・テストで学習し、テストでは損失・Accuracy、混同行列・クラス別Precision/Recallを出力（`scripts/evaluate_test.py`）。
- データとラベル設計: 元のデータセットは5段階（未熟/やや未熟/適熟/やや過熟/過熟）のフォルダ分類。デフォルトでは3クラス（未熟/適熟/過熟）とし、`CLASS_MODE`でフォルダ1・3・5のみを使う「select」などを選択可能。
- クラス不均衡への対応: 訓練データの分布に基づくクラス重み付きCrossEntropyLossを利用可能（`USE_CLASS_WEIGHTS`）。
- モバイル実行形式エクスポート: 学習済みチェックポイントからExecuTorch（`.pte`）へ変換（`scripts/export_to_executorch.py`）。
- 実運用の観点: 撮影環境（明るさ・照明）により見え方が変わりやすく、モデル出力に影響する可能性があります。データ拡張やアプリ側の前処理は継続的な改善が必要なポイントです。


## 今後の課題

- モバイル向け軽量モデルの利用と、軽量化と精度のトレードオフの検証
- ストアのフィードバックに基づく改善


## 開発者向け

### データセット

Mendeley Dataの'Hass' Avocado Ripening Photographic Datasetを利用しています。

- [提供元・ダウンロード（Mendeley Data）](https://data.mendeley.com/datasets/3xd9n945v8/1)

元データをdata/rawに配置し、`split_dataset.py`を実行すると、`data/processed/avocado_ripeness/` に `train` / `valid` / `test` として配置されます。

### 環境

```bash
pip install -e .
pip install -e ".[mobile]"   # ExecuTorch エクスポート用（任意）
pip install -e ".[dev]"     # テスト・Lint 用（任意）
```

### 主なコマンド

訓練（[`src/avocado_ripeness/config.py`](src/avocado_ripeness/config.py) の設定を使用）:

```bash
python scripts/run_train.py
```

単一画像・ディレクトリの推論:

```bash
python scripts/run_predict.py <画像ファイルまたはディレクトリのパス>
```

テストセット評価:

```bash
python scripts/evaluate_test.py
```

ExecuTorch（`.pte`）への変換:

```bash
python scripts/export_to_executorch.py
```

クラス数と出力次元:\
現在の設定では `NUM_CLASSES = 3`（3 段階）です。\
エクスポートする時はチェックポイントと同じクラス数になるよう`--num-classes`を指定します。

```bash
python scripts/export_to_executorch.py \
  --checkpoint checkpoints/best_model.pth \
  --output models/avocado_ripeness.pte \
  --num-classes 3 \
  --dropout-rate 0.3 \
  --image-size 224
```

モバイル推論時の入力:\
入力テンソル形状: `[1, 3, 224, 224]（Batch, Channel, Height, Width）`\
正規化は、mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`（ImageNet 統計）。\
出力はバッチ×クラス数のlogits。

3 クラス時のラベル対応（例）

| インデックス | 意味（例） |
|--------------|------------|
| 0 | 未熟 |
| 1 | 適熟 |
| 2 | 過熟 |

5 段階で学習する場合は `NUM_CLASSES = 5` およびデータセット側のラベル設計に合わせて読み替えてください。


## ライセンス

本プロジェクトのコードは [Apache-2.0](LICENSE) とします（`pyproject.toml` の記載に準拠）。\
データセットの利用条件は提供元のライセンスに従ってください。
