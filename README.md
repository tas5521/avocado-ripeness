# アボカド熟度判定

**カメラ画像からアボカドの熟度を推定するモバイルアプリ**です。CNN で学習したモデルを端末上で動かし、**App Store で公開**しています。  
本リポジトリは **AI モデルの学習・評価・モバイル向けエクスポート**をまとめた Python プロジェクトです（アプリの UI は別リポジトリ）。

---

## 概要

買い物や食事のタイミングで「食べ頃かどうか」を判断しづらい課題に対し、**撮影画像から熟度を推定する AI**を PyTorch で構築しました。**データの用意から学習・評価、オンデバイス用への変換までを一貫して実装**し、そのモデルをアプリに組み込んで**実ユーザーが使える形でストア公開**まで行っています。

---

## 特徴・強み

- **App Store で公開済み** — 実際にインストールして試せるプロダクトとして提供
- **画像分類モデル（CNN）を開発** — 転移学習による学習パイプラインを構築
- **モデル開発からデプロイ準備まで一貫** — 学習・テスト評価・ExecuTorch（`.pte`）への変換までを本リポジトリで管理
- **オンデバイス推論** — モバイル向け形式への変換を想定したエクスポート手順を用意
- **評価を可視化** — テストセットでの Accuracy に加え、混同行列・クラス別指標で振る舞いを確認可能

---

## デモ（App Store）

- [アボカド熟度チェッカー](https://apps.apple.com/jp/app/%E3%82%A2%E3%83%9C%E3%82%AB%E3%83%89%E7%86%9F%E5%BA%A6%E3%83%81%E3%82%A7%E3%83%83%E3%82%AB%E3%83%BC/id6759849445)

---

## 技術スタック

| 領域 | 技術 |
|------|------|
| **機械学習** | Python 3.10+, [PyTorch](https://pytorch.org/), torchvision, [timm](https://github.com/huggingface/pytorch-image-models)（EfficientNet 系の転移学習） |
| **評価** | scikit-learn（混同行列・分類レポートなど） |
| **オンデバイス推論** | [ExecuTorch](https://pytorch.org/executorch/)（`.pte` へのエクスポート。`pip install -e ".[mobile]"`） |
| **モバイルアプリ** | Flutter / Dart（**本リポジトリ外**。カメラ・表示はアプリ側で実装） |

開発用（テスト・Lint・整形）は `pip install -e ".[dev]"` で導入可能です（pytest など）。

---

## AI・機械学習のポイント（詳細）

- **転移学習**: ImageNet 事前学習済みバックボーン（例: EfficientNet-B0 / EfficientNet-Lite）による画像分類。[`src/avocado_ripeness/config.py`](src/avocado_ripeness/config.py) の `MODEL_NAME` で切り替え可能です。
- **学習〜評価のパイプライン**: 訓練・バリデーション・テストで学習し、テストでは損失・Accuracy に加え **混同行列・クラス別 Precision / Recall** を出力（`scripts/evaluate_test.py`）。
- **データとラベル設計**: 元データは 5 段階のフォルダ分類。デフォルトでは **3 クラス（未熟 / 適熟 / 過熟）** とし、`CLASS_MODE` で **フォルダ 1・3・5 のみを使う「select」** など、統合方針を選べます。
- **クラス不均衡への対応**: 訓練分布に基づく **クラス重み付き CrossEntropyLoss**（`USE_CLASS_WEIGHTS`）を利用可能です。
- **モバイル向けエクスポート**: 学習済みチェックポイントから **ExecuTorch（`.pte`）** へ変換（`scripts/export_to_executorch.py`）。推論時の前処理は **224×224 + ImageNet 平均・標準偏差の正規化** を学習時のバリデーションと揃える想定です。
- **実運用の観点**: 撮影環境（明るさ・照明）により見え方が変わりやすく、モデル出力に影響し得ます。データ拡張やアプリ側の前処理は継続的な改善ポイントです。

---

## 今後の課題

- ラベル定義と人間の感覚のギャップの整理
- **軽量モデルと精度のトレードオフ**の検証（モバイル向け）
- ストア公開後のフィードバックに基づく反復改善

---

## 開発者向け

### データセット

Mendeley Data の **'Hass' Avocado Ripening Photographic Dataset** を利用しています。

- [提供元・ダウンロード（Mendeley Data）](https://data.mendeley.com/datasets/3xd9n945v8/1)

データは `data/processed/avocado_ripeness/` に `train` / `valid` / `test` として配置する想定です（前処理済み分割は別途用意）。

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

**クラス数と出力次元**: 現在の設定では `NUM_CLASSES = 3`（3 段階）です。エクスポート時は **チェックポイントと同じクラス数** になるよう `--num-classes` を指定してください。

```bash
python scripts/export_to_executorch.py \
  --checkpoint checkpoints/best_model.pth \
  --output models/avocado_ripeness.pte \
  --num-classes 3 \
  --dropout-rate 0.3 \
  --image-size 224
```

**モバイル推論時の入力**: テンソル形状 `[1, 3, 224, 224]`。正規化は mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`（ImageNet 統計）。出力は **バッチ × クラス数** の logits（例: 3 クラスなら `[1, 3]`）。

**3 クラス時のラベル対応（例）**

| インデックス | 意味（例） |
|--------------|------------|
| 0 | 未熟 |
| 1 | 適熟 |
| 2 | 過熟 |

5 段階で学習する場合は `NUM_CLASSES = 5` およびデータセット側のラベル設計に合わせて読み替えてください。

---

## ライセンス

本プロジェクトのコードは [Apache-2.0](LICENSE) とします（`pyproject.toml` の記載に準拠）。データセットの利用条件は提供元のライセンスに従ってください。
