# Avocado Ripeness（アボカド熟度判定）

アボカドの熟度をカメラ画像から推定するモバイルアプリ向けの画像分類プロジェクトです。**CNN（転移学習）でモデルを学習し、オンデバイス推論用に変換してアプリへ組み込み、App Store で公開**しています。本リポジトリは学習・評価・エクスポート用の Python コードをまとめたものです（モバイルアプリ本体は別リポジトリで実装）。

---

## 概要

スーパーでの選び方や、食べ頃のタイミングは人によって感覚がばらつきやすい題材です。本プロジェクトでは、**撮影画像から熟度カテゴリを推定するモデル**を PyTorch で構築し、**学習から評価、モバイル向け形式への変換まで一貫して**行いました。実装したモデルをアプリに組み込み、**実際にエンドユーザーが利用できる形でストア公開**まで行っています。

---

## デモ（App Store）

- **App Store**: [アボカド熟度チェッカー（日本の App Store）](https://apps.apple.com/jp/app/%E3%82%A2%E3%83%9C%E3%82%AB%E3%83%89%E7%86%9F%E5%BA%A6%E3%83%81%E3%82%A7%E3%83%83%E3%82%AB%E3%83%BC/id6759849445)

---

## 技術スタック

| 領域 | 技術 |
|------|------|
| **機械学習** | Python 3.10+, [PyTorch](https://pytorch.org/), torchvision, [timm](https://github.com/huggingface/pytorch-image-models)（EfficientNet 系の転移学習） |
| **評価・分析** | scikit-learn（混同行列・分類レポートなど） |
| **オンデバイス推論** | [ExecuTorch](https://pytorch.org/executorch/)（`.pte` へのエクスポート。`pip install -e ".[mobile]"` でオプション依存を導入） |
| **モバイルアプリ** | Flutter / Dart（**本リポジトリ外**。アプリ側でカメラ画像の前処理と推論結果の表示を実装） |
| **開発** | pytest, ruff, black など（`dev` 依存） |

---

## AI のポイント（このリポジトリでやっていること）

- **転移学習**: ImageNet 事前学習済みバックボーン（例: EfficientNet-B0 / EfficientNet-Lite）を用いた画像分類。設定は [`src/avocado_ripeness/config.py`](src/avocado_ripeness/config.py) の `MODEL_NAME` で切り替え可能です。
- **学習〜評価のパイプライン**: 訓練・バリデーション・テストの各分割で学習し、テストセットでは損失・Accuracy に加え、**混同行列・クラス別 Precision / Recall** で振る舞いを確認できます（`scripts/evaluate_test.py`）。
- **データとラベル設計**: 元データは 5 段階のフォルダ分類です。現在のデフォルトでは **3 クラス（未熟 / 適熟 / 過熟）** とし、`CLASS_MODE` により **フォルダ 1・3・5 のみを採用する「select」** など、ラベル統合方針を選べます（境界クラスを除外して学習しやすくする、等）。
- **クラス不均衡への対応**: 訓練データの分布に基づく **クラス重み付き CrossEntropyLoss**（`USE_CLASS_WEIGHTS`）を利用可能です。
- **モバイル向けエクスポート**: 学習済みチェックポイントから **ExecuTorch（`.pte`）** へ変換するスクリプト（`scripts/export_to_executorch.py`）を用意。推論時の前処理は **224×224 リサイズ + ImageNet 平均・標準偏差での正規化** と学習時のバリデーションと揃える想定です。
- **実運用で意識したこと**: 撮影環境（明るさ・照明の色）によって見え方が変わり、モデル出力に影響しやすい題材です。データ拡張やアプリ側の前処理の見直しは継続的な改善ポイントです。

---

## 今後の課題

- ラベル定義と人間の感覚のギャップの整理
- モバイル向けの **軽量モデルと精度のトレードオフ** の検証
- ストア公開後のフィードバックを反映した反復改善

---

## 開発者向け

### データセット

本プロジェクトでは、Mendeley Data の **'Hass' Avocado Ripening Photographic Dataset** を利用しています。

- 提供元・ダウンロード: [Mendeley Data](https://data.mendeley.com/datasets/3xd9n945v8/1)

データは `data/processed/avocado_ripeness/` 配下に `train` / `valid` / `test` として配置する想定です（前処理済み分割は別途用意）。

### 環境

```bash
pip install -e .
# モバイル向けエクスポート（ExecuTorch）を使う場合
pip install -e ".[mobile]"
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

**クラス数と出力次元**: 現在の設定では `NUM_CLASSES = 3`（3 段階）です。エクスポート時は **チェックポイントと同じクラス数** になるよう `--num-classes` を指定してください（デフォルトは config の `NUM_CLASSES`）。

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
