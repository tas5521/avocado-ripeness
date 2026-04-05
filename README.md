# アボカド熟度判定

カメラ画像からアボカドの熟度を推定するための**画像分類モデル（CNN）**を学習・評価し、モバイル向けにエクスポートするPythonリポジトリです。

- **モバイルアプリ（UI）**: [avocado-ripeness_app](https://github.com/tas5521/avocado-ripeness_app)
- **App Store**: [アボカド熟度チェッカー](https://apps.apple.com/jp/app/%E3%82%A2%E3%83%9C%E3%82%AB%E3%83%89%E7%86%9F%E5%BA%A6%E3%83%81%E3%82%A7%E3%83%83%E3%82%AB%E3%83%BC/id6759849445)


## 1. 概要（問題定義・背景）

- **何を解くか**: 買い物や食事のタイミングで、アボカドが「食べ頃に近いか」を画像から推定し、判断を助けること。
- **なぜ重要か**: 食べ頃を見誤ると、食品ロス、無駄な購入、食感の不満などにつながりやすいため。
- **ゴール**:
  - **モデル面**: テストセットで損失・Accuracy、混同行列・クラス別指標で振る舞いを確認できること。
  - **プロダクト面**: 学習済みモデルをモバイル端末に載せ替え可能な形にし、**オンデバイス推論**を前提にしたアプリで利用できること。


## 2. アプローチ（全体戦略）

- **方針**: 
  - ImageNet事前学習済みバックボーンとして用いた**転移学習**により画像分類モデルを構築する。
  - **軽量〜中規模なモデル**を選択する。
  - モバイル向けには**ExecuTorch**で`.pte`にエクスポートする。
- **理由**:
  - 事前学習済みモデルをバックボーンにすることで、ゼロから大規模に学習する必要がなくなり、開発時間・コストを削減できる。
  - 軽量〜中規模のモデルを用いることで、モバイル端末でのリアルタイム推論が可能になる。
  - エッジ向けでより高速・軽量なExecuTorchを採用する（従来のTorchScriptはPyTorchで公式で非推奨となっている）。


**主要な技術**

| 言語 | Python3.10+ | 
| 機械学習 | PyTorch, torchvision（EfficientNet-B0）, timm（EfficientNet-Lite0） |
| 評価 | scikit-learn（混同行列・分類レポートなど） |
| オンデバイス推論 | ExecuTorch（`.pte` への変換） |


## 3. データ

- **出所**: 自前収集ではなく、公開データセット**Mendeley「Hass」Avocado Ripening Photographic Dataset**を利用しています。  
  - [提供元（Mendeley Data）](https://data.mendeley.com/datasets/3xd9n945v8/1)
- **分割**: 元データを`data/raw`に置き、[`scripts/split_dataset.py`](scripts/split_dataset.py)で`data/processed/avocado_ripeness/`以下にtrain / valid / testとして分割されます。
- **ラベル**: 元は5段階（フォルダ 1〜5, 未熟 / やや未熟 / 適熟 / やや過熟 / 過熟）。学習時は[`config.py`](src/avocado_ripeness/config.py)の`NUM_CLASSES`と`CLASS_MODE`により、**3クラス化**や**フォルダ1・3・5のみを使う`select`**など、統合方針を選べます。
- **課題**:
  - **撮影条件**（明るさや背景など）のばらつきは、見た目とラベルの対応関係を歪める。


## 4. 手法（実務フロー）

- **データ理解（EDA）**: `scripts/check_data_distribution.py`でtrain・valid・testのクラス別件数やクラス重みの目安を確認。
- **前処理・拡張**（訓練時、[`dataloader.py`](src/avocado_ripeness/dataloader.py)）:
  - リサイズ・ランダムクロップ、水平反転、小さな回転、明るさ・コントラスト等の変化、ImageNetの平均・標準偏差で正規化。
  - validとtestは拡張なしで、リサイズと正規化。
- **モデル選定**: **EfficientNet-B0**または**EfficientNet-Lite0**を`MODEL_NAME`で選択。
  - EfficientNetを選択した理由
    - 軽量でありながら精度が高い
    - モバイル推論に適している
    - 事前学習済みモデルが利用可能
- **訓練**（[`scripts/run_train.py`](scripts/run_train.py)、設定は [`config.py`](src/avocado_ripeness/config.py)）:
  - 損失関数: `CrossEntropyLoss`（`USE_CLASS_WEIGHTS`で、クラス重み付け選択可能）
  - 最適化: Adam、学習率スケジューラ（`ReduceLROnPlateau`）、Early stopping
  - `USE_OVERSAMPLING`で`WeightedRandomSampler`によるオーバーサンプリングが可能（任意, デフォルトでOFF）


## 5. 評価

- **指標**: テスト損失、**Accuracy**、**混同行列**、クラス別 **Precision / Recall / F1**（scikit-learn）。
- **実行方法**: 最新の数値は環境・チェックポイントに依存するため、次で再現してください。

```bash
python scripts/evaluate_test.py
```

- **摂動下での比較（任意）**: 通常テストと同じラベルで、**明るさ・コントラスト・回転・左右反転**などの摂動を画像に加えたうえで Accuracyを測り、`best_model.pth`（拡張なし学習）と `best_model_augumented.pth`（拡張あり学習）を並べて比較できます。実運用の撮影ばらつきのシミュレーション用（摂動の有無で「頑健性」の目安を見る）。

```bash
python scripts/evaluate_robustness.py
# 一部の preset のみ: python scripts/evaluate_robustness.py --presets clean brightness_0.75 rotate_10
```

  利用可能な preset 名は [`src/avocado_ripeness/robustness.py`](src/avocado_ripeness/robustness.py) の `list_perturbation_presets()` を参照してください。

- **結果（実行例）**: `checkpoints/best_model.pth`・テストデータ 1290 件・`python scripts/evaluate_test.py` 実行時のログより。チェックポイントや設定が変われば数値も変わります。

  - **テスト損失** 0.1090｜**Accuracy** 0.9527（95.27%）

  **混同行列**（行＝正解、列＝予測）

  | 正解＼予測 | 未熟 | 適熟 | 過熟 | 合計 |
  |------------|-----:|-----:|-----:|-----:|
  | 未熟 | 479 | 3 | 0 | 482 |
  | 適熟 | 4 | 395 | 17 | 416 |
  | 過熟 | 0 | 37 | 355 | 392 |
  | **合計** | 483 | 435 | 372 | 1290 |

  **クラス別** Precision / Recall / F1-score（Support）

  | クラス | Precision | Recall | F1-score | Support |
  |--------|----------:|-------:|---------:|--------:|
  | 未熟 | 0.9917 | 0.9938 | 0.9927 | 482 |
  | 適熟 | 0.9080 | 0.9495 | 0.9283 | 416 |
  | 過熟 | 0.9543 | 0.9056 | 0.9293 | 392 |
  | macro 平均 | 0.9514 | 0.9496 | 0.9501 | 1290 |
  | weighted 平均 | 0.9534 | 0.9527 | 0.9527 | 1290 |


## 6. 考察：実運用を踏まえたデータ拡張の採用

モバイル利用では、アボカドの**向き**や**照明（暗い部屋・光の当たり方）**が撮影ごとに変わります。テストセット上のAccuracyだけを追うと **データ拡張なし**の学習の方が高く出ることがありますが、**分布が少しずれた条件**では拡張あり学習の方が振る舞いが安定しやすい、という判断をしたい、というのが本節の動機です。

### 比較のしかた（同一条件での切り分け）

- **変えたこと**: 訓練時の **`USE_DATA_AUGMENTATION`のみ**（ほかは同一の条件で学習）。
- **評価**: [`scripts/evaluate_robustness.py`](scripts/evaluate_robustness.py)で、テスト画像に摂動（乱数なし）を加えたうえでAccuracyを測る。ここでは**摂動なし（`clean`）**と**明るさ0.75×（`brightness_0.75`、暗めの撮影の粗い近似）**を併記する。

```bash
python scripts/evaluate_robustness.py --presets clean brightness_0.75
```

### 結果の一例（同一テスト 1290 件・例示用）

チェックポイント例: `checkpoints/best_model.pth`（拡張なし学習）と `checkpoints/best_model_augumented.pth`（拡張あり学習）。環境・シード・再学習で数値は変わり得ます。

| preset | 条件の意味 | 拡張なし学習 Accuracy | 拡張あり学習 Accuracy |
|--------|------------|----------------------:|----------------------:|
| `clean` | 摂動なし（通常テストと同様の前処理） | 0.9814 | 0.9527 |
| `brightness_0.75` | 明るさ 0.75×（暗めのシミュレーション） | 0.9225 | 0.9519 |

**clean からの劣化幅**（各 Accuracy − `clean` の Accuracy）:

| preset | 拡張なし学習 | 拡張あり学習 |
|--------|-------------:|-------------:|
| `brightness_0.75` | −0.0589 | −0.0008 |

### 読み方と採用判断

- **通常テスト（clean）**だけ見ると、拡張なし学習の方が高く見える。
- **暗めの摂動**では、拡張なし学習は精度が大きく落ちる一方、拡張あり学習はほぼ維持される。この差は、実運用で想定しうる**照明のばらつき**に対する**頑健性の目安**として解釈できる（厳密な実機再現ではない）。
- 以上より、**テストの最高Accuracyより、撮影条件の変化に強い挙動を優先**し、**拡張ありで学習したモデルを採用する**方針とした。

**限界**: 摂動は合成であり、実機の「暗い部屋」とは一致しない。別の強さ・種類の摂動や実データでの確認が望ましい。


