# avocado-ripeness
アボカドの成熟度を判定するモデル

## データセット
本プロジェクトでは、アボカドの成熟度画像データセットを使用しています。

提供元：Mendeley Data 'Hass' Avocado Ripening Photographic Dataset

ダウンロード：
https://data.mendeley.com/datasets/3xd9n945v8/1

## 使い方

### 1. 訓練

```bash
python scripts/run_train.py
```

### 2. 推論

単一画像の推論:
```bash
python scripts/run_predict.py <画像ファイルのパス>
```

ディレクトリ内の全画像を推論:
```bash
python scripts/run_predict.py <画像ディレクトリのパス>
```

### 3. テストセット評価

```bash
python scripts/evaluate_test.py
```

### 4. モバイル用にエクスポート（Executorch）

訓練済みモデルをExecutorch形式に変換してモバイル推論用にエクスポートします。

#### 必要な依存関係のインストール

```bash
pip install executorch
```

または、オプショナル依存関係としてインストール:

```bash
pip install -e ".[mobile]"
```

#### 変換の実行

```bash
python scripts/export_to_executorch.py
```

オプション指定:
```bash
python scripts/export_to_executorch.py \
    --checkpoint checkpoints/best_model.pth \
    --output models/avocado_ripeness.pte \
    --num-classes 5 \
    --dropout-rate 0.3 \
    --image-size 224
```

#### モバイルアプリでの使用方法

1. 変換された `.pte` ファイルをモバイルアプリのアセットに配置
2. Executorchランタイムを使用してモデルを読み込み
3. 入力画像を前処理:
   - リサイズ: 224x224
   - 正規化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - テンソル形状: [1, 3, 224, 224]
4. 推論を実行
5. 出力は [1, 5] の形状で各クラスのlogits

#### クラス定義

- 0: 未熟 (Unripe)
- 1: やや未熟 (Slightly Unripe)
- 2: 適熟 (Ripe)
- 3: やや過熟 (Slightly Overripe)
- 4: 過熟 (Overripe)
