"""
訓練ロジックモジュール

1エポック分の訓練ループ。
"""

from tqdm import tqdm

from .utils import calculate_accuracy


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    1エポック分の訓練を実行する

    Args:
        model: 訓練するモデル
        dataloader: 訓練データのDataLoader
        criterion: 損失関数
        optimizer: オプティマイザー
        device: 使用するデバイス

    Returns:
        dict: 訓練結果（平均損失、平均Accuracy）
    """
    # モデルを訓練モードに設定
    model.train()

    # 累積用の変数
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0

    # プログレスバーを作成
    pbar = tqdm(dataloader, desc="Training")

    # 各バッチを処理
    for images, labels in pbar:
        # デバイスに移動
        images = images.to(device)
        labels = labels.to(device)

        # 1. 勾配をゼロにリセット
        optimizer.zero_grad()

        # 2. 順伝播（forward pass）
        outputs = model(images)

        # 3. 損失を計算
        loss = criterion(outputs, labels)

        # 4. 逆伝播（backward pass）
        loss.backward()

        # 5. パラメータを更新
        optimizer.step()

        # 統計を更新
        running_loss += loss.item()
        accuracy = calculate_accuracy(outputs, labels)
        running_accuracy += accuracy
        num_batches += 1

        # プログレスバーを更新
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })

    # 平均を計算
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }
