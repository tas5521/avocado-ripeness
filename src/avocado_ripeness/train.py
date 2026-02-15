"""
訓練ロジックモジュール

1エポック分の訓練ループ
バリデーションループ
複数エポック訓練
"""

import torch
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


def validate_one_epoch(model, dataloader, criterion, device):
    """
    1エポック分のバリデーションを実行する

    Args:
        model: 評価するモデル
        dataloader: バリデーションデータのDataLoader
        criterion: 損失関数
        device: 使用するデバイス

    Returns:
        dict: バリデーション結果（平均損失、平均Accuracy）

    注意:
        バリデーションでは勾配計算とパラメータ更新は行わない。
        model.eval()とtorch.no_grad()を使用する。
    """
    # モデルを評価モードに設定
    model.eval()

    # 累積用の変数
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0

    # プログレスバーを作成
    pbar = tqdm(dataloader, desc="Validation")

    # 勾配計算を無効化（メモリ節約と速度向上）
    with torch.no_grad():
        # 各バッチを処理
        for images, labels in pbar:
            # デバイスに移動
            images = images.to(device)
            labels = labels.to(device)

            # 順伝播（forward pass）のみ
            outputs = model(images)

            # 損失を計算
            loss = criterion(outputs, labels)

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


def train_multiple_epochs(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=10
):
    """
    複数エポックの訓練を実行する

    Args:
        model: 訓練するモデル
        train_dataloader: 訓練データのDataLoader
        valid_dataloader: バリデーションデータのDataLoader
        criterion: 損失関数
        optimizer: オプティマイザー
        device: 使用するデバイス
        num_epochs: 訓練エポック数（デフォルト: 10）

    Returns:
        dict: 訓練履歴（各エポックの訓練・バリデーション結果）
            {
                'train_loss': [loss1, loss2, ...],
                'train_accuracy': [acc1, acc2, ...],
                'valid_loss': [loss1, loss2, ...],
                'valid_accuracy': [acc1, acc2, ...]
            }
    """
    # 訓練履歴を記録するリスト
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'valid_loss': [],
        'valid_accuracy': []
    }

    print(f"\n{'=' * 60}")
    print(f"訓練開始: {num_epochs}エポック")
    print(f"{'=' * 60}\n")

    # 各エポックを実行
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        # 訓練を実行
        train_results = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        # バリデーションを実行
        valid_results = validate_one_epoch(
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            device=device
        )

        # 結果を記録
        history['train_loss'].append(train_results['loss'])
        history['train_accuracy'].append(train_results['accuracy'])
        history['valid_loss'].append(valid_results['loss'])
        history['valid_accuracy'].append(valid_results['accuracy'])

        # エポック結果を表示
        print(f"\nEpoch {epoch} 結果:")
        print(
            f"  訓練 - Loss: {train_results['loss']:.4f}, Accuracy: {train_results['accuracy']:.4f}")
        print(
            f"  バリデーション - Loss: {valid_results['loss']:.4f}, Accuracy: {valid_results['accuracy']:.4f}")

        # 改善の確認
        if epoch > 1:
            prev_valid_loss = history['valid_loss'][-2]
            if valid_results['loss'] < prev_valid_loss:
                print(
                    f"  ✓ バリデーション損失が改善しました！ ({prev_valid_loss:.4f} → {valid_results['loss']:.4f})")
            else:
                print(
                    f"  ⚠ バリデーション損失が悪化しました ({prev_valid_loss:.4f} → {valid_results['loss']:.4f})")

    print(f"\n{'=' * 60}")
    print("訓練完了！")
    print(f"{'=' * 60}\n")

    return history
