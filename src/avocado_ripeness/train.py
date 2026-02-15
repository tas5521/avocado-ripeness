"""
訓練ロジックモジュール

1エポック分の訓練ループを実装
バリデーションループを実装
複数エポック訓練と設定ファイル
チェックポイント保存、学習率スケジューラー、Early stopping
"""

import torch
from pathlib import Path
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


def save_checkpoint(
    model,
    optimizer,
    epoch,
    history,
    checkpoint_dir,
    is_best=False
):
    """
    チェックポイントを保存する

    Args:
        model: 保存するモデル
        optimizer: 保存するオプティマイザー
        epoch: 現在のエポック番号
        history: 訓練履歴
        checkpoint_dir: チェックポイント保存ディレクトリ
        is_best: 最良モデルかどうか
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }

    # 通常のチェックポイント
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)

    # 最良モデルの場合、別途保存
    if is_best:
        best_model_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_model_path)
        print(f"  ✓ 最良モデルを保存しました: {best_model_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    チェックポイントを読み込む

    Args:
        model: モデル
        optimizer: オプティマイザー
        checkpoint_path: チェックポイントファイルのパス
        device: 使用するデバイス

    Returns:
        dict: チェックポイント情報（エポック番号、履歴など）
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"チェックポイントを読み込みました: {checkpoint_path}")
    print(f"  エポック: {checkpoint['epoch']}")

    return checkpoint


def _print_training_start(num_epochs, scheduler, checkpoint_dir, early_stopping_patience):
    """訓練開始メッセージを表示"""
    print(f"\n{'=' * 60}")
    print(f"訓練開始: {num_epochs}エポック")
    if scheduler:
        print("学習率スケジューラー: 有効")
    if checkpoint_dir:
        print(f"チェックポイント保存: {checkpoint_dir}")
    if early_stopping_patience:
        print(f"Early stopping: patience={early_stopping_patience}")
    print(f"{'=' * 60}\n")


def _print_epoch_start(epoch, num_epochs, scheduler, optimizer):
    """エポック開始メッセージを表示"""
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch}/{num_epochs}")
    if scheduler:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学習率: {current_lr:.6f}")
    print(f"{'=' * 60}")


def _update_best_model(valid_loss, best_valid_loss, epochs_without_improvement, epoch):
    """最良モデルを更新し、改善状況を返す"""
    is_best = False
    new_epochs_without_improvement = epochs_without_improvement

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        new_epochs_without_improvement = 0
        is_best = True
        print(f"  ✓ バリデーション損失が改善しました！ (最良: {best_valid_loss:.4f})")
    else:
        new_epochs_without_improvement += 1
        if epoch > 1:
            print(f"  ⚠ バリデーション損失が改善していません ({new_epochs_without_improvement}エポック連続)")

    return is_best, best_valid_loss, new_epochs_without_improvement


def _update_scheduler(scheduler, valid_loss, optimizer):
    """学習率スケジューラーを更新"""
    if not scheduler:
        return

    current_lr = optimizer.param_groups[0]['lr']

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(valid_loss)
    else:
        scheduler.step()

    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != current_lr:
        print(f"  ✓ 学習率が更新されました: {current_lr:.6f} → {new_lr:.6f}")


def _save_checkpoint_if_needed(checkpoint_dir, save_every_epoch, is_best, model, optimizer, epoch, history):
    """必要に応じてチェックポイントを保存"""
    if checkpoint_dir and (save_every_epoch or is_best):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            checkpoint_dir=checkpoint_dir,
            is_best=is_best
        )


def _check_early_stopping(early_stopping_patience, epochs_without_improvement, epoch):
    """Early stoppingをチェックし、停止が必要かどうかを返す"""
    if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
        print(f"\n{'=' * 60}")
        print(f"Early stopping: {early_stopping_patience}エポック改善がありませんでした")
        print(f"訓練を停止します（エポック {epoch}）")
        print(f"{'=' * 60}\n")
        return True
    return False


def _process_epoch(
    epoch,
    num_epochs,
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    device,
    scheduler,
    checkpoint_dir,
    save_every_epoch,
    history,
    best_valid_loss,
    epochs_without_improvement
):
    """1エポックの処理を実行し、更新された状態を返す"""
    _print_epoch_start(epoch, num_epochs, scheduler, optimizer)

    train_results = train_one_epoch(
        model=model,
        dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    valid_results = validate_one_epoch(
        model=model,
        dataloader=valid_dataloader,
        criterion=criterion,
        device=device
    )

    history['train_loss'].append(train_results['loss'])
    history['train_accuracy'].append(train_results['accuracy'])
    history['valid_loss'].append(valid_results['loss'])
    history['valid_accuracy'].append(valid_results['accuracy'])

    print(f"\nEpoch {epoch} 結果:")
    print(
        f"  訓練 - Loss: {train_results['loss']:.4f}, Accuracy: {train_results['accuracy']:.4f}")
    print(
        f"  バリデーション - Loss: {valid_results['loss']:.4f}, Accuracy: {valid_results['accuracy']:.4f}")

    is_best, best_valid_loss, epochs_without_improvement = _update_best_model(
        valid_results['loss'], best_valid_loss, epochs_without_improvement, epoch
    )

    _update_scheduler(scheduler, valid_results['loss'], optimizer)
    _save_checkpoint_if_needed(checkpoint_dir, save_every_epoch,
                               is_best, model, optimizer, epoch, history)

    return best_valid_loss, epochs_without_improvement


def train_multiple_epochs(
    model,
    train_dataloader,
    valid_dataloader,
    criterion,
    optimizer,
    device,
    num_epochs=10,
    scheduler=None,
    checkpoint_dir=None,
    save_best_model=True,
    save_every_epoch=False,
    early_stopping_patience=None
):
    """
    複数エポックの訓練を実行する（チェックポイント、スケジューラー、Early stopping対応）

    Args:
        model: 訓練するモデル
        train_dataloader: 訓練データのDataLoader
        valid_dataloader: バリデーションデータのDataLoader
        criterion: 損失関数
        optimizer: オプティマイザー
        device: 使用するデバイス
        num_epochs: 訓練エポック数（デフォルト: 10）
        scheduler: 学習率スケジューラー（Noneの場合は使用しない）
        checkpoint_dir: チェックポイント保存ディレクトリ（Noneの場合は保存しない）
        save_best_model: 最良モデルを保存するか（デフォルト: True）
        save_every_epoch: 毎エポック保存するか（デフォルト: False）
        early_stopping_patience: Early stoppingのpatience（Noneの場合は使用しない）

    Returns:
        dict: 訓練履歴（各エポックの訓練・バリデーション結果）
            {
                'train_loss': [loss1, loss2, ...],
                'train_accuracy': [acc1, acc2, ...],
                'valid_loss': [loss1, loss2, ...],
                'valid_accuracy': [acc1, acc2, ...]
            }
    """
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'valid_loss': [],
        'valid_accuracy': []
    }

    best_valid_loss = float('inf')
    epochs_without_improvement = 0

    _print_training_start(num_epochs, scheduler, checkpoint_dir,
                          early_stopping_patience)

    for epoch in range(1, num_epochs + 1):
        best_valid_loss, epochs_without_improvement = _process_epoch(
            epoch, num_epochs, model, train_dataloader, valid_dataloader,
            criterion, optimizer, device, scheduler, checkpoint_dir,
            save_every_epoch, history, best_valid_loss, epochs_without_improvement
        )

        if _check_early_stopping(early_stopping_patience, epochs_without_improvement, epoch):
            break

    print(f"\n{'=' * 60}")
    print("訓練完了！")
    print(f"最良バリデーション損失: {best_valid_loss:.4f}")
    print(f"{'=' * 60}\n")

    return history
