from pathlib import Path
import csv
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class L1WithNaNs(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.L1Loss(reduction='none')

    def forward(self, preds, targets):
        mask = torch.isfinite(targets)
        valid_count = mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, dtype=preds.dtype, device=preds.device)
        targets_clean = torch.where(mask, targets, torch.zeros_like(targets))
        l1_raw = self.L1(preds, targets_clean)
        l1_masked = l1_raw * mask.float()
        L1_loss = l1_masked.sum() / mask.sum().clamp(min=1)
        return L1_loss


def r2_score_with_nans(
    preds: torch.Tensor, targets: torch.Tensor, per_target: bool = False
) -> float | list[float]:

    if per_target:
        r2_scores = []
        for i in range(targets.shape[1]):
            target_col = targets[:, i]
            pred_col = preds[:, i]
            mask = torch.isfinite(target_col)
            target_clean = target_col[mask]
            pred_clean = pred_col[mask]

            if target_clean.numel() == 0:
                r2_scores.append(float("nan"))
                continue

            mean_target = target_clean.mean()
            ss_tot = ((target_clean - mean_target) ** 2).sum()
            ss_res = ((target_clean - pred_clean) ** 2).sum()
            r2 = 1 - ss_res / ss_tot.clamp(min=1e-8)
            r2_scores.append(r2.detach().cpu().item())
        return r2_scores

    else:
        mask = torch.isfinite(targets)
        preds_clean = preds[mask]
        targets_clean = targets[mask]

        if targets_clean.numel() == 0:
            return float("nan")

        mean_target = targets_clean.mean()
        ss_tot = ((targets_clean - mean_target) ** 2).sum()
        ss_res = ((targets_clean - preds_clean) ** 2).sum()
        r2 = 1 - ss_res / ss_tot.clamp(min=1e-8)
        return r2.detach().cpu().item()


def pcc_score_with_nans(
    preds: torch.Tensor, targets: torch.Tensor, per_target: bool = False
) -> float | list[float]:
    # PCC: cov(x,y) / (std(x) * std(y))
    if per_target:
        pcc_scores = []
        for i in range(targets.shape[1]):
            target_col = targets[:, i]
            pred_col = preds[:, i]
            mask = torch.isfinite(target_col)
            target_clean = target_col[mask]
            pred_clean = pred_col[mask]

            if target_clean.numel() < 2:
                pcc_scores.append(float("nan"))
                continue

            vx = pred_clean - torch.mean(pred_clean)
            vy = target_clean - torch.mean(target_clean)
            pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)).clamp(min=1e-8))
            pcc_scores.append(pcc.detach().cpu().item())
        return pcc_scores

    else:
        mask = torch.isfinite(targets)
        preds_clean = preds[mask]
        targets_clean = targets[mask]

        if targets_clean.numel() < 2:
            return float("nan")

        vx = preds_clean - torch.mean(preds_clean)
        vy = targets_clean - torch.mean(targets_clean)
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) * torch.sum(vy ** 2)).clamp(min=1e-8))
        return pcc.detach().cpu().item()


def train_mpra_predictor(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    target_columns: list[str],
    batch_size: int,
    learning_rate: float,
    n_epochs: int,
    early_stopping_patience: int,
    weights_path: Path,
    logging_path: Path,
    verbose: bool = True
) -> None:

    loss_fn = L1WithNaNs()
    metric_fn = r2_score_with_nans

    optimizer = torch.optim.AdamW([{"params": model.non_base_parameters(), "lr": learning_rate},], weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        div_factor=2,
        final_div_factor=100,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )

    if not logging_path.exists():
        with open(logging_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["epoch", "train_loss", "val_loss", "train_r2_overall", "val_r2_overall"] +
                [f"train_r2_{i}" for i in range(len(target_columns))] +
                [f"val_r2_{i}" for i in range(len(target_columns))]
            )

    best_valid_loss = float('inf')
    patience_counter = 0
    for epoch in range(n_epochs):
        model.train()
        for name, module in model.named_children():  # NOTE: we freeze and eval base models (Sei and Malinois)
            if name.endswith('_base'):
                module.eval()
        train_losses = []
        train_metrics = []
        train_preds = []
        train_targets = []

        train_iter = tqdm(train_loader, desc='train') if verbose else train_loader
        for *x_batch, y_batch in train_iter:
            optimizer.zero_grad()
            preds = model(*x_batch)
            loss = loss_fn(preds, y_batch)
            metric = metric_fn(preds, y_batch, per_target=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            train_metrics.append(np.array(metric))
            train_preds.append(preds.detach().cpu())
            train_targets.append(y_batch.detach().cpu())

        model.eval()
        valid_losses = []
        valid_metrics = []
        valid_preds = []
        valid_targets = []

        valid_iter = tqdm(valid_loader, desc='validation') if verbose else valid_loader
        with torch.no_grad():
            for *x_batch, y_batch in valid_iter:
                preds = model(*x_batch)
                loss = loss_fn(preds, y_batch)
                metric = metric_fn(preds, y_batch, per_target=True)
                valid_losses.append(loss.item())
                valid_metrics.append(np.array(metric))
                valid_preds.append(preds.detach().cpu())
                valid_targets.append(y_batch.detach().cpu())

        avg_train_loss = np.mean(train_losses)
        avg_valid_loss = np.mean(valid_losses)
        avg_train_metric = np.nanmean(train_metrics, axis=0)
        avg_valid_metric = np.nanmean(valid_metrics, axis=0)

        all_train_preds = torch.cat(train_preds, dim=0)
        all_train_targets = torch.cat(train_targets, dim=0)
        all_valid_preds = torch.cat(valid_preds, dim=0)
        all_valid_targets = torch.cat(valid_targets, dim=0)
        overall_train_r2 = metric_fn(all_train_preds, all_train_targets, per_target=False)
        overall_val_r2 = metric_fn(all_valid_preds, all_valid_targets, per_target=False)

        if verbose:
            report_str = (
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_valid_loss:.4f}, "
                f"Train R^2 = {overall_train_r2:.4f}, Val R^2 = {overall_val_r2:.4f}"
            )
            print(report_str)

        with open(logging_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [epoch + 1, avg_train_loss, avg_valid_loss, overall_train_r2, overall_val_r2] +
                avg_train_metric.tolist() +
                avg_valid_metric.tolist()
            )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), weights_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
