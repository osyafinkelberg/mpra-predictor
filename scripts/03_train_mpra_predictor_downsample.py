from pathlib import Path
import sys
import csv
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # noqa F401

sys.path.insert(0, '..')
import mpra_predictor  # noqa E402
from mpra_predictor import malinois_base as malinois_module


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOGGING_PATH = DATA_DIR / "trained_models/logs/dataset_downsample.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
CONFIG = {
    'batch_size': 256,
    'learning_rate': 1e-3,
    'sei_pooler': {'n_heads': 1, 'hidden_dim': 128, 'pos_emb_dim': 8, 'dropout': 0.5, },
    'mal_pooler': {'n_heads': 1, 'hidden_dim': 64, 'pos_emb_dim': 8, 'dropout': 0.5, },
    'fusion': {'output_dim': 128, 'dropout': 0.7},
    'mlp': {'hidden_size': 128, 'num_res_blocks': 0, 'dropout': 0.5},
    'target_columns': [
        'lfc_mean_GM12878', 'lfc_mean_Jurkat', 'lfc_mean_MRC5',
        'lfc_mean_A549', 'lfc_mean_HEK293', 'lfc_mean_K562'
    ]
}
TARGETS = CONFIG['target_columns']
CREST_JURKAT_INDEX = 1
CREST_K562_INDEX = 5
MALINOIS_K562_INDEX = 0
N_REPLICAS = 3


def train_and_eval_loop(
    seed: int,
    dataset_train: pl.DataFrame,
    dataset_valid: pl.DataFrame,
    human_valid: pl.DataFrame,
    human_test: pl.DataFrame,
) -> None:

    model = mpra_predictor.load.load_model_structure(CONFIG, load_pretrained_base_weights=True).to(DEVICE)
    sei_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=4096)
    mal_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=600)

    train_size = dataset_train.shape[0]
    train_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=dataset_train,
        target_columns=TARGETS,
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
        augment=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    valid_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=dataset_valid,
        target_columns=TARGETS,
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
        augment=False,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'])

    hval_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=human_valid,
        target_columns=["log2FoldChange"],
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
    )
    hval_loader = DataLoader(hval_dataset, batch_size=CONFIG['batch_size'])

    htest_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=human_test,
        target_columns=["log2FoldChange"],
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
    )
    htest_loader = DataLoader(htest_dataset, batch_size=CONFIG['batch_size'])

    loss_fn = mpra_predictor.train.L1WithNaNs()
    metric_fn = mpra_predictor.train.pcc_score_with_nans
    optimizer = torch.optim.AdamW([{"params": model.non_base_parameters(), "lr": CONFIG["learning_rate"]},], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        div_factor=2,
        final_div_factor=100,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )

    if not LOGGING_PATH.exists():
        with open(LOGGING_PATH, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ["train_size", "seed", "epoch", "train_loss", "val_loss", "train_pcc_overall", "val_pcc_overall"] +
                [f"train_pcc_{i}" for i in range(len(TARGETS))] +
                [f"val_pcc_{i}" for i in range(len(TARGETS))] +
                ["human_val_pcc", "human_test_pcc"]
            )

    best_valid_loss = float('inf')
    patience_counter = 0
    for epoch in range(N_EPOCHS):
        model.train()
        for name, module in model.named_children():  # NOTE: we freeze and eval base models (Sei and Malinois)
            if name.endswith('_base'):
                module.eval()

        train_losses = []
        train_metrics = []
        train_preds = []
        train_targets = []
        for *x_batch, y_batch in train_loader:
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
        avg_train_loss = np.mean(train_losses)
        avg_train_metric = np.nanmean(train_metrics, axis=0)
        all_train_preds = torch.cat(train_preds, dim=0)
        all_train_targets = torch.cat(train_targets, dim=0)
        overall_train_pcc = metric_fn(all_train_preds, all_train_targets, per_target=False)

        model.eval()
        valid_losses = []
        valid_metrics = []
        valid_preds = []
        valid_targets = []
        with torch.no_grad():
            for *x_batch, y_batch in valid_loader:
                preds = model(*x_batch)
                loss = loss_fn(preds, y_batch)
                metric = metric_fn(preds, y_batch, per_target=True)
                valid_losses.append(loss.item())
                valid_metrics.append(np.array(metric))
                valid_preds.append(preds.detach().cpu())
                valid_targets.append(y_batch.detach().cpu())
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_metric = np.nanmean(valid_metrics, axis=0)
        all_valid_preds = torch.cat(valid_preds, dim=0)
        all_valid_targets = torch.cat(valid_targets, dim=0)
        overall_val_pcc = metric_fn(all_valid_preds, all_valid_targets, per_target=False)

        hval_preds = []
        hval_targets = []
        htest_preds = []
        htest_targets = []
        with torch.no_grad():
            for *x_batch, y_batch in hval_loader:
                hval_preds.append(model(*x_batch)[:, CREST_JURKAT_INDEX])
                hval_targets.append(y_batch[:, 0])
            for *x_batch, y_batch in htest_loader:
                htest_preds.append(model(*x_batch)[:, CREST_JURKAT_INDEX])
                htest_targets.append(y_batch[:, 0])
        overall_hval_pcc = metric_fn(torch.cat(hval_preds, dim=0), torch.cat(hval_targets, dim=0), per_target=False)
        overall_htest_pcc = metric_fn(torch.cat(htest_preds, dim=0), torch.cat(htest_targets, dim=0), per_target=False)

        with open(LOGGING_PATH, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [train_size, seed, epoch + 1, avg_train_loss, avg_valid_loss, overall_train_pcc, overall_val_pcc] +
                avg_train_metric.tolist() +
                avg_valid_metric.tolist() +
                [overall_hval_pcc, overall_htest_pcc]
            )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


def malinois_baseline(
    dataset_valid: pl.DataFrame,
    human_valid: pl.DataFrame,
    human_test: pl.DataFrame,
) -> None:

    model = malinois_module.load_pretrained_weights(
        DATA_DIR / "malinois_model/artifacts", model_cls=malinois_module.BassetBranched, freeze=True,
    ).to(DEVICE).eval()
    metric_fn = mpra_predictor.train.pcc_score_with_nans

    sei_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=4096)
    mal_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=600)

    valid_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=dataset_valid,
        target_columns=TARGETS,
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
        augment=False,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'])

    hval_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=human_valid,
        target_columns=["log2FoldChange"],
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
    )
    hval_loader = DataLoader(hval_dataset, batch_size=CONFIG['batch_size'])

    htest_dataset = mpra_predictor.dataloader.FusedDataset(
        dataframe=human_test,
        target_columns=["log2FoldChange"],
        sei_flank_builder=sei_flank_builder,
        mal_flank_builder=mal_flank_builder,
        device=DEVICE,
    )
    htest_loader = DataLoader(htest_dataset, batch_size=CONFIG['batch_size'])

    ol49_k562_preds, ol49_jurkat_targets, ol49_k562_targets = [], [], []
    hval_preds, hval_targets = [], []
    htest_preds, htest_targets = [], []
    with torch.no_grad():
        for (_, mal_input, y_batch) in valid_loader:
            ol49_k562_preds.append(model(mal_input)[:, MALINOIS_K562_INDEX])
            ol49_jurkat_targets.append(y_batch[:, CREST_JURKAT_INDEX])
            ol49_k562_targets.append(y_batch[:, CREST_K562_INDEX])
        for (_, mal_input, y_batch) in hval_loader:
            hval_preds.append(model(mal_input)[:, MALINOIS_K562_INDEX])
            hval_targets.append(y_batch[:, 0])
        for (_, mal_input, y_batch) in htest_loader:
            htest_preds.append(model(mal_input)[:, MALINOIS_K562_INDEX])
            htest_targets.append(y_batch[:, 0])

    ol49_k562_preds = torch.cat(ol49_k562_preds, dim=0)
    ol49_jurkat_targets = torch.cat(ol49_jurkat_targets, dim=0)
    ol49_k562_targets = torch.cat(ol49_k562_targets, dim=0)
    pl.DataFrame({
        "OL49_baseline_pcc": metric_fn(ol49_k562_preds, ol49_jurkat_targets, per_target=False),
        "OL49_sota_pcc": metric_fn(ol49_k562_preds, ol49_k562_targets, per_target=False),
        "human_valid_pcc": metric_fn(torch.cat(hval_preds, dim=0), torch.cat(hval_targets, dim=0), per_target=False),
        "human_test_pcc": metric_fn(torch.cat(htest_preds, dim=0), torch.cat(htest_targets, dim=0), per_target=False)
    }).write_csv(DATA_DIR / "trained_models/logs/dataset_malinois_k562_baseline.csv")


if __name__ == "__main__":

    # dataset with viral tiles and MPRA activity in 6 cell types
    dataset = pl.read_csv(DATA_DIR / "OL49.csv")
    dataset_train = dataset.filter(pl.col("split") == "train")
    dataset_valid = dataset.filter(pl.col("split") == "val")

    # dataset with human genome tiles and MPRA activity in Jurkat
    human_dataset = (
        pl.read_csv(DATA_DIR / "Rodrigo_all_jurkat_processed_10022026.csv", infer_schema_length=None)
        .filter(~pl.col("ID").is_in(dataset["ID"].to_numpy()))
        .filter(pl.col("DNA_mean") >= 50)
    )
    human_valid = human_dataset.filter((pl.col("project") == "TGWAS") & (pl.col("chr").is_in(["19", "21", "X"])))
    human_test = human_dataset.filter((pl.col("project") == "TGWAS") & (pl.col("chr").is_in(["7", "13"])))

    malinois_baseline(dataset_valid, human_valid, human_test)

    SIZE_GRID = np.concat([np.arange(100, 2_100, 100), np.arange(5_000, 50_000, 5_000), [dataset_train.shape[0]]])
    for train_size in SIZE_GRID:
        for seed in range(N_REPLICAS):
            ds_train = dataset_train.sample(train_size, with_replacement=False, seed=seed)
            train_and_eval_loop(seed, ds_train, dataset_valid, human_valid, human_test)
