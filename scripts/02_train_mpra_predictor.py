from pathlib import Path
import sys
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # noqa F401

sys.path.insert(0, '..')
import mpra_predictor  # noqa E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = DATA_DIR / "trained_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    N_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    CONFIG = {
        'batch_size': 256,
        'learning_rate': 1e-3,
        'sei_pooler': {'n_heads': 1, 'hidden_dim': 128, 'pos_emb_dim': 8, 'dropout': 0.5, },
        'mal_pooler': {'n_heads': 1, 'hidden_dim': 64, 'pos_emb_dim': 8, 'dropout': 0.5, },
        'fusion': {'output_dim': 128, 'dropout': 0.7},
        'mlp': {'hidden_size': 128, 'num_res_blocks': 0, 'dropout': 0.5},
        'target_columns': ['lfc_mean_Jurkat',],
        # 'target_columns': [
        #     'lfc_mean_GM12878', 'lfc_mean_Jurkat', 'lfc_mean_MRC5',
        #     'lfc_mean_A549', 'lfc_mean_HEK293', 'lfc_mean_K562'
        # ]
    }

    dataset = pl.read_csv(DATA_DIR / "OL49.csv")
    dataset_train = dataset.filter(pl.col("split") == "train")
    dataset_valid = dataset.filter(pl.col("split") == "val")
    TARGETS = CONFIG['target_columns']

    model = mpra_predictor.load.load_model_structure(CONFIG, load_pretrained_base_weights=True).to(DEVICE)
    model.to(DEVICE)

    weights_path = OUTPUT_DIR / f"{str(model)}.pt"
    logging_path = OUTPUT_DIR / (
        f"logs/BS{CONFIG['batch_size']}_LR{CONFIG['learning_rate']:.0e}_{str(model)}.csv"
    )

    sei_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=4096)
    mal_flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=600)

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

    mpra_predictor.train.train_mpra_predictor(
        model,
        train_loader,
        valid_loader,
        TARGETS,
        CONFIG['batch_size'],
        CONFIG['learning_rate'],
        N_EPOCHS,
        EARLY_STOPPING_PATIENCE,
        weights_path=weights_path,
        logging_path=logging_path,
        verbose=True,
    )
