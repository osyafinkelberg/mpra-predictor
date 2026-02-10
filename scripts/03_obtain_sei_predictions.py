from pathlib import Path
import sys
import polars as pl
import numpy as np
import torch
from tqdm import tqdm
import h5py

sys.path.insert(0, '../data/sei_model/')
from sei import Sei  # noqa E402

sys.path.insert(0, '..')
import mpra_predictor  # noqa E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100


def get_sei_predictions() -> None:

    dataset = (
        pl.read_csv(DATA_DIR / "OL53.csv")
        .filter(~pl.col("sequence").is_null())
        .unique(subset=["sequence"], keep="first", maintain_order=True)
        .sort("ID", "sequence")
    )
    sequences = dataset["sequence"].to_numpy()

    flank_builder = mpra_predictor.dataloader.prepare_flank_builder(input_size=4096)
    pretrained_model = mpra_predictor.sei_base.load_pretrained_weights(
        weights_path=DATA_DIR / "sei_model/sei.pth", model_cls=Sei, freeze=True,
    ).to(DEVICE)

    with open(DATA_DIR / "sei_model/target.names", 'r') as f:
        track_names = [line.strip() for line in f.readlines()]

    track_predictions = []
    with torch.no_grad():
        for i in tqdm(range((len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE)):
            batch_seqs = sequences[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            batch_raw = [mpra_predictor.dataloader.dna_to_tensor(seq) for seq in batch_seqs]
            batch_tensor = flank_builder.add_flanks(batch_raw).to(DEVICE)
            batch_preds = pretrained_model(batch_tensor)
            track_predictions.append(batch_preds.cpu())
    track_predictions = torch.cat(track_predictions, dim=0).numpy().astype(np.float32)

    with h5py.File(DATA_DIR / 'OL53_sei_predictions.h5', 'w') as h5f:
        h5f.create_dataset('track_predictions', data=track_predictions, compression='gzip')
        h5f.create_dataset('ID', data=np.array(dataset["ID"], dtype='S'), compression='gzip')
        h5f.create_dataset('track_names', data=np.array(track_names, dtype='S'), compression='gzip')


if __name__ == "__main__":
    get_sei_predictions()
