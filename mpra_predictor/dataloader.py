from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DNA_BASES = ['A', 'C', 'G', 'T']
DNA_COMPLEMENT_MAP = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def dna_to_tensor(sequence_str, vocab_list=DNA_BASES):
    seq_tensor = np.zeros((len(vocab_list), len(sequence_str)))
    for letter_idx, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), letter_idx] = 1
    seq_tensor = torch.Tensor(seq_tensor)
    return seq_tensor


def tensor_to_dna(one_hot_tensor, vocab_list=DNA_BASES):
    if isinstance(one_hot_tensor, torch.Tensor):
        one_hot_tensor = one_hot_tensor.detach().cpu().numpy()
    base_indices = one_hot_tensor.argmax(axis=0)
    sequence = ''.join(vocab_list[idx] for idx in base_indices)
    return sequence


def reverse_complement_onehot(x):
    comp_alphabet = [DNA_COMPLEMENT_MAP[nt] for nt in DNA_BASES]
    permutation = [DNA_BASES.index(nt) for nt in comp_alphabet]
    return torch.flip(x[..., permutation, :], dims=[-1])


class FlankBuilder():
    def __init__(
        self, up_flank: torch.Tensor, down_flank: torch.Tensor,
        target_size: int = 4096, rotation: int = 0
    ):
        self.up_flank = up_flank.detach().clone()
        self.down_flank = down_flank.detach().clone()
        self.target_size = target_size
        self.unit_size = up_flank.shape[-1] + down_flank.shape[-1]
        self.rotation = rotation

    def set_rotation(self, new_rotation: int) -> None:
        self.rotation = new_rotation

    def add_flanks(self, tiles: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if not isinstance(tiles, list):
            tiles = [tiles]
        flanked = []
        for tile in tiles:
            assert tile.ndim == 2, "Only supports [C, W] tensors"
            tile_size = tile.shape[-1]
            assert tile_size <= self.target_size, f"Tile size {tile_size} should be ≤ target size {self.target_size}"
            if tile_size == self.target_size:
                construct = tile
            else:
                unit_size = self.unit_size + tile_size
                concatemer_deg = (self.target_size + unit_size - 1) // unit_size
                concatemer_deg += (concatemer_deg + 1) % 2  # ensure odd repeat count for symmetry
                construct = torch.cat([self.up_flank, tile, self.down_flank], dim=-1).repeat(1, concatemer_deg)
                if self.rotation != 0:  # positive rotation value correspond to tile-downstream position
                    construct = torch.roll(construct, shifts=-self.rotation, dims=-1)
                flank_size = (unit_size * concatemer_deg - self.target_size) // 2
                construct = construct[..., flank_size: flank_size + self.target_size]
                assert construct.shape[-1] == self.target_size, f"{construct.shape[-1]} != {self.target_size}"
            flanked.append(construct)
        flanked = torch.stack(flanked)
        return flanked


def prepare_flank_builder(input_size: int = 4096, rotation: int = 0) -> FlankBuilder:
    plasmid_dct = dict()
    with open(DATA_DIR / "mpra_plasmid.fa") as fa:
        lines = [line.strip().strip('>') for line in fa.readlines()]
        for header_idx in range(0, len(lines), 2):
            plasmid_dct[lines[header_idx]] = lines[header_idx + 1]
    up_flank = dna_to_tensor(plasmid_dct['backbone_upstream'])
    down_flank = dna_to_tensor(plasmid_dct['backbone_downstream'])
    flank_builder = FlankBuilder(up_flank, down_flank, target_size=input_size, rotation=rotation)
    return flank_builder


class FusedDataset(Dataset):
    def __init__(
        self, dataframe: pl.dataframe, target_columns: list[str],
        sei_flank_builder: FlankBuilder, mal_flank_builder: FlankBuilder,
        device: str, augment: bool = False
    ):
        self.sequences = dataframe["sequence"].to_numpy()
        self.targets = dataframe[target_columns].to_numpy().astype(np.float32)
        self.sei_flank_builder = sei_flank_builder
        self.mal_flank_builder = mal_flank_builder
        self.device = device
        self.rng = np.random.default_rng(0)
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        onehot = dna_to_tensor(self.sequences[idx])
        if self.augment:
            sei_rotation = int(self.rng.integers(low=-10, high=11, size=1))
            self.sei_flank_builder.set_rotation(sei_rotation)
            mal_rotation = int(self.rng.integers(low=-10, high=11, size=1))
            self.mal_flank_builder.set_rotation(mal_rotation)
        sei_inp = self.sei_flank_builder.add_flanks([onehot])[0]
        mal_inp = self.mal_flank_builder.add_flanks([onehot])[0]
        y = torch.tensor(self.targets[idx], device=self.device)
        return sei_inp.to(self.device), mal_inp.to(self.device), y
