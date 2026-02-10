import typing as tp
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from . import dataloader
from . import classifier


VALID_BASES = {'A', 'C', 'G', 'T'}


class Predictor(nn.Module):
    def __init__(self, model: nn.Module, pred_idx: int):
        super().__init__()
        self.model = model
        self.pred_idx = pred_idx
        self.model.eval()

    def forward(self, *in_tensor: torch.Tensor):
        out_tensor = self.model(*in_tensor)[:, self.pred_idx]
        return out_tensor


def isg_contributions(
    sequences: torch.Tensor,
    predictor: nn.Module,
    num_steps: int = 50,
    step_chunk_size: int = 10,
    eval_batch_size: int = 64,
    DEVICE: str = 'cuda',
    baseline: tp.Optional[torch.Tensor] = None,
    real_contributions: bool = False,
    use_tqdm: bool = True,
) -> torch.Tensor:
    """
    Compute Integrated Sampled Gradients (ISG) for a batch of one-hot encoded DNA sequences.
    Based on: Sundararajan, Taly & Yan, “Axiomatic Attribution for Deep Networks” (arXiv:1703.01365)
    See: https://github.com/ankurtaly/Integrated-Gradients?tab=readme-ov-file
    Args:
        sequences (torch.Tensor): [batch, seq_len, 4] one-hot DNA input
        predictor (nn.Module): Trained model to predict MPRA values from one-hot inputs
        num_steps (int): Total number of integration steps (alphas)
        step_chunk_size (int): Number of steps to process at once (reduce if OOM)
        eval_batch_size (int): Number of sequences per batch
        DEVICE (str): Device to run on, 'cuda' or 'cpu'
        baseline (Optional[torch.Tensor]): Baseline input tensor of same shape as `sequences`.
            Defaults to a zero tensor of the same shape.
        real_contributions (bool): If True, return real ISG contributions (delta * avg_grads).
            If False (default), return hypothetical contributions (avg_grads only).
    Returns:
        torch.Tensor: ISG contributions, shape [batch, seq_len, 4]
    """
    sequences = sequences.to(DEVICE)
    if baseline is None:
        baseline = torch.zeros_like(sequences)
    else:
        baseline = baseline.to(DEVICE)

    alphas = torch.linspace(0.0, 1.0, steps=num_steps, device=DEVICE).view(-1, 1, 1, 1)
    all_contributions = []

    iters = tqdm(range(0, sequences.size(0), eval_batch_size))
    if use_tqdm:
        iters = tqdm(iters)
    for batch_start in iters:
        batch_end = batch_start + eval_batch_size
        input_batch = sequences[batch_start: batch_end]
        baseline_batch = baseline[batch_start: batch_end]
        delta = input_batch - baseline_batch  # shape: [batch, seq_len, 4]

        accumulated_grads = torch.zeros_like(input_batch)

        for step_start in range(0, num_steps, step_chunk_size):
            step_end = min(step_start + step_chunk_size, num_steps)
            step_alphas = alphas[step_start: step_end]  # shape: [chunk_size, 1, 1, 1]

            interp_inputs = baseline_batch.unsqueeze(0) + step_alphas * delta.unsqueeze(0)
            interp_inputs.requires_grad = True

            flat_inputs = interp_inputs.view(-1, *input_batch.shape[1:])
            preds = predictor(flat_inputs)  # [chunk_size * batch]
            preds = preds.view(-1, input_batch.shape[0])  # [chunk_size, batch]

            grads = torch.autograd.grad(
                outputs=preds.sum(),
                inputs=interp_inputs,
                create_graph=False,
                retain_graph=False
            )[0]  # [chunk_size, batch, seq_len, 4]

            weights = torch.ones(grads.shape[0], device=DEVICE)  # trapezoidal rule for numerical integration
            if step_start == 0:
                weights[0] = 0.5
            if step_end == num_steps:
                weights[-1] = 0.5

            weighted_grads = grads * weights.view(-1, 1, 1, 1)
            accumulated_grads += weighted_grads.sum(dim=0)

        avg_grads = accumulated_grads / (num_steps - 1)
        if real_contributions:
            isg = delta * avg_grads  # only real ISG, should sum up to the predicted MPRA activity
        else:
            isg = avg_grads  # hypothetical contribution

        all_contributions.append(isg.detach().cpu())

    return torch.cat(all_contributions, dim=0)  # shape: [total_batch, seq_len, 4]


def isg_contributions_multi_input(
    inputs: tp.Sequence[torch.Tensor],
    predictor: nn.Module,
    baselines: tp.Optional[tp.Sequence[tp.Optional[torch.Tensor]]] = None,
    num_steps: int = 50,
    step_chunk_size: int = 10,
    eval_batch_size: int = 64,
    DEVICE: str = 'cuda',
    real_contributions: bool = False,
    use_tqdm: bool = True,
) -> tp.List[torch.Tensor]:
    inputs = [x.to(DEVICE) for x in inputs]
    batch_size = inputs[0].shape[0]
    num_inputs = len(inputs)

    if baselines is None:
        baselines = [torch.zeros_like(x) for x in inputs]
    else:
        baselines = [
            torch.zeros_like(inp) if b is None else b.to(DEVICE)
            for inp, b in zip(inputs, baselines)
        ]

    deltas = [inp - base for inp, base in zip(inputs, baselines)]
    alphas = torch.linspace(0.0, 1.0, steps=num_steps, device=DEVICE).view(-1, *[1] * (inputs[0].dim()))

    all_contributions = [torch.zeros_like(inp) for inp in inputs]

    iters = range(0, batch_size, eval_batch_size)
    if use_tqdm:
        iters = tqdm(iters)
    for batch_start in iters:
        batch_end = batch_start + eval_batch_size
        batch_inputs = [x[batch_start:batch_end] for x in inputs]
        batch_bases = [b[batch_start:batch_end] for b in baselines]
        batch_deltas = [d[batch_start:batch_end] for d in deltas]

        grads_accum = [torch.zeros_like(x) for x in batch_inputs]

        for step_start in range(0, num_steps, step_chunk_size):
            step_end = min(step_start + step_chunk_size, num_steps)
            step_alphas = alphas[step_start:step_end]

            chunk_size = step_alphas.shape[0]

            for step in range(chunk_size):
                alpha = step_alphas[step]

                interp_inputs = [
                    base + alpha * delta for base, delta in zip(batch_bases, batch_deltas)
                ]

                for inp in interp_inputs:
                    inp.requires_grad_(True)

                preds = predictor(*interp_inputs)
                loss = preds.sum()

                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=interp_inputs,
                    create_graph=False,
                    retain_graph=False
                )

                weight = 1.0
                if step_start == 0 and step == 0:
                    weight = 0.5
                if step_end == num_steps and step == chunk_size - 1:
                    weight = 0.5

                for i in range(num_inputs):
                    grads_accum[i] += grads[i] * weight

                del grads, interp_inputs, loss, preds
                torch.cuda.empty_cache()

        for i in range(num_inputs):
            avg_grads = grads_accum[i] / (num_steps - 1)
            delta = batch_deltas[i]
            isg = delta * avg_grads if real_contributions else avg_grads
            all_contributions[i][batch_start:batch_end] = isg.detach().cpu()

        torch.cuda.empty_cache()

    return all_contributions


class ModelInterpreter:
    def __init__(self, model: classifier.FusedClassifier, target_index: int, batch_size: int, device: str):
        self.batch_size = batch_size
        self.batch_sequences = list()
        self.valid_ids = list()
        self.onehots = list()
        self.contribs = list()
        self.device = device
        self.sei_flank_builder = dataloader.prepare_flank_builder(input_size=4096)
        self.mal_flank_builder = dataloader.prepare_flank_builder(input_size=600)
        self.predictor = Predictor(model, pred_idx=target_index).to(self.device).eval()

    def update(self, tile_id: str, tile_sequence: str) -> None:
        if not self.validate_tile_sequence(tile_sequence):
            return
        self.valid_ids.append(tile_id)
        self.batch_sequences.append(tile_sequence)
        if len(self.batch_sequences) == self.batch_size:
            self.contribs.append(self.batch_infer(self.batch_sequences))
            self.batch_sequences.clear()

    @staticmethod
    def validate_tile_sequence(tile_sequence: str) -> bool:
        if len(tile_sequence) != 200:
            return False
        if len(set(tile_sequence) - VALID_BASES) != 0:  # invalid sequence
            return False
        return True

    def batch_infer(self, batch_sequences: list[str]) -> np.ndarray:
        onehots = [dataloader.dna_to_tensor(seq) for seq in batch_sequences]
        self.onehots.append(onehots)
        onehots_sei = self.sei_flank_builder.add_flanks(onehots).to(self.device)
        onehots_mal = self.mal_flank_builder.add_flanks(onehots).to(self.device)
        contribs_lst = isg_contributions_multi_input(
            [onehots_sei, onehots_mal], self.predictor, num_steps=50, step_chunk_size=10, use_tqdm=False
        )
        contribs_sei = contribs_lst[0][..., 1948: 2148]
        contribs_mal = contribs_lst[1][..., 200: 400]
        contribs = contribs_sei + contribs_mal
        return contribs.cpu().numpy().astype(np.float32)

    def get_predictions(self,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.batch_sequences:
            self.contribs.append(self.batch_infer(self.batch_sequences))
            self.batch_sequences.clear()
        valid_ids = np.array(self.valid_ids)
        onehots = np.concatenate(self.onehots)
        contribs = np.concatenate(self.contribs)
        return valid_ids, onehots, contribs
