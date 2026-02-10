import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_res_blocks: int, dropout: float, output_size: int,):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.output_size = output_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_size, dropout=dropout) for _ in range(num_res_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, output_size),
        )

    def __str__(self):
        return f"MLP:{self.hidden_size}-{self.num_res_blocks}-{self.dropout}-{self.output_size}"

    def forward(self, x):
        x_proj = self.input_proj(x)  # (B, hidden_size)
        x_res = self.res_blocks(x_proj)  # (B, hidden_size)
        return self.output_head(x_res + x_proj)  # (B, output_size)


class GatedFusion(nn.Module):
    def __init__(self, left_dim, right_dim, output_dim, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        self.left_proj = nn.Linear(left_dim, output_dim)
        self.right_proj = nn.Linear(right_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Sigmoid(),
        )

    def __str__(self):
        return f"Fusion:{str(self.output_dim)}-{str(self.dropout)}"

    def forward(self, left_out, right_out):
        left_proj = self.left_proj(left_out)
        right_proj = self.right_proj(right_out)
        concat = torch.cat([left_proj, right_proj], dim=1)
        gate_weight = self.gate(concat)
        fused = gate_weight * left_proj + (1 - gate_weight) * right_proj
        return fused


class FusedClassifier(nn.Module):
    def __init__(
        self, sei_base, sei_pooler,
        malinois_base, malinois_pooler,
        fusion, classifier,
    ):
        super().__init__()
        self.sei_base = sei_base
        self.sei_pooler = sei_pooler
        self.mal_base = malinois_base
        self.mal_pooler = malinois_pooler
        self.fusion = fusion
        self.classifier = classifier

    def __str__(self):
        return f"FusedClass_S{str(self.sei_pooler)}_M{str(self.mal_pooler)}_{str(self.fusion)}_{str(self.classifier)}"

    def base_parameters(self):
        return list(self.sei_base.parameters()) + list(self.mal_base.parameters())

    def non_base_parameters(self):
        base_param_ids = {id(p) for p in self.base_parameters()}
        return [p for p in self.parameters() if id(p) not in base_param_ids and p.requires_grad]

    def forward(self, x_sei, x_mal):
        sei_out = self.sei_pooler(self.sei_base(x_sei))  # (B, SP)
        mal_out = self.mal_pooler(self.mal_base(x_mal))  # (B, MP)
        fused = self.fusion(sei_out, mal_out)
        out = self.classifier(fused)  # (B, output_size)
        return out
