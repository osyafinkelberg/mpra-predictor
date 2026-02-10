import typing as tp
from pathlib import Path

from . import sei_base as sei_module
from . import malinois_base as malinois_module
from . import pooler as pooler_module
from . import classifier as classifier_module

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_model_structure(PARAM_CONFIG: dict[str, str | dict[str, tp.Any]]) -> classifier_module.FusedClassifier:
    sei_base = sei_module.load_pretrained_weights(
        weights_path=DATA_DIR / "sei_model/sei.pth", model_cls=sei_module.SeiWithoutSpline, freeze=True,
    )
    sei_pooler = pooler_module.PositionWeight(
        channels=sei_base.out_channels,
        seq_len=sei_base.out_length,
        n_heads=PARAM_CONFIG['sei_pooler']['n_heads'],
        hidden_dim=PARAM_CONFIG['sei_pooler']['hidden_dim'],
        pos_emb_dim=PARAM_CONFIG['sei_pooler']['pos_emb_dim'],
        dropout=PARAM_CONFIG['sei_pooler']['dropout'],
    )

    mal_base = malinois_module.load_pretrained_weights(
        DATA_DIR / "malinois_model/artifacts", malinois_module.MalinoisEncoder, freeze=True,
    )
    mal_pooler = pooler_module.PositionWeight(
        channels=mal_base.out_channels,
        seq_len=mal_base.out_length,
        n_heads=PARAM_CONFIG['mal_pooler']['n_heads'],
        hidden_dim=PARAM_CONFIG['mal_pooler']['hidden_dim'],
        pos_emb_dim=PARAM_CONFIG['mal_pooler']['pos_emb_dim'],
        dropout=PARAM_CONFIG['mal_pooler']['dropout'],
    )

    fusion = classifier_module.GatedFusion(
        left_dim=sei_pooler.out_size, right_dim=mal_pooler.out_size,
        output_dim=PARAM_CONFIG['fusion']['output_dim'],
        dropout=PARAM_CONFIG['fusion']['dropout'],
    )

    classifier = classifier_module.MLP(
        input_size=fusion.output_dim,
        hidden_size=PARAM_CONFIG['mlp']['hidden_size'],
        num_res_blocks=PARAM_CONFIG['mlp']['num_res_blocks'],
        dropout=PARAM_CONFIG['mlp']['dropout'],
        output_size=len(PARAM_CONFIG['target_columns']),
    )

    model = classifier_module.FusedClassifier(sei_base, sei_pooler, mal_base, mal_pooler, fusion, classifier)
    return model
