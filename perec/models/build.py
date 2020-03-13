from .bprmf import build_bprmf
from .dns import build_dns
from .irgan import build_irgan
from .advir import build_advir

_MODEL_BUILDER = {
    "BPR-MF": build_bprmf,
    "DNS": build_dns,
    "IRGAN": build_irgan,
    "ADVIR": build_advir,
}


def build_model(data_params, cfg):
    return _MODEL_BUILDER[cfg.MODEL.TYPE](data_params, cfg)
