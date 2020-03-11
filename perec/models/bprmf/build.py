from .bprmf import BPRMF


def build_bprmf(data_param, cfg):
    model = BPRMF(
        n_users=data_param.n_users,
        n_items=data_param.n_items,
        embed_size=cfg.MODEL.EMBEDDING_SIZE,
        regs=cfg.MODEL.REGS,
    )

    return model
