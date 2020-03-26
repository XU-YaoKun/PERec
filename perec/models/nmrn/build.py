from .nmrn import NMRN


def build_nmrn(data_param, cfg):
    model = NMRN(
        n_users=data_param.n_users,
        n_items=data_param.n_items,
        embed_size=cfg.MODEL.EMBEDDING_SIZE,
        regsD=cfg.MODEL.REGSD,
        regsG=cfg.MODEL.REGSG,
        margin=cfg.MODEL.MARGIN,
    )

    return model
