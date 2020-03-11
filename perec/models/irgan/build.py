from .irgan import IRGAN


def build_irgan(data_param, cfg):
    model = IRGAN(
        n_users=data_param.n_users,
        n_items=data_param.n_items,
        embed_size=cfg.MODEL.EMBEDDING_SIZE,
        regsD=cfg.MODEL.REGSD,
        regsG=cfg.MODEL.REGSG
    )

    return model
