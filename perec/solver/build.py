import torch


def build_optimizer(cfg, model):
    train_type = cfg.TRAIN.TYPE

    if train_type == "MF":
        name = cfg.SOLVER.TYPE
        if hasattr(torch.optim, name):
            def builder(cfg, model):
                return getattr(torch.optim, name)(
                    model.parameters(),
                    lr=cfg.SOLVER.BASE_LR,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAY
                )
        else:
            raise ValueError("Unsupported type of optimizer.")
    elif train_type == "GAN":
        nameD = cfg.SOLVER.TYPED
        nameG = cfg.SOLVER.TYPEG
        if hasattr(torch.optim, nameD):
            def builderD(cfg, model):
                return getattr(torch.optim, nameD)(
                    model.netD.parameters(),
                    lr=cfg.SOLVER.BASE_LRD,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAYD
                )
        else:
            raise ValueError("Unsupported type of optimizer for Dis")

        if hasattr(torch.optim, nameG):
            def builderG(cfg, model):
                return getattr(torch.optim, nameG)(
                    model.netG.parameters(),
                    lr=cfg.SOLVER.BASE_LRG,
                    weight_decay=cfg.SOLVER.WEIGHT_DECAYG
                )
        else:
            raise ValueError("Unsupported type of optimizer for Gen")

        def builder(cfg, model):
            return builderD(cfg, model), builderG(cfg, model)

    else:
        raise NotImplementedError("Not Implemented.")

    return builder(cfg, model)


