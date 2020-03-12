import time

import torch

from perec.models import build_model
from perec.solver import build_optimizer
from perec.engine import train_mf, train_gan, test_model_v2
from perec.data import build_dataloader
from perec.utils.torch_utils import set_random_seed
from perec.utils.variable import Data_params, User_dict
from perec.utils.misc import print_dict

_MODEL_TRAINERS = {
    "BPR-MF": train_mf,
    "DNS": train_mf,
    "IRGAN": train_gan,
    "NMRN": train_gan,
    "Advir": train_gan,
}


def train_model(model, data_loader, optimizer, cur_epoch, cfg):
    return _MODEL_TRAINERS[cfg.MODEL.TYPE](model, data_loader, optimizer, cur_epoch)


def train(cfg, output_dir=""):
    set_random_seed(cfg.RANDOM_SEED)

    train_data_loader = build_dataloader(cfg)
    dataset = train_data_loader.dataset

    data_params = Data_params(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        n_train=dataset.n_train,
        n_test=dataset.n_test,
    )

    user_dict = User_dict(
        train_user_dict=dataset.train_dict,
        test_user_dict=dataset.test_dict,
    )

    # build model
    model = build_model(data_params=data_params, cfg=cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # ------------------------------------------------------------------
    # Epoch-based training
    # ------------------------------------------------------------------
    max_epoch = cfg.TRAIN.MAX_EPOCH
    best_metric_name = cfg.TEST.METRIC
    best_metric, best_epoch, early_stop = 0, 0, 0
    best_result = None
    for epoch in range(max_epoch):
        cur_epoch = epoch + 1
        start_time = time.time()
        loss_str = train_model(
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            cur_epoch=cur_epoch,
            cfg=cfg,
        )
        epoch_time = time.time() - start_time

        print(
            "Epoch[{}]-Train {} \n total_time: {:.2f}s".format(
                cur_epoch, loss_str, epoch_time
            )
        )

        if cur_epoch % cfg.TRAIN.LOG_PERIOD == 0 or cur_epoch == max_epoch:
            with torch.no_grad():
                result = test_model_v2(model, cfg.TEST.KS, user_dict)
            print("Epoch[{}]-Test".format(cur_epoch))
            print_dict(result)

            cur_metric = result[best_metric_name][0]
            if cur_metric > best_metric:
                early_stop = 0
                best_metric, best_epoch = cur_metric, cur_epoch
                best_result = result
            else:
                early_stop += 1

            if early_stop == cfg.TRAIN.EARLY_STOP:
                break

    print("-" * 75)
    print("Best Epoch[{}]".format(best_epoch))
    print_dict(best_result)

    return model
