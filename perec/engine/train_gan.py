import torch

from tqdm import tqdm


def train_gan(model, data_loader, optimizer, cur_epoch):
    # unpack optimizer and model
    optimD, optimG = optimizer
    netD, netG = model.netD, model.netG

    lossD, lossG, reward = 0, 0, 0
    gan_lossD, gan_lossG = 0, 0
    regsD, regsG = 0, 0

    tbar = tqdm(data_loader, ascii=True)
    for data_batch in tbar:
        tbar.set_description("Epoch {}".format(cur_epoch))

        if torch.cuda.is_available():
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        user = data_batch["user"]
        negs = data_batch["neg"]
        pos = data_batch["pos"]

        with torch.no_grad():
            good_neg = netG.throw(user, negs)

        batch_gan_lossD, batch_regsD = netD(user, pos, good_neg, negs=negs)
        batch_lossD = batch_gan_lossD + batch_regsD

        optimD.zero_grad()
        batch_lossD.backward()
        optimD.step()

        gan_lossD += batch_gan_lossD
        lossD += batch_lossD
        regsD += batch_regsD

        with torch.no_grad():
            batch_reward = netD.step(user, negs)

        batch_gan_lossG, batch_regsG = netG(user, negs, batch_reward)
        batch_lossG = batch_gan_lossG + batch_regsG

        optimG.zero_grad()
        batch_lossG.backward()
        optimG.step()

        gan_lossG += batch_gan_lossG
        lossG += batch_lossG
        regsG += batch_regsG
        reward += torch.mean(batch_reward)

    loss_str = (
        "\n Dis Train loss: [{0:.5f} = {1:.5f} + {2:.5f}] \n Gen Train loss: [{3:.5f} = {4:.5f} + {5:.5f}]"
        "\n Reward: {6:.5f}".format(
            lossD, gan_lossD, regsD, lossG, gan_lossG, regsG, reward,
        )
    )

    return loss_str
