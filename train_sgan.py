from utils.utils import set_seed, initialize_weights
import argparse
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.read_cwt_figures import read_directory
from utils.losses import HingeLoss, gradient_normalize
from models import sgan
from utils.supcontrast_loss import SupContrastLoss
from utils.utils import get_label_data, model_test, Metric, str2bool

net_G = {'sgan': sgan.Generator}
net_D = {'sgan': sgan.Discriminator}
loss_fns = {'hinge': HingeLoss}


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # training
    parser.add_argument('--epochs', type=int, default=10000, help='max number of epoch')
    parser.add_argument('--add_supconloss', type=str2bool, default=True, help='whether add sp_loss?')
    parser.add_argument('--gn', type=str2bool, default=True, help='Gradient Normalization')
    # hyper parameters
    parser.add_argument('--n_critic', type=int, default=2, help='every epoch train D for n_critic')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.3, help='Trade-off parameter')
    parser.add_argument('--sup_wt', type=float, default=0.5, help='Trade-off parameter')
    parser.add_argument('--df_aux_wt', type=float, default=0.5, help='Trade-off parameter')
    # default
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--betas_g', nargs='+', type=float, default=(0.5, 0.999))
    parser.add_argument('--betas_d', nargs='+', type=float, default=(0.5, 0.999))
    parser.add_argument('--model', type=str, default='sgan')
    parser.add_argument('--loss', type=str, default='hinge')
    parser.add_argument('--n_train', type=int, default=30)
    parser.add_argument('--n_label', type=int, default=10)
    parser.add_argument('--n_test', type=int, default=30)
    parser.add_argument('--save_epochs', type=int, default=2000)
    parser.add_argument('--n_class', type=int, default=5, help='number of category')
    parser.add_argument('--z_dim', type=int, default=100, help='dimension of noise for generator')
    parser.add_argument('--n_channel', type=int, default=3, help='image channel')
    parser.add_argument('--img_size', type=int, default=64, help=' H x W')
    parser.add_argument('--real_folder', type=str, help=r'./datasets/Gear/real_images')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()
    return args


def train(args):
    # load data
    train_x, train_y = read_directory(os.path.join(args.real_folder, 'train'), args.n_train, args.n_class)
    train_x, train_y, label_x, label_y = get_label_data(train_x, train_y, args.n_class, args.n_label)
    valid_x, valid_y = read_directory(os.path.join(args.real_folder, 'valid'), args.n_test, args.n_class)
    test_x, test_y = read_directory(os.path.join(args.real_folder, 'test'), args.n_test, args.n_class)

    # dataset
    train_ds = TensorDataset(train_x, train_y)
    train_ds_label = TensorDataset(label_x, label_y)
    valid_ds = TensorDataset(valid_x, valid_y)
    test_ds = TensorDataset(test_x, test_y)

    times = int(np.ceil(train_ds.__len__() * 1. / train_ds_label.__len__()))
    t1 = train_ds_label.tensors[0].clone()
    t2 = train_ds_label.tensors[1].clone()
    train_ds_label = TensorDataset(t1.repeat(times, 1, 1, 1), t2.repeat(times))

    # dataloader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    train_dl_label = DataLoader(train_ds_label, batch_size=args.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    # model
    G = net_G[args.model](args.z_dim, args.n_class, args.img_size, args.n_channel).to(args.device)
    D = net_D[args.model](args.n_class, args.img_size, args.n_channel).to(args.device)
    G.apply(initialize_weights)
    D.apply(initialize_weights)

    # optimizer
    optimizer_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=args.betas_g)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=args.betas_d)

    # loss function
    loss_fn = loss_fns[args.loss]().to(args.device)
    auxiliary_fn = torch.nn.CrossEntropyLoss().to(args.device)
    supcontrast_fn = SupContrastLoss(args.temperature, args.device).to(args.device)

    # for loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        e_dloss, e_gloss, e_train_acc, e_train_loss = Metric(), Metric(), Metric(), Metric()
        e_dr_acc, e_dr_loss, e_df_acc, e_df_loss = Metric(), Metric(), Metric(), Metric()
        e_gf_acc, e_gf_loss, e_valid_loss, e_valid_acc, e_sploss = Metric(), Metric(), Metric(), Metric(), Metric()
        count = 0
        iter_label_dl = iter(train_dl_label)
        for step, (imgs, _) in enumerate(train_dl, start=1):
            n, imgs = imgs.shape[0], imgs.to(args.device)
            label_x, label_y = next(iter_label_dl)
            label_x, label_y = label_x.to(args.device), label_y.to(args.device)

            """update D"""
            optimizer_d.zero_grad()
            z = torch.normal(0, 1, (n, args.z_dim)).to(args.device)
            y = torch.randint(args.n_class, (n,)).to(args.device)
            fake_imgs = G(z, y).detach()
            _, aux_label = D(label_x)

            if args.gn:
                x_real_fake = torch.cat([imgs, fake_imgs], dim=0).requires_grad_(True)
                adv_f, aux = D(x_real_fake)
                pred = gradient_normalize(adv_f, x_real_fake)
                pred_real, pred_fake = torch.split(pred, [n, n])
                _, aux_fake = torch.split(aux, [n, n])
            else:
                pred_real, _ = D(imgs)
                pred_fake, aux_fake = D(fake_imgs)

            d_adv_loss = loss_fn(pred_real, pred_fake)
            dr_aux_loss = auxiliary_fn(aux_label, label_y)
            temp_loss = d_adv_loss + dr_aux_loss

            dr_aux_acc = torch.eq(aux_label.argmax(1), label_y).float().mean().item() # real aux acc
            if args.model == "sgan":
                # fake
                df_aux_loss = auxiliary_fn(aux_fake, y)
                df_aux_acc = torch.eq(aux_fake.argmax(1), y).float().mean().item()
                temp_loss += args.df_aux_wt * df_aux_loss

                # real and fake
                x_ = torch.cat([aux_label, aux_fake])
                y_ = torch.cat([label_y, y])
                d_rf_aux_acc = torch.eq(x_.argmax(1), y_).float().mean().item()
                d_rf_aux_loss = (dr_aux_loss + df_aux_loss).mean()

                e_dr_acc.update(dr_aux_acc)
                e_dr_loss.update(dr_aux_loss.item())
                e_df_acc.update(df_aux_acc)
                e_df_loss.update(df_aux_loss.item())
                e_train_acc.update(d_rf_aux_acc)
                e_train_loss.update(d_rf_aux_loss.item())
            else:
                e_train_acc.update(dr_aux_acc)
                e_train_loss.update(dr_aux_loss.item())

            e_dloss.update(temp_loss.item())
            temp_loss.backward()
            optimizer_d.step()

            """update G"""
            if (step % args.n_critic) == 0 or (step==len(train_dl)):
                count += 1
                optimizer_g.zero_grad()
                fake_imgs = G(z, y)
                adv_f, aux_fake = D(fake_imgs)

                if args.gn:
                    adv_f = gradient_normalize(adv_f, fake_imgs)

                temp_loss = loss_fn(adv_f)  # adversarial loss G

                if args.model == "sgan":
                    gf_aux_loss = auxiliary_fn(aux_fake, y)  # fake aux loss
                    gf_aux_acc = torch.eq(aux_fake.argmax(1), y).float().mean().item() # fake aux acc
                    temp_loss += gf_aux_loss
                    e_gf_acc.update(gf_aux_acc)
                    e_gf_loss.update(gf_aux_loss.item())

                if args.add_supconloss:
                    real_features = D(label_x, return_features=True)
                    fake_features = D(fake_imgs, return_features=True)
                    x_new = torch.cat([real_features, fake_features])
                    y_new = torch.cat([label_y, y])
                    supcontrast_loss = supcontrast_fn(x_new, y_new)
                    temp_loss = (1 - args.sup_wt) * temp_loss + args.sup_wt * supcontrast_loss
                    e_sploss.update(supcontrast_loss.item())

                e_gloss.update(temp_loss.item())
                temp_loss.backward()
                optimizer_g.step()

        '''valid phase'''
        D.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(args.device), yb.to(args.device)
                _,  output = D(xb)
                loss = auxiliary_fn(output, yb)
                acc = torch.eq(output.argmax(1), yb).float().mean().item()
                e_valid_loss.update(loss.item())
                e_valid_acc.update(acc)

        if args.model == "sgan":
            print(f"[{epoch}/{args.epochs}], dloss:{e_dloss.avg:.4f}, gloss:{e_gloss.avg:.4f}, "
                  f"dr_acc:{e_dr_acc.avg:.4f}, df_acc:{e_df_acc.avg:.4f}, gf_acc:{e_gf_acc.avg:.4f}, "
                  f"train_acc:{e_train_acc.avg:.4f}, valid_acc:{e_valid_acc.avg:.4f}")
        else:
            print(f"[{epoch}/{args.epochs}], dloss:{e_dloss.avg:.4f}, gloss:{e_gloss.avg:.4f}, "
                  f"train_acc:{e_train_acc.avg:.4f}, valid_acc:{e_valid_acc.avg:.4f}")

        if (epoch == 0) or (((epoch + 1) % args.save_epochs) == 0) or (epoch == (args.epochs - 1)):
            ckpt = {'G': G.state_dict(),'D': D.state_dict()}
            if e_valid_acc.avg > best_acc:
                best_acc = e_valid_acc.avg
                torch.save(ckpt, r'./best_gan_model.pth')


    '''test phase'''
    model_d = net_D[args.model](args.n_class, args.img_size, args.n_channel).to(args.device)
    ckpt = torch.load(r'./best_gan_model.pth')
    model_d.load_state_dict(ckpt['D'])
    test_acc = model_test(model_d, test_dl, args.device)
    return test_acc


if __name__ == "__main__":
    set_seed(2023)
    args = parse_args()
    test_acc = train(args)

